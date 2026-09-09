// DLSS Neural Rendering calls, isolated in a module the snippet will accept as a caller.
//
// The snippet resolves the module owning its return address and requires that module's path to contain
// "nvngx.dll" (the driver core is _nvngx.dll), rejecting anything else with FAIL_PlatformError before it
// inspects a single argument. Neither a ReShade add-on nor OptiScaler is named anything like that, so the
// calls are made from here instead and reached through the exports below.
//
// The parameter block is the core's capability block rather than a fresh one: it carries the snippet and
// preset callbacks a feature expects at create time. The core exports no Set/Get helpers (they are
// static-library inlines), so it is driven through its vtable. NVSDK_NGX_Parameter declares eight Set
// overloads then eight Get overloads, in this order: ULL, float, double, uint, int, ID3D11Resource*,
// ID3D12Resource*, void*.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

namespace {

// Slot indices confirmed by round-tripping values through the live block: a setter at N is read back by
// the getter at N+8. The unsigned setter is slot 3 (a feature create driven through it succeeds) and the
// resource getter answers at slot 8, so resources are written through slot 0 -- the 64-bit setter, which
// is what a resource handle is. Writing them through the typed D3D12 setter left them unset.
constexpr int VT_SET_ULL = 0;
// Where the float setter actually lives. The public header declares it at slot 1, and this block --
// the driver's own, not the header's implementation -- does not keep a float there: every float written
// to slot 1 reads back as FAIL_UnsupportedParameter while every uint lands. The host discovers the real
// slot by round-tripping a value and sets it here before anything else is written.
int g_floatSlot = 1;
constexpr int VT_SET_UINT = 3;

using PFN_SetULL = void(__thiscall *)(void *, const char *, unsigned long long);
using PFN_SetFloat = void(__thiscall *)(void *, const char *, float);
using PFN_SetUInt = void(__thiscall *)(void *, const char *, unsigned int);

void setUInt(void *params, const char *name, unsigned int v) {
    void **vt = *reinterpret_cast<void ***>(params);
    reinterpret_cast<PFN_SetUInt>(vt[VT_SET_UINT])(params, name, v);
}

void setFloat(void *params, const char *name, float v) {
    void **vt = *reinterpret_cast<void ***>(params);
    reinterpret_cast<PFN_SetFloat>(vt[g_floatSlot])(params, name, v);
}

void setResourcePtr(void *params, const char *name, void *v) {
    void **vt = *reinterpret_cast<void ***>(params);
    reinterpret_cast<PFN_SetULL>(vt[VT_SET_ULL])(params, name, (unsigned long long) v);
}

void setResource(void *params, const char *name, ID3D12Resource *v) {
    void **vt = *reinterpret_cast<void ***>(params);
    reinterpret_cast<PFN_SetULL>(vt[VT_SET_ULL])(params, name, (unsigned long long) v);
}

using PFN_NrInitExt = int(__cdecl *)(unsigned long long, const wchar_t *, ID3D12Device *, int,
                                     const void *);
// An explicit ControlMask takes precedence over the runtime's automatic mask.
// We do not supply one; clear it on the reusable parameter block instead of
// allowing another feature's stale mask/resource to silently override this toggle.
void setAutomaticMask(void *params, int enabled) {
    setResourcePtr(params, "DLSSNR.ControlMask", nullptr);
    setUInt(params, "DLSSNR.UseAutoMask", enabled != 0 ? 1u : 0u);
}
using PFN_NrCreate = int(__cdecl *)(ID3D12GraphicsCommandList *, int, const void *, void **);
using PFN_NrEvaluate = int(__cdecl *)(ID3D12GraphicsCommandList *, const void *, const void *, void *);
using PFN_NrRelease = int(__cdecl *)(void *);

struct Snippet {
    HMODULE module = nullptr;
    PFN_NrInitExt init = nullptr;
    PFN_NrCreate create = nullptr;
    PFN_NrEvaluate evaluate = nullptr;
    PFN_NrRelease release = nullptr;
    bool initialised = false;

};

Snippet g_snip;

bool loadSnippet(const wchar_t *path) {
    if (g_snip.module) {
        return g_snip.create != nullptr;
    }
    g_snip.module = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g_snip.module) {
        return false;
    }
    g_snip.init = (PFN_NrInitExt) GetProcAddress(g_snip.module, "NVSDK_NGX_D3D12_Init_Ext");
    g_snip.create = (PFN_NrCreate) GetProcAddress(g_snip.module, "NVSDK_NGX_D3D12_CreateFeature");
    g_snip.evaluate = (PFN_NrEvaluate) GetProcAddress(g_snip.module, "NVSDK_NGX_D3D12_EvaluateFeature");
    g_snip.release = (PFN_NrRelease) GetProcAddress(g_snip.module, "NVSDK_NGX_D3D12_ReleaseFeature");


    return g_snip.create != nullptr && g_snip.evaluate != nullptr;
}


} // namespace

extern "C" {

// Called once, after the host has worked out which slot this block keeps floats in.
__declspec(dllexport) void dlssnr_call_set_float_slot(int slot) {
    if (slot >= 0 && slot < 8) {
        g_floatSlot = slot;
    }
}

// Writes a float through an arbitrary slot, so the host can find the right one by testing.
__declspec(dllexport) void dlssnr_call_probe_float(void *params, const char *name, float value,
                                                   int slot) {
    if (!params || slot < 0 || slot >= 8) {
        return;
    }
    void **vt = *reinterpret_cast<void ***>(params);
    reinterpret_cast<PFN_SetFloat>(vt[slot])(params, name, value);
}

// Last init and create results, so the add-on can log why a feature never appeared.
__declspec(dllexport) int dlssnr_call_last_init = 0;
__declspec(dllexport) int dlssnr_call_last_create = 0;

// ---------------------------------------------------------------------------------------------
// Vulkan.
// ---------------------------------------------------------------------------------------------

using PFN_NrVkInitExt = int(__cdecl *)(unsigned long long, const wchar_t *, void *, void *, void *,
                                       const void *, int);
using PFN_NrVkCreate = int(__cdecl *)(void *, int, const void *, void **);
using PFN_NrVkEvaluate = int(__cdecl *)(void *, const void *, const void *, void *);

struct VkSnippet {
    HMODULE module = nullptr;
    PFN_NrVkInitExt init = nullptr;
    PFN_NrVkCreate create = nullptr;
    PFN_NrVkEvaluate evaluate = nullptr;
    PFN_NrRelease release = nullptr;
    bool initialised = false;
};

VkSnippet g_vk;

bool loadVkSnippet(const wchar_t *path) {
    if (g_vk.module) {
        return g_vk.create != nullptr;
    }

    g_vk.module = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);

    if (!g_vk.module) {
        return false;
    }

    g_vk.init = (PFN_NrVkInitExt) GetProcAddress(g_vk.module, "NVSDK_NGX_VULKAN_Init_Ext");
    g_vk.create = (PFN_NrVkCreate) GetProcAddress(g_vk.module, "NVSDK_NGX_VULKAN_CreateFeature");
    g_vk.evaluate = (PFN_NrVkEvaluate) GetProcAddress(g_vk.module, "NVSDK_NGX_VULKAN_EvaluateFeature");
    g_vk.release = (PFN_NrRelease) GetProcAddress(g_vk.module, "NVSDK_NGX_VULKAN_ReleaseFeature");

    return g_vk.create != nullptr && g_vk.evaluate != nullptr;
}

// ---------------------------------------------------------------------------------------------
// Scaling Ratio Query
// ---------------------------------------------------------------------------------------------

using PFN_NrPopulate = int(__cdecl *)(void *);
using PFN_NrRatioCallback = int(__cdecl *)(void *);

constexpr int VT_GET_ULL = VT_SET_ULL + 8;

using PFN_GetULL = int(__thiscall *)(void *, const char *, unsigned long long *);
using PFN_GetFloat = int(__thiscall *)(void *, const char *, float *);

__declspec(dllexport) int dlssnr_last_ratio_result = 0;
__declspec(dllexport) int dlssnr_last_ratio_stage = 0;

__declspec(dllexport) int dlssnr_query_scaling_ratio(const wchar_t *snippetPath, void *capabilityParams,
                                                     unsigned int perfQuality, float *outRatio) {
    dlssnr_last_ratio_stage = 0;

    if (!loadSnippet(snippetPath) || !capabilityParams || !outRatio) {
        return 0;
    }

    dlssnr_last_ratio_stage = 1;

    auto populate = (PFN_NrPopulate) GetProcAddress(g_snip.module, "NVSDK_NGX_D3D12_PopulateParameters_Impl");

    if (populate != nullptr) {
        volatile int populated = populate(capabilityParams);
        (void) populated;
        dlssnr_last_ratio_stage = 2;
    }

    void **vt = *reinterpret_cast<void ***>(capabilityParams);
    unsigned long long raw = 0;

    if (reinterpret_cast<PFN_GetULL>(vt[VT_GET_ULL])(capabilityParams, "DLSSNRComputeScalingRatioCallback",
                                                     &raw) != 1 ||
        raw == 0) {
        return 0;
    }

    dlssnr_last_ratio_stage = 3;

    setUInt(capabilityParams, "PerfQualityValue", perfQuality);
    setFloat(capabilityParams, "DLSSNR.ScalingRatio", -1.0f);

    volatile int result = reinterpret_cast<PFN_NrRatioCallback>((void *) raw)(capabilityParams);
    dlssnr_last_ratio_result = (int) result;

    if (result != 1) {
        return -1;
    }

    dlssnr_last_ratio_stage = 4;

    float ratio = -1.0f;

    if (reinterpret_cast<PFN_GetFloat>(vt[g_floatSlot + 8])(capabilityParams, "DLSSNR.ScalingRatio",
                                                            &ratio) != 1) {
        return -1;
    }

    dlssnr_last_ratio_stage = 5;
    *outRatio = ratio;
    return 1;
}

// ---------------------------------------------------------------------------------------------
// Native Direct3D 11
// ---------------------------------------------------------------------------------------------

using PFN_NrD3D11Init = int(__cdecl *)(unsigned long long, const wchar_t *, void *, int, const void *);
using PFN_NrD3D11Create = int(__cdecl *)(void *, int, const void *, void **);
using PFN_NrD3D11Evaluate = int(__cdecl *)(void *, const void *, const void *, void *);

struct D3D11Snippet {
    HMODULE module = nullptr;
    PFN_NrD3D11Init init = nullptr;
    PFN_NrD3D11Create create = nullptr;
    PFN_NrD3D11Evaluate evaluate = nullptr;
    PFN_NrRelease release = nullptr;
    bool initialised = false;
};

D3D11Snippet g_d3d11;

bool loadD3D11Snippet(const wchar_t *path) {
    if (g_d3d11.module) {
        return g_d3d11.create != nullptr;
    }

    wchar_t probeCopy[MAX_PATH] = {};
    wchar_t probeDir[MAX_PATH] = {};

    if (GetTempPathW(MAX_PATH, probeDir) == 0) {
        return false;
    }

    wcscpy_s(probeCopy, probeDir);
    wcscat_s(probeCopy, L"nvngx.dll_dlssnr_d3d11probe.dll");

    if (!CopyFileW(path, probeCopy, FALSE)) {
        return false;
    }

    g_d3d11.module = LoadLibraryExW(probeCopy, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);

    if (!g_d3d11.module) {
        return false;
    }

    g_d3d11.init = (PFN_NrD3D11Init) GetProcAddress(g_d3d11.module, "NVSDK_NGX_D3D11_Init_Ext");
    g_d3d11.create = (PFN_NrD3D11Create) GetProcAddress(g_d3d11.module, "NVSDK_NGX_D3D11_CreateFeature");
    g_d3d11.evaluate = (PFN_NrD3D11Evaluate) GetProcAddress(g_d3d11.module, "NVSDK_NGX_D3D11_EvaluateFeature");
    g_d3d11.release = (PFN_NrRelease) GetProcAddress(g_d3d11.module, "NVSDK_NGX_D3D11_ReleaseFeature");

    return g_d3d11.create != nullptr && g_d3d11.evaluate != nullptr;
}

using PFN_NrD3D11Requirements = int(__cdecl *)(const void *, const void *, void *);

__declspec(dllexport) int dlssnr_d3d11_requirements(const wchar_t *snippetPath, void *adapter,
                                                    unsigned int *outFlags, unsigned int *outArch,
                                                    unsigned int *outOsVersion) {
    if (!loadD3D11Snippet(snippetPath)) {
        return -2;
    }

    auto req = (PFN_NrD3D11Requirements) GetProcAddress(g_d3d11.module,
                                                        "NVSDK_NGX_D3D11_GetFeatureRequirements");

    if (req == nullptr) {
        return -1;
    }

    struct {
        int sdkVersion;
        int featureId;
        struct { int idType; unsigned long long appId; } identifier;
        const wchar_t *dataPath;
        const void *loggingInfo;
        const void *featureInfo;
    } discovery {};

    discovery.sdkVersion = 0x0000015;
    discovery.featureId = 18;
    discovery.identifier.idType = 0;
    discovery.identifier.appId = 0x24480451ull;
    discovery.dataPath = L".";

    unsigned char requirement[512] = {};
    volatile int result = req(adapter, &discovery, requirement);

    const unsigned int *out = reinterpret_cast<const unsigned int *>(requirement);

    if (outFlags != nullptr) {
        *outFlags = out[0];
    }

    if (outArch != nullptr) {
        *outArch = out[1];
    }

    if (outOsVersion != nullptr) {
        *outOsVersion = out[2];
    }

    return (int) result;
}

__declspec(dllexport) int dlssnr_d3d11_last_init = 0;
__declspec(dllexport) int dlssnr_d3d11_last_create = 0;

__declspec(dllexport) int dlssnr_d3d11_probe(const wchar_t *snippetPath) {
    loadD3D11Snippet(snippetPath);

    int bits = 0;
    bits |= g_d3d11.init != nullptr ? 1 : 0;
    bits |= g_d3d11.create != nullptr ? 2 : 0;
    bits |= g_d3d11.evaluate != nullptr ? 4 : 0;
    bits |= g_d3d11.release != nullptr ? 8 : 0;

    return bits;
}

using PFN_NrD3D11InitPlain = int(__cdecl *)(unsigned long long, const wchar_t *, void *, const void *, int);

__declspec(dllexport) int dlssnr_d3d11_init(const wchar_t *snippetPath, const wchar_t *dataPath,
                                            void *device, int sdkVersion, int *attemptOut,
                                            int *resultsOut) {
    if (device == nullptr) {
        return -1;
    }

    if (g_d3d11.initialised) {
        if (attemptOut != nullptr) *attemptOut = 0;
        return 1;
    }

    const bool haveCopy = loadD3D11Snippet(snippetPath);
    const bool haveShared = loadSnippet(snippetPath);

    struct Attempt { HMODULE module; bool ext; };

    const Attempt attempts[4] = {
        { haveCopy ? g_d3d11.module : nullptr, true },
        { haveCopy ? g_d3d11.module : nullptr, false },
        { haveShared ? g_snip.module : nullptr, true },
        { haveShared ? g_snip.module : nullptr, false },
    };

    int last = -1;

    for (int i = 0; i < 4; ++i) {
        if (attempts[i].module == nullptr) {
            if (resultsOut != nullptr) resultsOut[i] = -2;
            continue;
        }

        volatile int result = 0;

        if (attempts[i].ext) {
            auto fn = (PFN_NrD3D11Init) GetProcAddress(attempts[i].module, "NVSDK_NGX_D3D11_Init_Ext");

            if (fn == nullptr) {
                if (resultsOut != nullptr) resultsOut[i] = -3;
                continue;
            }

            result = fn(0x24480451ull, dataPath, device, sdkVersion, nullptr);
        } else {
            auto fn = (PFN_NrD3D11InitPlain) GetProcAddress(attempts[i].module, "NVSDK_NGX_D3D11_Init");

            if (fn == nullptr) {
                if (resultsOut != nullptr) resultsOut[i] = -3;
                continue;
            }

            result = fn(0x24480451ull, dataPath, device, nullptr, sdkVersion);
        }

        last = (int) result;

        if (resultsOut != nullptr) resultsOut[i] = last;

        if (last == 1) {
            g_d3d11.module = attempts[i].module;
            g_d3d11.create = (PFN_NrD3D11Create) GetProcAddress(attempts[i].module,
                                                                "NVSDK_NGX_D3D11_CreateFeature");
            g_d3d11.evaluate = (PFN_NrD3D11Evaluate) GetProcAddress(attempts[i].module,
                                                                    "NVSDK_NGX_D3D11_EvaluateFeature");
            g_d3d11.release = (PFN_NrRelease) GetProcAddress(attempts[i].module,
                                                             "NVSDK_NGX_D3D11_ReleaseFeature");
            g_d3d11.initialised = true;

            if (attemptOut != nullptr) *attemptOut = i + 1;

            dlssnr_d3d11_last_init = last;
            return last;
        }
    }

    if (attemptOut != nullptr) *attemptOut = 0;

    dlssnr_d3d11_last_init = last;
    return last;
}

__declspec(dllexport) void *dlssnr_d3d11_create(void *deviceContext, void *capabilityParams,
                                                unsigned int width, unsigned int height, int preset,
                                                float intensity, int style, float localStructure,
                                                float localTone, float skinStructure, int useAutoMask,
                                                int uiCorrection) {
    if (g_d3d11.create == nullptr || deviceContext == nullptr || capabilityParams == nullptr) {
        return nullptr;
    }

    setUInt(capabilityParams, "DLSSNR.Enabled", 1);
    setUInt(capabilityParams, "DLSSNR.Width", width);
    setUInt(capabilityParams, "DLSSNR.Height", height);
    setUInt(capabilityParams, "CreationNodeMask", 1);
    setUInt(capabilityParams, "VisibilityNodeMask", 1);
    setUInt(capabilityParams, "DLSSNR.Hint.Render.Preset", (unsigned int) preset);

    setFloat(capabilityParams, "DLSSNR.Intensity", intensity);
    setUInt(capabilityParams, "DLSSNR.Style", (unsigned int) style);
    setFloat(capabilityParams, "DLSSNR.LocalStructureStrength", localStructure);
    setFloat(capabilityParams, "DLSSNR.LocalToneStrength", localTone);
    setFloat(capabilityParams, "DLSSNR.SkinStructureStrength", skinStructure);
    setAutomaticMask(capabilityParams, useAutoMask);
    setUInt(capabilityParams, "DLSSNR.UICorrection", (unsigned int) uiCorrection);

    void *feature = nullptr;
    volatile int result = g_d3d11.create(deviceContext, 18, capabilityParams, &feature);

    dlssnr_d3d11_last_create = (int) result;

    return result == 1 ? feature : nullptr;
}

__declspec(dllexport) void dlssnr_d3d11_release(void *feature) {
    if (g_d3d11.release != nullptr && feature != nullptr) {
        volatile int ignored = g_d3d11.release(feature);
        (void) ignored;
    }
}

__declspec(dllexport) int dlssnr_vk_last_init = 0;
__declspec(dllexport) int dlssnr_vk_last_create = 0;

__declspec(dllexport) int dlssnr_vk_probe(const wchar_t *snippetPath) {
    loadVkSnippet(snippetPath);

    int bits = 0;
    bits |= g_vk.init != nullptr ? 1 : 0;
    bits |= g_vk.create != nullptr ? 2 : 0;
    bits |= g_vk.evaluate != nullptr ? 4 : 0;
    bits |= g_vk.release != nullptr ? 8 : 0;

    return bits;
}

__declspec(dllexport) int dlssnr_vk_init(const wchar_t *snippetPath, const wchar_t *dataPath,
                                         void *instance, void *physicalDevice, void *device,
                                         int sdkVersion) {
    if (!loadVkSnippet(snippetPath) || g_vk.init == nullptr) {
        return -1;
    }

    if (g_vk.initialised) {
        return 1;
    }

    volatile int result = g_vk.init(0x0, dataPath, instance, physicalDevice, device, nullptr, sdkVersion);

    dlssnr_vk_last_init = (int) result;
    g_vk.initialised = result == 1;

    return (int) result;
}

__declspec(dllexport) void *dlssnr_vk_create(void *cmdBuffer, void *capabilityParams, unsigned int width,
                                             unsigned int height, int preset, float intensity, int style,
                                             float localStructure, float localTone, float skinStructure,
                                             int useAutoMask, int uiCorrection) {
    if (g_vk.create == nullptr || cmdBuffer == nullptr || capabilityParams == nullptr) {
        return nullptr;
    }

    setUInt(capabilityParams, "DLSSNR.Enabled", 1);
    setUInt(capabilityParams, "DLSSNR.Width", width);
    setUInt(capabilityParams, "DLSSNR.Height", height);
    setUInt(capabilityParams, "CreationNodeMask", 1);
    setUInt(capabilityParams, "VisibilityNodeMask", 1);
    setUInt(capabilityParams, "DLSSNR.Hint.Render.Preset", (unsigned int) preset);

    setFloat(capabilityParams, "DLSSNR.Intensity", intensity);
    setUInt(capabilityParams, "DLSSNR.Style", (unsigned int) style);
    setFloat(capabilityParams, "DLSSNR.LocalStructureStrength", localStructure);
    setFloat(capabilityParams, "DLSSNR.LocalToneStrength", localTone);
    setFloat(capabilityParams, "DLSSNR.SkinStructureStrength", skinStructure);
    setAutomaticMask(capabilityParams, useAutoMask);
    setUInt(capabilityParams, "DLSSNR.UICorrection", (unsigned int) uiCorrection);

    void *feature = nullptr;
    volatile int result = g_vk.create(cmdBuffer, 18, capabilityParams, &feature);

    dlssnr_vk_last_create = (int) result;

    return result == 1 ? feature : nullptr;
}

// Evaluates on Vulkan with Motion Vector Fix & Automatic Mask
__declspec(dllexport) int dlssnr_vk_evaluate(void *cmdBuffer, void *feature, void *capabilityParams,
                                             void *color, void *depth, void *motion, void *output,
                                             unsigned int width, unsigned int height,
                                             unsigned int guideWidth, unsigned int guideHeight,
                                             unsigned int motionWidth, unsigned int motionHeight,
                                             unsigned int depthBaseX, unsigned int depthBaseY,
                                             unsigned int motionBaseX, unsigned int motionBaseY,
                                             int depthInverted, int reset, float intensity, int style,
                                             float localStructure, float localTone, float skinStructure,
                                             int useAutoMask, float mvScaleX, float mvScaleY) {
    if (g_vk.evaluate == nullptr || feature == nullptr || capabilityParams == nullptr) {
        return -1;
    }

    setResourcePtr(capabilityParams, "DLSSNR.Color", color);
    setResourcePtr(capabilityParams, "DLSSNR.Depth", depth);
    setResourcePtr(capabilityParams, "DLSSNR.MVec", motion);
    setResourcePtr(capabilityParams, "DLSSNR.Output", output);

    setUInt(capabilityParams, "DLSSNR.Enabled", 1);
    setUInt(capabilityParams, "DLSSNR.Width", width);
    setUInt(capabilityParams, "DLSSNR.Height", height);
    setUInt(capabilityParams, "DLSSNR.DepthInverted", (unsigned int) depthInverted);
    setUInt(capabilityParams, "DLSSNR.Reset", (unsigned int) reset);

    setUInt(capabilityParams, "DLSSNR.ColorSubrectBaseX", 0);
    setUInt(capabilityParams, "DLSSNR.ColorSubrectBaseY", 0);
    setUInt(capabilityParams, "DLSSNR.ColorSubrectWidth", width);
    setUInt(capabilityParams, "DLSSNR.ColorSubrectHeight", height);
    setUInt(capabilityParams, "DLSSNR.OutputSubrectBaseX", 0);
    setUInt(capabilityParams, "DLSSNR.OutputSubrectBaseY", 0);
    setUInt(capabilityParams, "DLSSNR.OutputSubrectWidth", width);
    setUInt(capabilityParams, "DLSSNR.OutputSubrectHeight", height);
    setUInt(capabilityParams, "DLSSNR.DepthSubrectBaseX", depthBaseX);
    setUInt(capabilityParams, "DLSSNR.DepthSubrectBaseY", depthBaseY);
    setUInt(capabilityParams, "DLSSNR.DepthSubrectWidth", guideWidth);
    setUInt(capabilityParams, "DLSSNR.DepthSubrectHeight", guideHeight);
    setUInt(capabilityParams, "DLSSNR.MVecSubrectBaseX", motionBaseX);
    setUInt(capabilityParams, "DLSSNR.MVecSubrectBaseY", motionBaseY);
    setUInt(capabilityParams, "DLSSNR.MVecSubrectWidth", motionWidth);
    setUInt(capabilityParams, "DLSSNR.MVecSubrectHeight", motionHeight);

    setFloat(capabilityParams, "DLSSNR.MVecScaleX", mvScaleX);
    setFloat(capabilityParams, "DLSSNR.MVecScaleY", mvScaleY);

    setFloat(capabilityParams, "DLSSNR.Intensity", intensity);
    setUInt(capabilityParams, "DLSSNR.Style", (unsigned int) style);
    setFloat(capabilityParams, "DLSSNR.LocalStructureStrength", localStructure);
    setFloat(capabilityParams, "DLSSNR.LocalToneStrength", localTone);
    setFloat(capabilityParams, "DLSSNR.SkinStructureStrength", skinStructure);
    setAutomaticMask(capabilityParams, useAutoMask);

    volatile int result = g_vk.evaluate(cmdBuffer, feature, capabilityParams, nullptr);

    return (int) result;
}

__declspec(dllexport) void dlssnr_vk_release(void *feature) {
    if (g_vk.release != nullptr && feature != nullptr) {
        volatile int ignored = g_vk.release(feature);
        (void) ignored;
    }
}

__declspec(dllexport) void *dlssnr_call_create(const wchar_t *snippetPath, const wchar_t *dataPath,
                                               ID3D12Device *device, ID3D12GraphicsCommandList *cmd,
                                               void *capabilityParams, unsigned int width,
                                               unsigned int height, int preset, float intensity,
                                               int style, float localStructure, float localTone,
                                               float skinStructure, int useAutoMask,
                                               int uiCorrection) {
    if (!loadSnippet(snippetPath) || !capabilityParams) {
        return nullptr;
    }
    if (!g_snip.initialised && g_snip.init) {
        dlssnr_call_last_init = g_snip.init(0x24480451ull, dataPath, device, 0x0000015, capabilityParams);
        g_snip.initialised = (dlssnr_call_last_init == 1);
        if (!g_snip.initialised) {
            return nullptr;
        }
    }
    setUInt(capabilityParams, "DLSSNR.Enabled", 1);
    setUInt(capabilityParams, "DLSSNR.Width", width);
    setUInt(capabilityParams, "DLSSNR.Height", height);
    setUInt(capabilityParams, "CreationNodeMask", 1);
    setUInt(capabilityParams, "VisibilityNodeMask", 1);
    setUInt(capabilityParams, "DLSSNR.Hint.Render.Preset", (unsigned int) preset);

    setFloat(capabilityParams, "DLSSNR.Intensity", intensity);
    setUInt(capabilityParams, "DLSSNR.Style", (unsigned int) style);
    setFloat(capabilityParams, "DLSSNR.LocalStructureStrength", localStructure);
    setFloat(capabilityParams, "DLSSNR.LocalToneStrength", localTone);
    setFloat(capabilityParams, "DLSSNR.SkinStructureStrength", skinStructure);
    setAutomaticMask(capabilityParams, useAutoMask);
    setUInt(capabilityParams, "DLSSNR.UICorrection", (unsigned int) uiCorrection);
    void *handle = nullptr;
    dlssnr_call_last_create = g_snip.create(cmd, 18, capabilityParams, &handle);
    return dlssnr_call_last_create == 1 ? handle : nullptr;
}

// Evaluates on D3D12 with Motion Vector Fix & Automatic Mask
__declspec(dllexport) int dlssnr_call_evaluate(ID3D12GraphicsCommandList *cmd, void *feature,
                                               void *capabilityParams, ID3D12Resource *color,
                                               ID3D12Resource *depth, ID3D12Resource *motion,
                                               ID3D12Resource *output, unsigned int width,
                                               unsigned int height, unsigned int guideWidth,
                                               unsigned int guideHeight, unsigned int motionWidth,
                                               unsigned int motionHeight, unsigned int depthBaseX,
                                               unsigned int depthBaseY, unsigned int motionBaseX,
                                               unsigned int motionBaseY, int depthInverted, int reset,
                                               float intensity, int style, float localStructure,
                                               float localTone, float skinStructure, int useAutoMask,
                                               float mvScaleX, float mvScaleY) {
    if (!feature || !capabilityParams || !g_snip.evaluate) {
        return 0;
    }
    setResource(capabilityParams, "DLSSNR.Color", color);
    setResource(capabilityParams, "DLSSNR.Depth", depth);
    setResource(capabilityParams, "DLSSNR.MVec", motion);
    setResource(capabilityParams, "DLSSNR.Output", output);

    setUInt(capabilityParams, "DLSSNR.Enabled", 1);
    setUInt(capabilityParams, "DLSSNR.Width", width);
    setUInt(capabilityParams, "DLSSNR.Height", height);
    setUInt(capabilityParams, "DLSSNR.DepthInverted", (unsigned int) depthInverted);
    setUInt(capabilityParams, "DLSSNR.Reset", (unsigned int) reset);

    setUInt(capabilityParams, "DLSSNR.ColorSubrectBaseX", 0);
    setUInt(capabilityParams, "DLSSNR.ColorSubrectBaseY", 0);
    setUInt(capabilityParams, "DLSSNR.ColorSubrectWidth", width);
    setUInt(capabilityParams, "DLSSNR.ColorSubrectHeight", height);
    setUInt(capabilityParams, "DLSSNR.OutputSubrectBaseX", 0);
    setUInt(capabilityParams, "DLSSNR.OutputSubrectBaseY", 0);
    setUInt(capabilityParams, "DLSSNR.OutputSubrectWidth", width);
    setUInt(capabilityParams, "DLSSNR.OutputSubrectHeight", height);
    setUInt(capabilityParams, "DLSSNR.DepthSubrectBaseX", depthBaseX);
    setUInt(capabilityParams, "DLSSNR.DepthSubrectBaseY", depthBaseY);
    setUInt(capabilityParams, "DLSSNR.DepthSubrectWidth", guideWidth);
    setUInt(capabilityParams, "DLSSNR.DepthSubrectHeight", guideHeight);
    setUInt(capabilityParams, "DLSSNR.MVecSubrectBaseX", motionBaseX);
    setUInt(capabilityParams, "DLSSNR.MVecSubrectBaseY", motionBaseY);
    setUInt(capabilityParams, "DLSSNR.MVecSubrectWidth", motionWidth);
    setUInt(capabilityParams, "DLSSNR.MVecSubrectHeight", motionHeight);

    setFloat(capabilityParams, "DLSSNR.MVecScaleX", mvScaleX);
    setFloat(capabilityParams, "DLSSNR.MVecScaleY", mvScaleY);

    setFloat(capabilityParams, "DLSSNR.Intensity", intensity);
    setUInt(capabilityParams, "DLSSNR.Style", (unsigned int) style);
    setFloat(capabilityParams, "DLSSNR.LocalStructureStrength", localStructure);
    setFloat(capabilityParams, "DLSSNR.LocalToneStrength", localTone);
    setFloat(capabilityParams, "DLSSNR.SkinStructureStrength", skinStructure);
    setAutomaticMask(capabilityParams, useAutoMask);

    volatile int result = g_snip.evaluate(cmd, feature, capabilityParams, nullptr);
    return result;
}

__declspec(dllexport) void dlssnr_call_set_extras(void *capabilityParams, float globalTone,
                                                  ID3D12Resource *ui, ID3D12Resource *uiAlpha,
                                                  ID3D12Resource *backbuffer, unsigned int uiWidth,
                                                  unsigned int uiHeight, unsigned int bbWidth,
                                                  unsigned int bbHeight) {
    if (!capabilityParams) {
        return;
    }
    (void) globalTone;

    setResource(capabilityParams, "DLSSNR.UI", ui);
    setResource(capabilityParams, "DLSSNR.UIAlpha", uiAlpha);
    setResource(capabilityParams, "DLSSNR.Backbuffer", backbuffer);
    setUInt(capabilityParams, "DLSSNR.UISubrectBaseX", 0);
    setUInt(capabilityParams, "DLSSNR.UISubrectBaseY", 0);
    setUInt(capabilityParams, "DLSSNR.UISubrectWidth", uiWidth);
    setUInt(capabilityParams, "DLSSNR.UISubrectHeight", uiHeight);
    setUInt(capabilityParams, "DLSSNR.UIAlphaSubrectBaseX", 0);
    setUInt(capabilityParams, "DLSSNR.UIAlphaSubrectBaseY", 0);
    setUInt(capabilityParams, "DLSSNR.UIAlphaSubrectWidth", uiWidth);
    setUInt(capabilityParams, "DLSSNR.UIAlphaSubrectHeight", uiHeight);
    setUInt(capabilityParams, "DLSSNR.BackbufferSubrectBaseX", 0);
    setUInt(capabilityParams, "DLSSNR.BackbufferSubrectBaseY", 0);
    setUInt(capabilityParams, "DLSSNR.BackbufferSubrectWidth", bbWidth);
    setUInt(capabilityParams, "DLSSNR.BackbufferSubrectHeight", bbHeight);
}

__declspec(dllexport) void dlssnr_call_release(void *feature) {
    if (feature && g_snip.release) {
        volatile int result = g_snip.release(feature);
        (void) result;
    }
}

} // extern "C"