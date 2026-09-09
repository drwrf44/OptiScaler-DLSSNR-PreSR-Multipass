#include "pch.h"

#include "DlssNrFeature_Vk.h"
#include "DlssNrFeature_Dx12.h"
#include "PassProfiles.h"

#include <Config.h>
#include <State.h>
#include <Util.h>
#include <NVNGX_Parameter.h>

#include <shaders/dlssnr/DlssNr_Vk.h>
#include <shaders/output_scaling/OS_Vk.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

namespace DlssNr
{

namespace
{

// The forwarder's Vulkan surface. The model checks its caller's module path and requires nvngx.dll in
// it, whichever API is being used, so these calls go through the same shim the D3D12 path does.
using PFN_VkProbe = int(__cdecl*)(const wchar_t*);
using PFN_VkInit = int(__cdecl*)(const wchar_t*, const wchar_t*, void*, void*, void*, int);
using PFN_VkCreate = void*(__cdecl*)(void*, void*, unsigned int, unsigned int, int, float, int, float, float, float,
                                     int, int);
// Updated to 27 parameters to support independent Motion Vector / Depth subrects (Motion Vector Fix)
using PFN_VkEvaluate = int(__cdecl*)(void*, void*, void*, void*, void*, void*, void*, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, int, int, float, int, float, float, float, int, float,
                                     float);
using PFN_VkRelease = void(__cdecl*)(void*);

// One image this pass owns: the storage, the view, and the NGX wrapper that describes it. Kept
// together because they are created, resized and destroyed as one thing.
struct OwnedImage
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    NVSDK_NGX_Resource_VK ngx {};
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;

    bool Valid() const { return image != VK_NULL_HANDLE && view != VK_NULL_HANDLE; }
};

struct VkState
{
    bool failed = false;
    const char* reason = "";

    HMODULE forwarder = nullptr;
    PFN_VkProbe probe = nullptr;
    PFN_VkInit init = nullptr;
    PFN_VkCreate create = nullptr;
    PFN_VkEvaluate evaluate = nullptr;
    PFN_VkRelease release = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    bool ngxInitialised = false;
    void* feature = nullptr;
    void* laterFeatures[DlssNr::MaxPassCount] {};
    Profiles::NrPassTuning builtTuning[DlssNr::MaxPassCount] {};
    unsigned int builtPreset[DlssNr::MaxPassCount] {};
    unsigned int builtStyle[DlssNr::MaxPassCount] {};
    unsigned int activePasses = 0;
    VkEvent creationReady = VK_NULL_HANDLE;
    bool creationPending = false;
    NVSDK_NGX_Parameter* capabilityParams = nullptr;

    // What the model writes, the proxy it is shown, and the frame as the upscaler left it.
    OwnedImage output;
    OwnedImage scratch;
    OwnedImage proxy;
    OwnedImage keep;
    OwnedImage preColor;
    bool beforeSr = false;

    // The proxy at the model's working size, when that is below the frame. The model -- 98% of the
    // cost -- then runs on this instead of the full proxy, which is the whole point of the working
    // scale slider. Unused (and never created) at scale 1, so the default path is unchanged.
    OwnedImage proxySmall;

    // Supersampling (working scale > 1): the model runs above native, superUp enlarges the proxy to
    // that size and superDown averages the answer (output) back into outputNative at native for a 1:1
    // composite. nrScaler is the filter both were built with, so a changed DlssNrScalingDownscaler
    // rebuilds them. Unused and never created at scale <= 1.
    OwnedImage outputNative;
    std::unique_ptr<OS_Vk> superUp;
    std::unique_ptr<OS_Vk> superDown;
    Scaler nrScaler = Scaler::Count;

    std::unique_ptr<DlssNr_Vk> pass;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t workWidth = 0;
    uint32_t workHeight = 0;
    bool reset = true;
    unsigned long long frames = 0;

    // Timing. A pair of timestamps per frame across a ring, read back three frames later: a query
    // read the frame it was written stalls the CPU on the GPU, which would cost more than the pass
    // it is measuring. Vulkan reports ticks, and timestampPeriod is how many nanoseconds a tick is.
    VkQueryPool queryPool = VK_NULL_HANDLE;
    float timestampPeriod = 0.0f;
    unsigned long long timedFrames = 0;
    std::optional<double> lastGpuTime;

    // Whether the game hands over an exposure texture, and what it said when it did.
    bool exposureOffered = false;

    // The game's own exposure, read off its 1x1 texture, and the scale it multiplied its buffer by.
    //
    // gameExposure holds its last good value rather than resetting when a frame arrives without a
    // texture: GTA V dropped it three times in one session on the D3D12 path, and falling back to a
    // default on those frames is a flicker, not a fallback.
    float gameExposure = 0.0f;
    float gamePreExposure = 1.0f;

    // The exposure's courier: an 8x8 R32_FLOAT image the meter writes, and a ring of host-visible
    // buffers it is copied into. Only texel (0,0) is ever read -- the rest of the grid belongs to the
    // frame-statistics meter that was removed from the shared shader, and 8x8 is here only so that a
    // single 8x8 thread group lands entirely inside the image.
    OwnedImage meter;
    VkBuffer meterReadback[4] = {};
    VkDeviceMemory meterReadbackMemory[4] = {};
    void* meterMapped[4] = {};
    unsigned long long meterFrames = 0;
};

constexpr uint32_t kMeterSide = 8;
constexpr VkDeviceSize kMeterBytes = kMeterSide * kMeterSide * sizeof(float);
constexpr unsigned long long kMeterSlots = 4;
constexpr uint32_t kTimingSlots = 4;

VkState g_vk;
std::mutex g_vkMutex;

void Fail(const char* why)
{
    if (g_vk.failed)
        return;

    g_vk.failed = true;
    g_vk.reason = why;
    LOG_ERROR("DLSS-NR Vulkan unavailable: {}", why);
}

void DestroyImage(OwnedImage& img)
{
    if (g_vk.device == VK_NULL_HANDLE)
        return;

    if (img.view != VK_NULL_HANDLE)
        vkDestroyImageView(g_vk.device, img.view, nullptr);

    if (img.image != VK_NULL_HANDLE)
        vkDestroyImage(g_vk.device, img.image, nullptr);

    if (img.memory != VK_NULL_HANDLE)
        vkFreeMemory(g_vk.device, img.memory, nullptr);

    img = OwnedImage {};
}

uint32_t FindMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProps {};
    vkGetPhysicalDeviceMemoryProperties(g_vk.physicalDevice, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }

    return UINT32_MAX;
}

static VkImageInfo ImageInfoOf(const OwnedImage& img)
{
    VkImageInfo info {};
    info.ImageView = img.view;
    info.Image = img.image;
    info.SubresourceRange = VkImageSubresourceRange { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    info.Format = img.format;
    info.Width = img.width;
    info.Height = img.height;
    return info;
}

bool CreateImage(OwnedImage& img, uint32_t width, uint32_t height, VkFormat format, bool readWrite)
{
    DestroyImage(img);

    VkImageCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = { width, height, 1 };
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(g_vk.device, &info, nullptr, &img.image) != VK_SUCCESS)
    {
        LOG_ERROR("DLSS-NR Vulkan: could not create a {}x{} image", width, height);
        return false;
    }

    VkMemoryRequirements req {};
    vkGetImageMemoryRequirements(g_vk.device, img.image, &req);

    VkMemoryAllocateInfo alloc {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = FindMemoryTypeIndex(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (alloc.memoryTypeIndex == UINT32_MAX ||
        vkAllocateMemory(g_vk.device, &alloc, nullptr, &img.memory) != VK_SUCCESS ||
        vkBindImageMemory(g_vk.device, img.image, img.memory, 0) != VK_SUCCESS)
    {
        LOG_ERROR("DLSS-NR Vulkan: could not back a {}x{} image", width, height);
        DestroyImage(img);
        return false;
    }

    VkImageViewCreateInfo view {};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = img.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    if (vkCreateImageView(g_vk.device, &view, nullptr, &img.view) != VK_SUCCESS)
    {
        LOG_ERROR("DLSS-NR Vulkan: could not view a {}x{} image", width, height);
        DestroyImage(img);
        return false;
    }

    img.width = width;
    img.height = height;
    img.format = format;
    img.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    img.ngx.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
    img.ngx.Resource.ImageViewInfo.ImageView = img.view;
    img.ngx.Resource.ImageViewInfo.Image = img.image;
    img.ngx.Resource.ImageViewInfo.SubresourceRange = view.subresourceRange;
    img.ngx.Resource.ImageViewInfo.Format = format;
    img.ngx.Resource.ImageViewInfo.Width = width;
    img.ngx.Resource.ImageViewInfo.Height = height;
    img.ngx.ReadWrite = readWrite;

    return true;
}

bool CreateMeterReadback()
{
    for (unsigned long long i = 0; i < kMeterSlots; ++i)
    {
        if (g_vk.meterReadback[i] != VK_NULL_HANDLE)
            continue;

        VkBufferCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = kMeterBytes;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(g_vk.device, &info, nullptr, &g_vk.meterReadback[i]) != VK_SUCCESS)
        {
            LOG_WARN("DLSS-NR Vulkan: could not create the exposure readback buffer");
            return false;
        }

        VkMemoryRequirements req {};
        vkGetBufferMemoryRequirements(g_vk.device, g_vk.meterReadback[i], &req);

        VkMemoryAllocateInfo alloc {};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = FindMemoryTypeIndex(
            req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (alloc.memoryTypeIndex == UINT32_MAX ||
            vkAllocateMemory(g_vk.device, &alloc, nullptr, &g_vk.meterReadbackMemory[i]) != VK_SUCCESS ||
            vkBindBufferMemory(g_vk.device, g_vk.meterReadback[i], g_vk.meterReadbackMemory[i], 0) != VK_SUCCESS ||
            vkMapMemory(g_vk.device, g_vk.meterReadbackMemory[i], 0, kMeterBytes, 0, &g_vk.meterMapped[i]) !=
                VK_SUCCESS)
        {
            LOG_WARN("DLSS-NR Vulkan: could not back the exposure readback buffer");
            return false;
        }
    }

    return true;
}

void DestroyMeterReadback()
{
    for (unsigned long long i = 0; i < kMeterSlots; ++i)
    {
        if (g_vk.meterReadbackMemory[i] != VK_NULL_HANDLE)
        {
            if (g_vk.meterMapped[i] != nullptr)
                vkUnmapMemory(g_vk.device, g_vk.meterReadbackMemory[i]);

            vkFreeMemory(g_vk.device, g_vk.meterReadbackMemory[i], nullptr);
        }

        if (g_vk.meterReadback[i] != VK_NULL_HANDLE)
            vkDestroyBuffer(g_vk.device, g_vk.meterReadback[i], nullptr);

        g_vk.meterMapped[i] = nullptr;
        g_vk.meterReadbackMemory[i] = VK_NULL_HANDLE;
        g_vk.meterReadback[i] = VK_NULL_HANDLE;
    }

    g_vk.meterFrames = 0;
}

void Transition(VkCommandBuffer cmd, OwnedImage& img, VkImageLayout to)
{
    if (img.image == VK_NULL_HANDLE || img.layout == to)
        return;

    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = img.layout;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = img.image;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    const auto access = [](VkImageLayout layout) -> VkAccessFlags {
        switch (layout)
        {
        case VK_IMAGE_LAYOUT_UNDEFINED: return 0;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return VK_ACCESS_TRANSFER_READ_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return VK_ACCESS_TRANSFER_WRITE_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return VK_ACCESS_SHADER_READ_BIT;
        default: return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
    };
    barrier.srcAccessMask = access(img.layout);
    barrier.dstAccessMask = access(to);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);

    img.layout = to;
}

void TransitionForeign(VkCommandBuffer cmd, VkImage image, VkImageSubresourceRange range, VkImageLayout from,
                       VkImageLayout to)
{
    if (image == VK_NULL_HANDLE || from == to)
        return;

    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = range;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
}

bool LoadForwarder()
{
    if (g_vk.forwarder != nullptr)
        return g_vk.create != nullptr;

    auto path = Util::FindFilePath(Util::DllPath().remove_filename(), "nvngx.dll_dlssnr.dll");

    if (!path.has_value())
        path = Util::FindFilePath(Util::ExePath().remove_filename(), "nvngx.dll_dlssnr.dll");

    if (!path.has_value())
    {
        Fail("nvngx.dll_dlssnr.dll was not found beside OptiScaler or the game");
        return false;
    }

    g_vk.forwarder = LoadLibraryW(path->wstring().c_str());

    if (g_vk.forwarder == nullptr)
    {
        Fail("the forwarder would not load");
        return false;
    }

    g_vk.probe = (PFN_VkProbe) GetProcAddress(g_vk.forwarder, "dlssnr_vk_probe");
    g_vk.init = (PFN_VkInit) GetProcAddress(g_vk.forwarder, "dlssnr_vk_init");
    g_vk.create = (PFN_VkCreate) GetProcAddress(g_vk.forwarder, "dlssnr_vk_create");
    g_vk.evaluate = (PFN_VkEvaluate) GetProcAddress(g_vk.forwarder, "dlssnr_vk_evaluate");
    g_vk.release = (PFN_VkRelease) GetProcAddress(g_vk.forwarder, "dlssnr_vk_release");

    if (g_vk.init == nullptr || g_vk.create == nullptr || g_vk.evaluate == nullptr)
    {
        Fail("the forwarder is missing its Vulkan entry points");
        return false;
    }

    return true;
}

bool FormatCanHoldLinearHdr(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R16G16B16_SFLOAT:
    case VK_FORMAT_R32G32B32_SFLOAT:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        return true;
    default:
        return false;
    }
}

unsigned int GameCreateFlags(NVSDK_NGX_Parameter* params)
{
    unsigned int flags = 0;

    if (params != nullptr)
        params->Get(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, &flags);

    return flags;
}

std::optional<std::filesystem::path> FindSnippet()
{
    auto snippet = Util::FindFilePath(Util::DllPath().remove_filename(), "nvngx_dlssnr.dll");

    if (!snippet.has_value())
        snippet = Util::FindFilePath(Util::ExePath().remove_filename(), "nvngx_dlssnr.dll");

    return snippet;
}

} // namespace

bool IsRunningVk() { return g_vk.feature != nullptr && !g_vk.failed; }
const char* FailureReasonVk() { return g_vk.failed ? g_vk.reason : ""; }
unsigned long long FramesVk() { return g_vk.frames; }
bool ExposureOfferedVk() { return g_vk.exposureOffered; }
std::optional<double> LastGpuTimeVk() { return g_vk.lastGpuTime; }

static void EvaluateAtSeamVk(VkCommandBuffer cmdBuffer, NVSDK_NGX_Parameter* params, VkInstance instance,
                             VkPhysicalDevice physicalDevice, VkDevice device, bool beforeSr, bool forcePost,
                             bool& applied, bool* handled = nullptr)
{
    applied = false;
    auto& cfg = *Config::Instance();

    if (cfg.DlssNrDeferredDlss.value_or_default())
    {
        static bool warnedDeferred = false;
        if (!warnedDeferred)
        {
            LOG_WARN("DLSS-NR DeferredDLSS requires the D3D12 path or a D3D12 bridge; "
                     "native Vulkan leaves the clean SR frame unchanged");
            warnedDeferred = true;
        }
        return;
    }

    if (!cfg.DlssNrEnabled.value_or_default())
        return;

    if (forcePost && !cfg.DlssNrApplyAfterRR.value_or_default())
        return;

    if (cmdBuffer == VK_NULL_HANDLE || params == nullptr || device == VK_NULL_HANDLE ||
        physicalDevice == VK_NULL_HANDLE)
        return;

    std::lock_guard<std::mutex> lock(g_vkMutex);

    if (g_vk.failed)
        return;

    NVSDK_NGX_Resource_VK* colour = nullptr;
    NVSDK_NGX_Resource_VK* depth = nullptr;
    NVSDK_NGX_Resource_VK* motion = nullptr;

    params->Get(beforeSr ? NVSDK_NGX_Parameter_Color : NVSDK_NGX_Parameter_Output, (void**) &colour);
    params->Get(NVSDK_NGX_Parameter_Depth, (void**) &depth);
    params->Get(NVSDK_NGX_Parameter_MotionVectors, (void**) &motion);

    NVSDK_NGX_Resource_VK* exposure = nullptr;
    float preExposure = 1.0f;
    const bool havePre =
        params->Get(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, &preExposure) == NVSDK_NGX_Result_Success;

    params->Get(NVSDK_NGX_Parameter_ExposureTexture, (void**) &exposure);

    static bool saidExposure = false;
    if (!saidExposure)
    {
        saidExposure = true;
        LOG_INFO("DLSS-NR Vulkan: exposure from the game: DLSS.Pre.Exposure {}, ExposureTexture {}",
                 havePre ? std::to_string(preExposure) : std::string("not supplied"),
                 exposure != nullptr ? "supplied" : "not supplied");
    }

    g_vk.exposureOffered = exposure != nullptr;

    if (havePre && std::isfinite(preExposure) && preExposure > 0.0f)
        g_vk.gamePreExposure = preExposure;

    if (g_vk.meterFrames >= kMeterSlots)
    {
        const void* mapped = g_vk.meterMapped[g_vk.meterFrames % kMeterSlots];

        if (mapped != nullptr)
        {
            float measured = 0.0f;
            std::memcpy(&measured, mapped, sizeof(float));

            if (std::isfinite(measured) && measured > 0.0f)
                g_vk.gameExposure = measured;
        }
    }

    static float loggedExposure = -1.0f;
    if (g_vk.gameExposure > 1e-6f &&
        std::abs(loggedExposure - g_vk.gameExposure) > std::max(0.02f * g_vk.gameExposure, 1e-5f))
    {
        loggedExposure = g_vk.gameExposure;
        LOG_INFO("DLSS-NR Vulkan: the game's exposure is {}, pre-exposure {}, so white point {}",
                 g_vk.gameExposure, g_vk.gamePreExposure, g_vk.gamePreExposure / g_vk.gameExposure);
    }

    if (colour == nullptr || depth == nullptr || motion == nullptr)
    {
        static bool said = false;
        if (!said)
        {
            said = true;
            LOG_INFO("DLSS-NR Vulkan: the parameter block carried no {}",
                     colour == nullptr ? "output" : (depth == nullptr ? "depth" : "motion vectors"));
        }
        return;
    }

    if (colour->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW ||
        depth->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW ||
        motion->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW ||
        colour->Resource.ImageViewInfo.ImageView == VK_NULL_HANDLE ||
        depth->Resource.ImageViewInfo.ImageView == VK_NULL_HANDLE ||
        motion->Resource.ImageViewInfo.ImageView == VK_NULL_HANDLE)
        return;

    uint32_t width = colour->Resource.ImageViewInfo.Width;
    uint32_t height = colour->Resource.ImageViewInfo.Height;
    uint32_t guideWidth = depth->Resource.ImageViewInfo.Width;
    uint32_t guideHeight = depth->Resource.ImageViewInfo.Height;
    const uint32_t motionAllocationWidth = motion->Resource.ImageViewInfo.Width;
    const uint32_t motionAllocationHeight = motion->Resource.ImageViewInfo.Height;

    uint32_t renderWidth = 0, renderHeight = 0, baseX = 0, baseY = 0;
    params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, &renderWidth);
    params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, &renderHeight);
    if (beforeSr)
    {
        params->Get(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, &baseX);
        params->Get(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, &baseY);
        if (baseX || baseY || ((renderWidth == 0) != (renderHeight == 0)) ||
            renderWidth > width || renderHeight > height)
            return;
        if (renderWidth && renderHeight)
        {
            width = renderWidth;
            height = renderHeight;
        }
    }

    if (width == 0 || height == 0)
        return;
    if (handled)
        *handled = true;

    // --- MOTION VECTOR & SUBRECT FIX APPLIED HERE ---
    const unsigned int createFlags = GameCreateFlags(params);
    const bool gameSaysHdr = (createFlags & NVSDK_NGX_DLSS_Feature_Flags_IsHDR) != 0;
    const bool depthInverted = (createFlags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) != 0;
    const bool lowResolutionMotion = (createFlags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes) != 0;

    uint32_t depthBaseX = 0, depthBaseY = 0, motionBaseX = 0, motionBaseY = 0;
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, &depthBaseX);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, &depthBaseY);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, &motionBaseX);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, &motionBaseY);

    depthBaseX = std::min(depthBaseX, guideWidth);
    depthBaseY = std::min(depthBaseY, guideHeight);
    guideWidth = std::min(renderWidth ? renderWidth : guideWidth, guideWidth - depthBaseX);
    guideHeight = std::min(renderHeight ? renderHeight : guideHeight, guideHeight - depthBaseY);
    motionBaseX = std::min(motionBaseX, motionAllocationWidth);
    motionBaseY = std::min(motionBaseY, motionAllocationHeight);
    const uint32_t motionWidth = std::min(lowResolutionMotion ? (renderWidth ? renderWidth : width) : width,
                                          motionAllocationWidth - motionBaseX);
    const uint32_t motionHeight = std::min(lowResolutionMotion ? (renderHeight ? renderHeight : height) : height,
                                           motionAllocationHeight - motionBaseY);

    float mvScaleX = 1.0f, mvScaleY = 1.0f;
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &mvScaleX);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &mvScaleY);

    if (!guideWidth || !guideHeight || guideWidth > depth->Resource.ImageViewInfo.Width ||
        guideHeight > depth->Resource.ImageViewInfo.Height || motionWidth > motion->Resource.ImageViewInfo.Width ||
        motionHeight > motion->Resource.ImageViewInfo.Height)
        return;

    float workScale = forcePost ? cfg.DlssNrRRWorkingScale.value_or_default()
                               : cfg.DlssNrWorkingScale.value_or_default();
    workScale = std::isfinite(workScale) ? std::clamp(workScale, 0.25f, 2.0f) : 1.0f;
    const uint32_t workWidth = std::max(1u, (uint32_t) (width * workScale + 0.5f));
    const uint32_t workHeight = std::max(1u, (uint32_t) (height * workScale + 0.5f));
    const bool reduced = workWidth != width || workHeight != height;
    const unsigned int passes = std::clamp(forcePost ? cfg.DlssNrRRPasses.value_or_default()
                                                     : cfg.DlssNrPasses.value_or_default(),
                                           1u, cfg.DlssNrUnlockPasses.value_or_default() ? DlssNr::MaxPassCount
                                                                                       : DlssNr::DefaultMaxPassCount);

    g_vk.instance = instance;
    g_vk.physicalDevice = physicalDevice;

    if (g_vk.device != device)
    {
        ShutdownVk(false);
        g_vk.device = device;
        g_vk.instance = instance;
        g_vk.physicalDevice = physicalDevice;
    }

    if (!LoadForwarder())
        return;

    if (g_vk.creationPending)
    {
        if (vkGetEventStatus(device, g_vk.creationReady) != VK_EVENT_SET)
            return;
        g_vk.creationPending = false;
    }

    if (!g_vk.ngxInitialised)
    {
        auto snippet = FindSnippet();
        if (!snippet.has_value())
        {
            Fail("nvngx_dlssnr.dll was not found beside OptiScaler or the game");
            return;
        }

        const int probe = g_vk.probe != nullptr ? g_vk.probe(snippet->wstring().c_str()) : 0;
        if (probe != 15)
        {
            LOG_ERROR("DLSS-NR Vulkan: the model's Vulkan surface is incomplete (probe {})", probe);
            Fail("the model does not expose a complete Vulkan surface");
            return;
        }

        const int result =
            g_vk.init(snippet->wstring().c_str(), State::Instance().NVNGX_ApplicationDataPath.c_str(),
                      (void*) instance, (void*) physicalDevice, (void*) device, 0x0000015);

        if (result != 1)
        {
            LOG_ERROR("DLSS-NR Vulkan: NVSDK_NGX_VULKAN_Init_Ext returned {}", result);
            Fail("the model would not initialise on this Vulkan device");
            return;
        }

        g_vk.ngxInitialised = true;
        LOG_INFO("DLSS-NR Vulkan: the model initialised on this device");
    }

    if (g_vk.capabilityParams == nullptr)
    {
        if (NVSDK_NGX_VULKAN_AllocateParameters(&g_vk.capabilityParams) != NVSDK_NGX_Result_Success ||
            g_vk.capabilityParams == nullptr)
        {
            Fail("a parameter block could not be allocated");
            return;
        }
    }

    if (g_vk.queryPool == VK_NULL_HANDLE)
    {
        VkPhysicalDeviceProperties props {};
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        g_vk.timestampPeriod = props.limits.timestampPeriod;

        if (g_vk.timestampPeriod > 0.0f)
        {
            VkQueryPoolCreateInfo info {};
            info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            info.queryCount = kTimingSlots * 2;

            if (vkCreateQueryPool(device, &info, nullptr, &g_vk.queryPool) != VK_SUCCESS)
            {
                g_vk.queryPool = VK_NULL_HANDLE;
                LOG_INFO("DLSS-NR Vulkan: no timestamp pool, the pass will not report its cost");
            }
        }
    }

    if (g_vk.pass == nullptr)
    {
        g_vk.pass = std::make_unique<DlssNr_Vk>("Neural Rendering", device, physicalDevice);
        if (!g_vk.pass->IsInit())
        {
            g_vk.pass.reset();
            Fail("the composition pass could not be created");
            return;
        }
    }

    bool profileChanged = g_vk.activePasses != passes;
    for (unsigned int pass = 0; pass < passes; ++pass)
        profileChanged |= g_vk.builtTuning[pass] != Profiles::PassTuning(cfg, pass) ||
                          g_vk.builtPreset[pass] != Profiles::PassPreset(cfg, pass) ||
                          g_vk.builtStyle[pass] != Profiles::PassStyle(cfg, pass);

    if (g_vk.width != width || g_vk.height != height || g_vk.workWidth != workWidth ||
        g_vk.workHeight != workHeight || g_vk.beforeSr != beforeSr || profileChanged)
    {
        if (g_vk.device != VK_NULL_HANDLE)
            vkDeviceWaitIdle(g_vk.device);

        if (g_vk.feature != nullptr && g_vk.release != nullptr)
        {
            g_vk.release(g_vk.feature);
            g_vk.feature = nullptr;
        }
        for (auto& feature : g_vk.laterFeatures)
        {
            if (feature && g_vk.release)
                g_vk.release(feature);
            feature = nullptr;
        }
        DestroyImage(g_vk.scratch);

        const VkFormat working = VK_FORMAT_R16G16B16A16_SFLOAT;
        const bool meterReady = (g_vk.meter.Valid() || CreateImage(g_vk.meter, kMeterSide, kMeterSide,
                                                                   VK_FORMAT_R32_SFLOAT, true)) &&
                                CreateMeterReadback();

        if (!meterReady)
            LOG_WARN("DLSS-NR Vulkan: no exposure meter; the white point stays on the slider");

        DestroyImage(g_vk.proxySmall);
        DestroyImage(g_vk.outputNative);

        const bool ok = CreateImage(g_vk.output, workWidth, workHeight, working, true) &&
                        (passes == 1 || CreateImage(g_vk.scratch, workWidth, workHeight, working, true)) &&
                        CreateImage(g_vk.proxy, width, height, working, true) &&
                        CreateImage(g_vk.keep, width, height, working, true) &&
                        (!beforeSr || CreateImage(g_vk.preColor, width, height, working, false)) &&
                        (!reduced || CreateImage(g_vk.proxySmall, workWidth, workHeight, working, true)) &&
                        (workScale <= 1.0f || CreateImage(g_vk.outputNative, width, height, working, true));

        if (!ok)
        {
            Fail("the pass could not allocate its own surfaces");
            return;
        }

        g_vk.width = width;
        g_vk.height = height;
        g_vk.workWidth = workWidth;
        g_vk.workHeight = workHeight;
        g_vk.beforeSr = beforeSr;
        g_vk.activePasses = passes;
        for (unsigned int pass = 0; pass < passes; ++pass)
        {
            g_vk.builtTuning[pass] = Profiles::PassTuning(cfg, pass);
            g_vk.builtPreset[pass] = Profiles::PassPreset(cfg, pass);
            g_vk.builtStyle[pass] = Profiles::PassStyle(cfg, pass);
        }
        g_vk.reset = true;
    }

    bool created = false;
    for (unsigned int pass = 0; pass < passes; ++pass)
    {
        void*& feature = pass == 0 ? g_vk.feature : g_vk.laterFeatures[pass];
        if (feature)
            continue;
        const auto tuning = Profiles::PassTuning(cfg, pass);
        feature = g_vk.create((void*) cmdBuffer, g_vk.capabilityParams, workWidth, workHeight,
                             (int) Profiles::PassPreset(cfg, pass), tuning.intensity,
                             (int) Profiles::PassStyle(cfg, pass), tuning.structure, tuning.tone,
                             tuning.skin, tuning.autoMask ? 1 : 0, 1);
        if (!feature)
        {
            Fail("the model would not build a feature on this device");
            return;
        }

        LOG_INFO("DLSS-NR Vulkan: pass {} built at {}x{} (frame {}x{}, {} SR)", pass + 1,
                 workWidth, workHeight, width, height, beforeSr ? "before" : "after");
        created = true;
        g_vk.reset = true;
    }

    if (created)
    {
        if (g_vk.creationReady == VK_NULL_HANDLE)
        {
            VkEventCreateInfo info { VK_STRUCTURE_TYPE_EVENT_CREATE_INFO };
            if (vkCreateEvent(device, &info, nullptr, &g_vk.creationReady) != VK_SUCCESS)
            {
                Fail("could not allocate the model creation marker");
                return;
            }
        }
        else
            vkResetEvent(device, g_vk.creationReady);
        vkCmdSetEvent(cmdBuffer, g_vk.creationReady, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        g_vk.creationPending = true;
        return;
    }

    // Reset detection
    {
        unsigned int gameReset = 0;
        if (params->Get(NVSDK_NGX_Parameter_Reset, &gameReset) == NVSDK_NGX_Result_Success &&
            gameReset != 0)
        {
            g_vk.reset = true;
            static unsigned long long resets = 0;
            ++resets;
            if (resets <= 3 || resets % 100 == 0)
                LOG_INFO("DLSS-NR Vulkan: the game asked for a history reset ({} so far)", resets);
        }
    }

    const bool linearHdr = gameSaysHdr && FormatCanHoldLinearHdr(colour->Resource.ImageViewInfo.Format);
    float whitePoint = cfg.DlssNrWhitePointScale.value_or_default();

    if (cfg.DlssNrWhitePointSource.value_or_default() == 1 && g_vk.gameExposure > 1e-6f)
    {
        const float trim = std::clamp(cfg.DlssNrWhitePointTrim.value_or_default(), 0.25f, 4.0f);
        whitePoint = std::clamp(g_vk.gamePreExposure / g_vk.gameExposure * trim, 0.01f, 4096.0f);
    }

    static bool saidEncoding = false;
    if (!saidEncoding)
    {
        saidEncoding = true;
        LOG_INFO("DLSS-NR Vulkan: the game's buffer is {} (flag {}, format {}), depth {}",
                 linearHdr ? "linear HDR" : "already tone-mapped", gameSaysHdr ? "set" : "clear",
                 (int) colour->Resource.ImageViewInfo.Format, depthInverted ? "inverted" : "normal");
    }

    DlssNrConstants encode {};
    encode.Mode = DlssNrMode_Encode;
    encode.Width = width;
    encode.Height = height;
    encode.WhitePoint = whitePoint;
    encode.Passthrough = linearHdr ? 0u : 1u;
    encode.ReversibleMode = cfg.DlssNrReversibleMode.value_or_default();
    encode.ApplyModel = cfg.DlssNrApplyModel.value_or_default() ? 1u : 0u;
    encode.TransferStrength = cfg.DlssNrTransferStrength.value_or_default();
    const auto strength = [](float v) { return std::isfinite(v) ? std::clamp(v, 0.0f, 1.0f) : 1.0f; };
    encode.SkinProtection = cfg.DlssNrSkinProtection.value_or_default();
    encode.ShowSkinMask = cfg.DlssNrShowSkinMask.value_or_default();
    encode.SkinDetail = strength(cfg.DlssNrSkinDetail.value_or_default());
    encode.SkinColour = cfg.DlssNrSkinToneEnabled.value_or_default() ? strength(cfg.DlssNrSkinColour.value_or_default()) : 0.0f;
    encode.EnvironmentDetail = strength(cfg.DlssNrEnvironmentDetail.value_or_default());
    encode.EnvironmentColour = strength(cfg.DlssNrEnvironmentColour.value_or_default());
    encode.ColourStrength = cfg.DlssNrColourStrength.value_or_default();
    encode.MaxRatio = cfg.DlssNrMaxRatio.value_or_default();
    encode.Transfer = cfg.DlssNrTransfer.value_or_default();
    encode.DebugScale = cfg.DlssNrWhitePointScale.value_or_default();
    encode.GuideWidth = guideWidth;
    encode.GuideHeight = guideHeight;

    const uint32_t timingSlot = (uint32_t) (g_vk.timedFrames % kTimingSlots);
    if (g_vk.queryPool != VK_NULL_HANDLE)
    {
        vkCmdResetQueryPool(cmdBuffer, g_vk.queryPool, timingSlot * 2, 2);
        vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_vk.queryPool, timingSlot * 2);
    }

    Transition(cmdBuffer, g_vk.proxy, VK_IMAGE_LAYOUT_GENERAL);
    Transition(cmdBuffer, g_vk.keep, VK_IMAGE_LAYOUT_GENERAL);

    if (!g_vk.pass->Dispatch(cmdBuffer, encode, width, height, colour->Resource.ImageViewInfo.ImageView,
                             VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, g_vk.proxy.view, g_vk.keep.view,
                             beforeSr ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL))
    {
        Fail("the encode dispatch failed");
        return;
    }

    OwnedImage* modelInput = &g_vk.proxy;
    if (reduced && g_vk.proxySmall.Valid())
    {
        bool built = false;
        if (workScale > 1.0f)
        {
            const Scaler wantScaler = cfg.DlssNrScalingDownscaler.value_or_default();
            if (g_vk.nrScaler != wantScaler)
            {
                if (g_vk.device != VK_NULL_HANDLE)
                    vkDeviceWaitIdle(g_vk.device);
                g_vk.superUp.reset();
                g_vk.superDown.reset();
                g_vk.nrScaler = wantScaler;
            }
            if (!g_vk.superUp)
                g_vk.superUp = std::make_unique<OS_Vk>("DLSS-NR VK supersample up", device, physicalDevice,
                                                       true, wantScaler);
            if (!g_vk.superDown)
                g_vk.superDown = std::make_unique<OS_Vk>("DLSS-NR VK supersample down", device,
                                                         physicalDevice, false, wantScaler);

            Transition(cmdBuffer, g_vk.proxy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            Transition(cmdBuffer, g_vk.proxySmall, VK_IMAGE_LAYOUT_GENERAL);

            VkImageInfo upin = ImageInfoOf(g_vk.proxy);
            VkImageInfo upout = ImageInfoOf(g_vk.proxySmall);

            if (g_vk.superUp && g_vk.superUp->IsInit() && g_vk.superUp->Dispatch(cmdBuffer, upin, upout))
                built = true;
            else
            {
                static bool warnedVkSuper = false;
                if (!warnedVkSuper)
                {
                    warnedVkSuper = true;
                    LOG_WARN("DLSS-NR Vulkan supersample: upscaler unavailable, falling back to box enlarge.");
                }
            }
        }

        if (!built)
        {
            DlssNrConstants down = encode;
            down.Mode = DlssNrMode_Downsample;
            down.Width = workWidth;
            down.Height = workHeight;

            Transition(cmdBuffer, g_vk.proxy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            Transition(cmdBuffer, g_vk.proxySmall, VK_IMAGE_LAYOUT_GENERAL);

            if (!g_vk.pass->Dispatch(cmdBuffer, down, workWidth, workHeight, g_vk.proxy.view, VK_NULL_HANDLE,
                                     VK_NULL_HANDLE, VK_NULL_HANDLE, g_vk.proxySmall.view, VK_NULL_HANDLE,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
            {
                Fail("the downsample dispatch failed");
                return;
            }
        }

        modelInput = &g_vk.proxySmall;
    }

    // Exposure Meter
    if (cfg.DlssNrWhitePointSource.value_or_default() == 1 && exposure != nullptr &&
        exposure->Type == NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW &&
        exposure->Resource.ImageViewInfo.ImageView != VK_NULL_HANDLE && g_vk.meter.Valid())
    {
        const unsigned long long slot = g_vk.meterFrames % kMeterSlots;

        if (g_vk.meterReadback[slot] != VK_NULL_HANDLE)
        {
            DlssNrConstants meter = encode;
            meter.Mode = DlssNrMode_Meter;
            meter.Width = kMeterSide;
            meter.Height = kMeterSide;

            Transition(cmdBuffer, g_vk.meter, VK_IMAGE_LAYOUT_GENERAL);

            if (g_vk.pass->Dispatch(cmdBuffer, meter, kMeterSide, kMeterSide, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                    VK_NULL_HANDLE, exposure->Resource.ImageViewInfo.ImageView, g_vk.meter.view,
                                    VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
            {
                Transition(cmdBuffer, g_vk.meter, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

                VkBufferImageCopy region {};
                region.bufferOffset = 0;
                region.bufferRowLength = 0;
                region.bufferImageHeight = 0;
                region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                region.imageOffset = { 0, 0, 0 };
                region.imageExtent = { kMeterSide, kMeterSide, 1 };

                vkCmdCopyImageToBuffer(cmdBuffer, g_vk.meter.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       g_vk.meterReadback[slot], 1, &region);

                VkBufferMemoryBarrier toHost {};
                toHost.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                toHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toHost.buffer = g_vk.meterReadback[slot];
                toHost.offset = 0;
                toHost.size = kMeterBytes;

                vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                                     nullptr, 1, &toHost, 0, nullptr);

                g_vk.meterFrames++;
            }
        }
    }

    // --- EVALUATE PASSES WITH MOTION VECTOR FIX ---
    const float mvToWorkX = width != 0 ? (float) workWidth / (float) width : 1.0f;
    const float mvToWorkY = height != 0 ? (float) workHeight / (float) height : 1.0f;
    OwnedImage* answer = &g_vk.output;
    OwnedImage* input = modelInput;
    int evaluated = 1;

    for (unsigned int pass = 0; pass < passes; ++pass)
    {
        Transition(cmdBuffer, *input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cmdBuffer, *answer, VK_IMAGE_LAYOUT_GENERAL);
        const auto tuning = Profiles::PassTuning(cfg, pass);
        evaluated = g_vk.evaluate(
            (void*) cmdBuffer, pass == 0 ? g_vk.feature : g_vk.laterFeatures[pass], g_vk.capabilityParams,
            &input->ngx, depth, motion, &answer->ngx, workWidth, workHeight, guideWidth, guideHeight,
            motionWidth, motionHeight, depthBaseX, depthBaseY, motionBaseX, motionBaseY,
            depthInverted ? 1 : 0, g_vk.reset ? 1 : 0, tuning.intensity,
            (int) Profiles::PassStyle(cfg, pass), tuning.structure, tuning.tone, tuning.skin,
            tuning.autoMask ? 1 : 0, mvScaleX * mvToWorkX, mvScaleY * mvToWorkY);

        if (evaluated != 1)
            break;

        if (pass + 1 < passes)
        {
            input = answer;
            answer = answer == &g_vk.output ? &g_vk.scratch : &g_vk.output;
        }
    }

    g_vk.reset = false;
    g_vk.frames++;

    if (evaluated != 1)
    {
        LOG_ERROR("DLSS-NR Vulkan: evaluate returned {}", evaluated);
        Fail("the model refused to evaluate");
        return;
    }

    // Resolve Pass
    DlssNrConstants resolve = encode;
    resolve.Mode = DlssNrMode_Resolve;

    OwnedImage* resolveProxy = modelInput;
    OwnedImage* resolveAnswer = answer;

    if (workScale > 1.0f && g_vk.superDown && g_vk.superDown->IsInit() && g_vk.outputNative.Valid())
    {
        Transition(cmdBuffer, *answer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cmdBuffer, g_vk.outputNative, VK_IMAGE_LAYOUT_GENERAL);

        VkImageInfo dsin = ImageInfoOf(*answer);
        VkImageInfo dsout = ImageInfoOf(g_vk.outputNative);

        if (g_vk.superDown->Dispatch(cmdBuffer, dsin, dsout))
        {
            resolveProxy = &g_vk.proxy;
            resolveAnswer = &g_vk.outputNative;
        }
    }

    Transition(cmdBuffer, *resolveProxy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(cmdBuffer, *resolveAnswer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(cmdBuffer, g_vk.keep, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (beforeSr)
        Transition(cmdBuffer, g_vk.preColor, VK_IMAGE_LAYOUT_GENERAL);

    if (!g_vk.pass->Dispatch(cmdBuffer, resolve, width, height, resolveProxy->view, resolveAnswer->view,
                             g_vk.keep.view, VK_NULL_HANDLE,
                             beforeSr ? g_vk.preColor.view : colour->Resource.ImageViewInfo.ImageView,
                             VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
    {
        Fail("the resolve dispatch failed");
        return;
    }

    if (beforeSr)
        Transition(cmdBuffer, g_vk.preColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    applied = true;

    if (g_vk.queryPool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_vk.queryPool, timingSlot * 2 + 1);
        g_vk.timedFrames++;

        if (g_vk.timedFrames > kTimingSlots)
        {
            const uint32_t readSlot = (uint32_t) (g_vk.timedFrames % kTimingSlots);
            uint64_t ticks[2] = {};

            if (vkGetQueryPoolResults(device, g_vk.queryPool, readSlot * 2, 2, sizeof(ticks), ticks, sizeof(uint64_t),
                                      VK_QUERY_RESULT_64_BIT) == VK_SUCCESS &&
                ticks[1] > ticks[0])
            {
                const double ms = (double) (ticks[1] - ticks[0]) * (double) g_vk.timestampPeriod / 1e6;
                if (ms > 0.0 && ms < 1000.0)
                    g_vk.lastGpuTime = ms;
            }
        }
    }

    static bool reported = false;
    if (!reported && g_vk.frames > 2)
    {
        reported = true;
        LOG_INFO("DLSS-NR Vulkan: running {} SR at {}x{}, guides {}x{}", beforeSr ? "before" : "after",
                 width, height, guideWidth, guideHeight);
    }
}

NVSDK_NGX_Resource_VK* EvaluateBeforeUpscaleVk(VkCommandBuffer cmd, NVSDK_NGX_Parameter* params,
                                             VkInstance instance, VkPhysicalDevice pd, VkDevice device, bool& handled)
{
    handled = false;
    if (!Config::Instance()->DlssNrRunBeforeSr.value_or_default())
        return nullptr;
    bool applied = false;
    EvaluateAtSeamVk(cmd, params, instance, pd, device, true, false, applied, &handled);
    return applied ? &g_vk.preColor.ngx : nullptr;
}

void EvaluateAfterUpscaleVk(VkCommandBuffer cmd, NVSDK_NGX_Parameter* params, VkInstance instance,
                            VkPhysicalDevice pd, VkDevice device, bool forcePost, bool ranBefore)
{
    if (ranBefore)
        return;
    bool applied = false;
    EvaluateAtSeamVk(cmd, params, instance, pd, device, false, forcePost, applied);
}

void ShutdownVk(bool deviceAlive)
{
    if (!deviceAlive)
    {
        g_vk.pass.release();
        g_vk.superUp.release();
        g_vk.superDown.release();
        g_vk.nrScaler = Scaler::Count;
        g_vk.feature = nullptr;
        for (auto& feature : g_vk.laterFeatures)
            feature = nullptr;
        g_vk.activePasses = 0;
        g_vk.creationReady = VK_NULL_HANDLE;
        g_vk.creationPending = false;
        g_vk.capabilityParams = nullptr;
        g_vk.queryPool = VK_NULL_HANDLE;
        g_vk.output = OwnedImage {};
        g_vk.scratch = OwnedImage {};
        g_vk.proxy = OwnedImage {};
        g_vk.proxySmall = OwnedImage {};
        g_vk.outputNative = OwnedImage {};
        g_vk.keep = OwnedImage {};
        g_vk.preColor = OwnedImage {};
        g_vk.meter = OwnedImage {};

        for (int i = 0; i < 4; ++i)
        {
            g_vk.meterReadback[i] = VK_NULL_HANDLE;
            g_vk.meterReadbackMemory[i] = VK_NULL_HANDLE;
            g_vk.meterMapped[i] = nullptr;
        }

        g_vk.device = VK_NULL_HANDLE;
        g_vk.width = 0;
        g_vk.height = 0;
        g_vk.workWidth = 0;
        g_vk.workHeight = 0;
        g_vk.timedFrames = 0;
        g_vk.meterFrames = 0;
        g_vk.lastGpuTime.reset();
        g_vk.ngxInitialised = false;
        g_vk.reset = true;
        return;
    }

    if (g_vk.device != VK_NULL_HANDLE)
        vkDeviceWaitIdle(g_vk.device);

    if (g_vk.feature != nullptr && g_vk.release != nullptr)
        g_vk.release(g_vk.feature);

    g_vk.feature = nullptr;

    for (auto& feature : g_vk.laterFeatures)
    {
        if (feature && g_vk.release)
            g_vk.release(feature);
        feature = nullptr;
    }
    g_vk.activePasses = 0;
    if (g_vk.creationReady != VK_NULL_HANDLE)
        vkDestroyEvent(g_vk.device, g_vk.creationReady, nullptr);
    g_vk.creationReady = VK_NULL_HANDLE;
    g_vk.creationPending = false;

    DestroyImage(g_vk.output);
    DestroyImage(g_vk.scratch);
    DestroyImage(g_vk.proxy);
    DestroyImage(g_vk.proxySmall);
    DestroyImage(g_vk.outputNative);
    DestroyImage(g_vk.keep);
    DestroyImage(g_vk.preColor);
    DestroyImage(g_vk.meter);
    DestroyMeterReadback();

    g_vk.pass.reset();
    g_vk.superUp.reset();
    g_vk.superDown.reset();
    g_vk.nrScaler = Scaler::Count;

    if (g_vk.capabilityParams != nullptr)
    {
        NVSDK_NGX_VULKAN_DestroyParameters(g_vk.capabilityParams);
        g_vk.capabilityParams = nullptr;
    }

    if (g_vk.queryPool != VK_NULL_HANDLE && g_vk.device != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(g_vk.device, g_vk.queryPool, nullptr);
        g_vk.queryPool = VK_NULL_HANDLE;
    }

    g_vk.timedFrames = 0;
    g_vk.lastGpuTime.reset();

    g_vk.device = VK_NULL_HANDLE;
    g_vk.width = 0;
    g_vk.height = 0;
    g_vk.ngxInitialised = false;
    g_vk.reset = true;
}

} // namespace DlssNr