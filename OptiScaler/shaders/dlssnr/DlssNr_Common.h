#pragma once

// Everything about the Neural Rendering composition pass that is not Direct3D 12.
//
// Two kinds of thing live here. The constants the shader reads, so a Vulkan implementation can share
// the struct rather than redefine it and drift; and the parameter names the model is driven by, which
// are the model's own and identical whatever API is calling it.
//
// The Direct3D 12 side is DlssNr_Dx12, which implements Shader_Dx12 the way RCAS and Output Scaling
// do. The model itself is separate again: creating and evaluating an NGX feature is not a dispatch,
// so it does not belong in a shader class.

#include <cstdint>

// Which of the passes a dispatch is. One shader, because they read and write the same set of
// resources and differ only in what they compute.
enum DlssNrMode : uint32_t
{
    DlssNrMode_Encode = 0,     // the frame -> a tone-mapped proxy, plus an untouched copy
    DlssNrMode_Resolve = 1,    // proxy + the model's answer + the untouched copy -> the edited frame
    DlssNrMode_Downsample = 2, // the proxy -> a smaller proxy, when the model works below full size
    DlssNrMode_Meter = 3,      // the exposure texture -> tile (0,0), for the white point
    DlssNrMode_Calibrate = 4,  // the untouched frame -> a grid of tile peak luminances
    DlssNrMode_EncodeResidual = 5, // NR-composed minus original; signed difference encoded around 0.5
    DlssNrMode_ApplyResidual = 6,  // decode private DLSS result and add to clean SR output
    DlssNrMode_UnitExposure = 7,   // constant exposure for the private DLSS feature
    DlssNrMode_NormalizeMotion = 8, // current-to-previous motion in normalized image coordinates
    DlssNrMode_ComposeMotion = 9, // compose two successive fields at the displaced coordinate
    DlssNrMode_ApplyInterpolatedResidual = 10, // t4: R8_UNORM NVIDIA suppression flag
    DlssNrMode_ZeroMotion = 11 // private reset-only NR/SR guide, never passed to residual FG
};

// A successful sample may be reused only on the immediately following frame.
// Failed composition, cuts and gaps must not turn a two-frame hold into a freeze.
struct DlssNrResidualHold
{
    uint64_t sampleEpoch = 0;
    bool valid = false;
    bool CanReuse(uint64_t epoch) const { return valid && epoch > sampleEpoch && epoch - sampleEpoch == 1; }
    void SampleSucceeded(uint64_t epoch) { sampleEpoch = epoch; valid = true; }
    void Reset() { valid = false; }
};

// The meter's grid. 64 x 64 tiles over the whole frame, whatever its size.
constexpr uint32_t kDlssNrMeterGrid = 64;

// What the caller knows about the frame, and the pass cannot work out for itself.
struct DlssNrFrameInfo
{
    // Which way round depth runs. The game states this when it creates its own upscaler.
    bool DepthInverted = false;

    // How the game encodes its motion vectors, as the game itself reports it. Passed through: every
    // resource already carries a subrect saying how big it is, so scaling by the resolution ratio on
    // top of that counts it twice.
    float MvScaleX = 1.0f;
    float MvScaleY = 1.0f;

    // Throw away the model's history. Set it on a cut, a teleport, or the first frame of a feature.
    bool Reset = false;

    // Whether the colour buffer holds linear, open-ended light or a frame that has already been
    // through a tonemapper. Getting this wrong encodes an encoded frame a second time, which looks
    // washed out and banded.
    bool ColourIsLinearHdr = true;

    // The SR colour input arrives readable, whereas a completed upscaler output normally arrives as
    // a UAV. The DX12 pass uses this to preserve the caller's state and to fall back through a copy
    // when a pre-SR colour resource was not created with UAV support.
    bool BeforeUpscale = false;
    // Owned copy, not the game's Color: always arrives/returns NON_PIXEL_SHADER_RESOURCE.
    bool PrivateColorCopy = false;
    bool IndependentCommands = false; // owned command list, no game root signature to restore
    // A native RR result selects independent NR cost controls and a separate history lifecycle.
    bool AfterRayReconstruction = false;

    // Submission epoch supplied by the caller. Native DX12 uses the wrapped swapchain Present count;
    // the DX11/Vulkan bridges use their successfully submitted frame counter. A feature created in an
    // epoch is never evaluated until this value changes.
    unsigned long long SubmissionEpoch = 0;

    // The game's own exposure: a 1x1 texture holding, in the SDK's words, "the final exposure scale".
    void* ExposureTexture = nullptr;

    // The scale the game multiplied its buffer by for float precision, which DLSS is told so it can
    // undo it. Usually 1. Divided out before the exposure is applied, exactly as FSR's PrepareRgb does.
    float PreExposure = 1.0f;

    // How much of the depth and motion vector textures the game actually rendered into.
    // Before SR this also selects the origin-zero active colour rectangle, not its allocation.
    unsigned int RenderSubrectWidth = 0;
    unsigned int RenderSubrectHeight = 0;

    // --- MOTION VECTOR FIX (cmh1448) ---
    unsigned int DepthSubrectBaseX = 0;
    unsigned int DepthSubrectBaseY = 0;
    unsigned int MotionSubrectBaseX = 0;
    unsigned int MotionSubrectBaseY = 0;

    // DLSS permits motion vectors at either render or output resolution. This flag comes from the
    // feature-create flags and decides which valid-region dimensions apply to the motion texture.
    bool MotionVectorsLowResolution = false;
};

struct alignas(256) DlssNrConstants
{
    uint32_t Mode;
    float WhitePoint;

    uint32_t Width;
    uint32_t Height;

    // How much of the model's edit lands, and how much of it is allowed to be colour rather than
    // luminance. Separating the two is what keeps saturated highlights from shifting hue.
    float TransferStrength;
    float ColourStrength;

    uint32_t DebugView;

    // A ceiling on how far a pixel may be brightened. The transfer is a ratio, and a ratio against a
    // near-black proxy pixel is unbounded without one.
    float MaxRatio;

    // Set when the game's buffer is already tone-mapped, in which case there is nothing to convert
    // and the transfer is the identity.
    uint32_t Passthrough;

    float MvScaleX;
    float MvScaleY;

    // Depth and motion vectors come from the upscaler's inputs and so may be at render resolution
    // while colour and output are at display resolution.
    uint32_t GuideWidth;
    uint32_t GuideHeight;

    // Showing the pass against itself. 0 off, 1 side by side, 2 a wipe.
    uint32_t CompareMode;
    float CompareSplit;

    // How much of the frame side by side shows.
    float CompareZoom;

    // Which side the edited frame is on.
    uint32_t CompareSwap;

    // How a model that worked below the frame's size is brought back. 0 classic, 1 matched residual.
    uint32_t Transfer;

    // What the debug views are multiplied by on their way out.
    float DebugScale;

    // The reversible-proxy mode.
    uint32_t ReversibleMode;

    // 0 = output the clean upscaler frame, 1 = apply the model's edit.
    uint32_t ApplyModel;

    // D3D12 source-1 zero-latency exposure.
    uint32_t UseGameExposure;
    float ExposurePreMul;

    // Optional colour-based final-composition mask.
    uint32_t SkinProtection;
    uint32_t ShowSkinMask;
    float SkinDetail;
    float SkinColour;
    float EnvironmentDetail;
    float EnvironmentColour;
};

class DlssNr_Common
{
  protected:
    static constexpr const char* kEnabled = "DLSSNR.Enabled";
    static constexpr const char* kWidth = "DLSSNR.Width";
    static constexpr const char* kHeight = "DLSSNR.Height";

    static constexpr const char* kColor = "DLSSNR.Color";
    static constexpr const char* kDepth = "DLSSNR.Depth";
    static constexpr const char* kMotion = "DLSSNR.MVec";
    static constexpr const char* kOutput = "DLSSNR.Output";

    static constexpr const char* kDepthInverted = "DLSSNR.DepthInverted";
    static constexpr const char* kReset = "DLSSNR.Reset";
    static constexpr const char* kMvScaleX = "DLSSNR.MVecScaleX";
    static constexpr const char* kMvScaleY = "DLSSNR.MVecScaleY";

    static constexpr const char* kPreset = "DLSSNR.Hint.Render.Preset";
    static constexpr const char* kIntensity = "DLSSNR.Intensity";
    static constexpr const char* kStyle = "DLSSNR.Style";
    static constexpr const char* kLocalStructure = "DLSSNR.LocalStructureStrength";
    static constexpr const char* kLocalTone = "DLSSNR.LocalToneStrength";

    static constexpr const char* kAutoMask = "DLSSNR.UseAutoMask";
    static constexpr const char* kSkinStructure = "DLSSNR.SkinStructureStrength";

    static constexpr const char* kUi = "DLSSNR.UI";
    static constexpr const char* kUiAlpha = "DLSSNR.UIAlpha";
    static constexpr const char* kBackbuffer = "DLSSNR.Backbuffer";
    static constexpr const char* kUiCorrection = "DLSSNR.UICorrection";
};