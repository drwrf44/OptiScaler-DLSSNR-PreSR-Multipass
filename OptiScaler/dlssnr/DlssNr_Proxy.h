#pragma once

#include <d3d12.h>

namespace DlssNr
{
namespace Proxy
{
bool Available();

unsigned int Run(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, ID3D12Resource* color,
                 ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                 unsigned int width, unsigned int height, unsigned int guideWidth,
                 unsigned int guideHeight, unsigned int motionWidth, unsigned int motionHeight,
                 unsigned int depthBaseX, unsigned int depthBaseY, unsigned int motionBaseX,
                 unsigned int motionBaseY, bool depthInverted, bool reset, float mvScaleX, float mvScaleY);

// Drops the feature and its parameter block, for a resolution change or shutdown.
void Release();
} // namespace Proxy
} // namespace DlssNr