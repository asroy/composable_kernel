#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "gridwise_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw.hpp"

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void gridwise_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw(
    InDesc,
    const Tensor<T>& in_nchw,
    WeiDesc,
    const Tensor<T>& wei_kcyx,
    OutDesc,
    Tensor<T>& out_nkhw,
    ConvStrides,
    ConvDilations,
    InLeftPads,
    InRightPads,
    ck::index_t nrepeat)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc =
        make_native_tensor_descriptor(InDesc::GetLengths(), InDesc::GetStrides());
    constexpr auto wei_kcyx_desc =
        make_native_tensor_descriptor(WeiDesc::GetLengths(), WeiDesc::GetStrides());
    constexpr auto out_nkhw_desc =
        make_native_tensor_descriptor(OutDesc::GetLengths(), OutDesc::GetStrides());

    // read params: problem description
    constexpr index_t G = 1;

    constexpr index_t N  = out_nkhw_desc.GetLength(I0);
    constexpr index_t K  = out_nkhw_desc.GetLength(I1);
    constexpr index_t Ho = out_nkhw_desc.GetLength(I2);
    constexpr index_t Wo = out_nkhw_desc.GetLength(I3);

    constexpr index_t C  = in_nchw_desc.GetLength(I1);
    constexpr index_t Hi = in_nchw_desc.GetLength(I2);
    constexpr index_t Wi = in_nchw_desc.GetLength(I3);

    constexpr index_t Y = wei_kcyx_desc.GetLength(I2);
    constexpr index_t X = wei_kcyx_desc.GetLength(I3);

    constexpr auto CPerGroup = C / G;

    constexpr auto in_n_c_hi_wi_desc =
        make_native_tensor_descriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto wei_k_cpergroup_y_x_desc =
        make_native_tensor_descriptor_packed(Sequence<K, CPerGroup, Y, X>{});
    constexpr auto out_n_k_ho_wo_desc =
        make_native_tensor_descriptor_packed(Sequence<N, K, Ho, Wo>{});

    // read params: tunning parameters
    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;
    constexpr index_t GemmMPerWave  = 128;
    constexpr index_t GemmNPerWave  = 64;
    constexpr index_t GemmKPack     = 4;

    // read params: dependent parameters
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmM = K;
    constexpr index_t GemmN = N * Ho * Wo;
    constexpr index_t GemmK = C * Y * X;

    constexpr index_t GridSize = math::integer_divide_ceil(GemmM, GemmMPerBlock) *
                                 math::integer_divide_ceil(GemmN, GemmNPerBlock);

    // A matrix copy
    constexpr index_t GemmABlockCopyClusterLengths_GemmK     = 4;
    constexpr index_t GemmABlockCopyClusterLengths_GemmM     = 64;
    constexpr index_t GemmABlockCopyClusterLengths_GemmKPack = 1;

    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmABlockCopyClusterLengths_GemmK;
    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmM =
        GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;
    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmKPack =
        GemmKPack / GemmABlockCopyClusterLengths_GemmKPack;

    using GemmABlockCopyClusterLengths_GemmG_GemmK_GemmM_GemmKPack =
        Sequence<1,
                 GemmABlockCopyClusterLengths_GemmK,
                 GemmABlockCopyClusterLengths_GemmM,
                 GemmABlockCopyClusterLengths_GemmKPack>;
    using GemmABlockCopySubLengths_GemmG_GemmK_GemmM_GemmKPack =
        Sequence<1,
                 GemmABlockCopyThreadSliceLengths_GemmK,
                 GemmABlockCopyThreadSliceLengths_GemmM,
                 GemmABlockCopyThreadSliceLengths_GemmKPack>;

    using GemmABlockCopyThreadClusterArrangeOrder =
        Sequence<0, 2, 1, 3>;                                  // [GemmG, GemmM, GemmK, GemmKPack]
    using GemmABlockCopySrcAccessOrder = Sequence<0, 2, 1, 3>; // [GemmG, GemmM, GemmK, GemmKPack]
    using GemmABlockCopyDstAccessOrder = Sequence<0, 1, 2, 3>; // [GemmG, GemmK, GemmM, GemmKPack]

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmKPack  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmKPack = 4;

    // B matrix Copy
    constexpr index_t GemmBBlockCopyClusterLengths_GemmK     = 4;
    constexpr index_t GemmBBlockCopyClusterLengths_GemmN     = 64;
    constexpr index_t GemmBBlockCopyClusterLengths_GemmKPack = 1;

    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmBBlockCopyClusterLengths_GemmK;
    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmN =
        GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;
    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmKPack =
        GemmKPack / GemmBBlockCopyClusterLengths_GemmKPack;

    using GemmBBlockCopyClusterLengths_GemmG_GemmK_GemmN_GemmKPack =
        Sequence<1,
                 GemmBBlockCopyClusterLengths_GemmK,
                 GemmBBlockCopyClusterLengths_GemmN,
                 GemmBBlockCopyClusterLengths_GemmKPack>;
    using GemmBBlockCopySubLengths_GemmG_GemmK_GemmN_GemmKPack =
        Sequence<1,
                 GemmBBlockCopyThreadSliceLengths_GemmK,
                 GemmBBlockCopyThreadSliceLengths_GemmN,
                 GemmBBlockCopyThreadSliceLengths_GemmKPack>;

    using GemmBBlockCopyThreadClusterArrangeOrder =
        Sequence<0, 1, 3, 2>;                                  // [GemmG, GemmK, GemmKPack, GemmN]
    using GemmBBlockCopySrcAccessOrder = Sequence<0, 1, 3, 2>; // [GemmG, GemmK, GemmKPack, GemmN]
    using GemmBBlockCopyDstAccessOrder = Sequence<0, 1, 2, 3>; // [GemmG, GemmK, GemmN, GemmKPack]

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN      = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmKPack = 1;

    // gridwise GEMM
    constexpr auto wkgrp_schd_order = NBlock1MBlock0;

    using TDevice = float;

    using gridwise_conv = GridwiseConvolutionForwardImplicitGemm_v4r4_xdlops_nchw_kcyx_nkhw<
        GridSize,
        BlockSize,
        TDevice, // Input data type
        TDevice, // Acc data type
        TDevice, // Ouput data type
        decltype(in_n_c_hi_wi_desc),
        decltype(wei_k_cpergroup_y_x_desc),
        decltype(out_n_k_ho_wo_desc),
        G,
        ConvStrides,
        ConvDilations,
        InLeftPads,
        InRightPads,
        GemmMPerBlock,
        GemmNPerBlock,
        GemmKPerBlock,
        GemmMPerWave,
        GemmNPerWave,
        GemmKPack,
        GemmABlockCopySubLengths_GemmG_GemmK_GemmM_GemmKPack,
        GemmABlockCopyClusterLengths_GemmG_GemmK_GemmM_GemmKPack,
        GemmABlockCopyThreadClusterArrangeOrder,
        GemmABlockCopySrcAccessOrder,
        GemmABlockCopyDstAccessOrder,
        GemmABlockCopySrcDataPerRead_GemmKPack,
        GemmABlockCopyDstDataPerWrite_GemmKPack,
        GemmBBlockCopySubLengths_GemmG_GemmK_GemmN_GemmKPack,
        GemmBBlockCopyClusterLengths_GemmG_GemmK_GemmN_GemmKPack,
        GemmBBlockCopyThreadClusterArrangeOrder,
        GemmBBlockCopySrcAccessOrder,
        GemmBBlockCopyDstAccessOrder,
        GemmBBlockCopySrcDataPerRead_GemmN,
        GemmBBlockCopyDstDataPerWrite_GemmKPack,
        wkgrp_schd_order>;

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_kcyx_device_buf(data_sz * wei_kcyx.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_kcyx_device_buf.ToDevice(wei_kcyx.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

    for(index_t i = 0; i < 5; ++i)
    {
        std::cout << "Start running " << nrepeat << " times..." << std::endl;

        KernelTimer timer;
        timer.Start();

        for(index_t j = 0; j < nrepeat; ++j)
        {
            launch_kernel(run_gridwise_operation<gridwise_conv,
                                                 const TDevice* const __restrict__,
                                                 const TDevice* const __restrict__,
                                                 TDevice* const __restrict__>,
                          dim3(GridSize),
                          dim3(BlockSize),
                          0,
                          0,
                          static_cast<TDevice*>(in_nchw_device_buf.GetDeviceBuffer()),
                          static_cast<TDevice*>(wei_kcyx_device_buf.GetDeviceBuffer()),
                          static_cast<TDevice*>(out_nkhw_device_buf.GetDeviceBuffer()));
        }

        timer.End();

        float ave_time = timer.GetElapsedTime() / nrepeat;

        float perf = (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
