#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "transform_forward_convolution_into_gemm_v4r4_nhwc_kyxc_nhwk.hpp"
#include "driver_dynamic_gemm_v1r1.hpp"

template <typename TInWei,
          typename TAcc,
          typename TOut,
          typename InLengths,
          typename WeiLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_dynamic_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk(
    const InLengths& in_n_hi_wi_c_lengths,
    const WeiLengths& wei_k_y_x_c_lengths,
    const OutLengths& out_n_ho_wo_k_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_hi_wi_c,
    const Tensor<TInWei>& wei_k_y_x_c,
    Tensor<TOut>& out_n_ho_wo_k,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};
    constexpr auto I7 = Number<7>{};
    constexpr auto I8 = Number<8>{};

    const auto N = out_n_ho_wo_k_lengths[I0];
    const auto K = out_n_ho_wo_k_lengths[I3];
    const auto C = wei_k_y_x_c_lengths[I3];

    const auto Hi = in_n_hi_wi_c_lengths[I1];
    const auto Wi = in_n_hi_wi_c_lengths[I2];

    const auto Ho = out_n_ho_wo_k_lengths[I1];
    const auto Wo = out_n_ho_wo_k_lengths[I2];

    const auto Y = wei_k_y_x_c_lengths[I1];
    const auto X = wei_k_y_x_c_lengths[I2];

    DeviceMem in_n_hi_wi_c_device_buf(sizeof(TInWei) * in_n_hi_wi_c.mDesc.GetElementSpace());
    DeviceMem wei_k_y_x_c_device_buf(sizeof(TInWei) * wei_k_y_x_c.mDesc.GetElementSpace());
    DeviceMem out_n_ho_wo_k_device_buf(sizeof(TOut) * out_n_ho_wo_k.mDesc.GetElementSpace());

    in_n_hi_wi_c_device_buf.ToDevice(in_n_hi_wi_c.mData.data());
    wei_k_y_x_c_device_buf.ToDevice(wei_k_y_x_c.mData.data());
    out_n_ho_wo_k_device_buf.ToDevice(out_n_ho_wo_k.mData.data());

    const auto in_n_hi_wi_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(in_n_hi_wi_c_lengths);
    const auto wei_k_y_x_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(wei_k_y_x_c_lengths);
    const auto out_n_ho_wo_k_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(out_n_ho_wo_k_lengths);

#if 0
    // cdata = 16, BlockSize = 64, 16x64x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmM1PerThread = 2;
    constexpr index_t GemmN1PerThread = 2;
    constexpr index_t GemmKPerThread  = 1;

    constexpr index_t GemmM1N1ThreadClusterM11 = 2;
    constexpr index_t GemmM1N1ThreadClusterN11 = 2;
    constexpr index_t GemmM1N1ThreadClusterM10 = 2;
    constexpr index_t GemmM1N1ThreadClusterN10 = 8;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM11 = 2;
#elif 0
    // cdata = 32, BlockSize = 64, 16x128x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmM1PerThread = 2;
    constexpr index_t GemmN1PerThread = 4;
    constexpr index_t GemmKPerThread  = 1;

    constexpr index_t GemmM1N1ThreadClusterM11 = 2;
    constexpr index_t GemmM1N1ThreadClusterN11 = 2;
    constexpr index_t GemmM1N1ThreadClusterM10 = 2;
    constexpr index_t GemmM1N1ThreadClusterN10 = 8;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM11 = 2;
#elif 0
    // cdata = 64, BlockSize = 64, 16x256x2
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 2;

    constexpr index_t GemmM1PerThread = 4;
    constexpr index_t GemmN1PerThread = 4;
    constexpr index_t GemmKPerThread  = 1;

    constexpr index_t GemmM1N1ThreadClusterM11 = 1;
    constexpr index_t GemmM1N1ThreadClusterN11 = 2;
    constexpr index_t GemmM1N1ThreadClusterM10 = 2;
    constexpr index_t GemmM1N1ThreadClusterN10 = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 2;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM11 = 4;
#elif 0
    // cdata = 64, BlockSize = 64, 16x256x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmM1PerThread = 4;
    constexpr index_t GemmN1PerThread = 4;
    constexpr index_t GemmKPerThread  = 1;

    constexpr index_t GemmM1N1ThreadClusterM11 = 2;
    constexpr index_t GemmM1N1ThreadClusterN11 = 2;
    constexpr index_t GemmM1N1ThreadClusterM10 = 1;
    constexpr index_t GemmM1N1ThreadClusterN10 = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM11 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 32x256x4
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmM1PerThread = 4;
    constexpr index_t GemmN1PerThread = 4;
    constexpr index_t GemmKPerThread  = 1;

    constexpr index_t GemmM1N1ThreadClusterM11 = 2;
    constexpr index_t GemmM1N1ThreadClusterN11 = 2;
    constexpr index_t GemmM1N1ThreadClusterM10 = 2;
    constexpr index_t GemmM1N1ThreadClusterN10 = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 32>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM11 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 32x256x8
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmM1PerThread = 4;
    constexpr index_t GemmN1PerThread = 4;
    constexpr index_t GemmKPerThread  = 1;

    constexpr index_t GemmM1N1ThreadClusterM11 = 2;
    constexpr index_t GemmM1N1ThreadClusterN11 = 2;
    constexpr index_t GemmM1N1ThreadClusterM10 = 2;
    constexpr index_t GemmM1N1ThreadClusterN10 = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 32>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 2;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<8, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM11 = 4;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x8
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmM1PerThread = 4;
    constexpr index_t GemmN1PerThread = 4;
    constexpr index_t GemmKPerThread  = 1;

    constexpr index_t GemmM1N1ThreadClusterM11 = 2;
    constexpr index_t GemmM1N1ThreadClusterN11 = 2;
    constexpr index_t GemmM1N1ThreadClusterM10 = 8;
    constexpr index_t GemmM1N1ThreadClusterN10 = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM11 = 4;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 16;

    constexpr index_t GemmM1PerThread = 4;
    constexpr index_t GemmN1PerThread = 4;
    constexpr index_t GemmKPerThread  = 1;

    constexpr index_t GemmM1N1ThreadClusterM11 = 2;
    constexpr index_t GemmM1N1ThreadClusterN11 = 2;
    constexpr index_t GemmM1N1ThreadClusterM10 = 8;
    constexpr index_t GemmM1N1ThreadClusterN10 = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 64>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 2;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<8, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM11 = 4;
#endif

    constexpr index_t GemmM1 =
        GemmM1PerThread * GemmM1N1ThreadClusterM11 * GemmM1N1ThreadClusterM10;
    constexpr index_t GemmN1 =
        GemmN1PerThread * GemmM1N1ThreadClusterN11 * GemmM1N1ThreadClusterN10;

    const auto descs =
#if 1
        transform_forward_convolution_into_gemm_v4r4_nhwc_kyxc_nhwk_pad
#else
        transform_forward_convolution_into_gemm_v4r4_nhwc_kyxc_nhwk_1x1
#endif
        <GemmMPerBlock, GemmNPerBlock, GemmM1, GemmN1>(wei_k_y_x_c_desc,
                                                       in_n_hi_wi_c_desc,
                                                       out_n_ho_wo_k_desc,
                                                       conv_strides,
                                                       conv_dilations,
                                                       in_left_pads,
                                                       in_right_pads);

    for(index_t i = 0; i < 5; ++i)
    {
        float ave_time = launch_kernel_dynamic_gemm_v1r1<
            BlockSize,
            TInWei,
            TAcc,
            TOut,
            InMemoryDataOperation::Set,
            decltype(descs[I0]),
            decltype(descs[I1]),
            decltype(descs[I2]),
            decltype(descs[I3]),
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmM1PerThread,
            GemmN1PerThread,
            GemmKPerThread,
            GemmM1N1ThreadClusterM10,
            GemmM1N1ThreadClusterN10,
            GemmM1N1ThreadClusterM11,
            GemmM1N1ThreadClusterN11,
            GemmABlockTransferThreadSliceLengths_GemmK_GemmM,
            GemmABlockTransferThreadClusterLengths_GemmK_GemmM,
            Sequence<1, 0>,
            Sequence<1, 0>,
            0,
            GemmABlockTransferSrcScalarPerVector_GemmK,
            GemmABlockTransferDstScalarPerVector_GemmM,
            false, // don't move back src coordinate after threadwise copy
            GemmBBlockTransferThreadSliceLengths_GemmK_GemmN,
            GemmBBlockTransferThreadClusterLengths_GemmK_GemmN,
            Sequence<1, 0>,
            Sequence<1, 0>,
            0,
            GemmBBlockTransferSrcScalarPerVector_GemmK,
            GemmBBlockTransferDstScalarPerVector_GemmN,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<2, 3, 0, 1>,
            1,
            GemmCThreadTransferDstScalarPerVector_GemmM11,
            decltype(descs[I4]),
            decltype(descs[I5]),
            decltype(descs[I6]),
            decltype(descs[I7]),
            decltype(descs[I8])>(static_cast<TInWei*>(wei_k_y_x_c_device_buf.GetDeviceBuffer()),
                                 static_cast<TInWei*>(in_n_hi_wi_c_device_buf.GetDeviceBuffer()),
                                 static_cast<TOut*>(out_n_ho_wo_k_device_buf.GetDeviceBuffer()),
                                 descs[I0],
                                 descs[I1],
                                 descs[I2],
                                 descs[I3],
                                 descs[I4],
                                 descs[I5],
                                 descs[I6],
                                 descs[I7],
                                 descs[I8],
                                 nrepeat);

        float perf = (float)(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    // copy result back to host
    out_n_ho_wo_k_device_buf.FromDevice(out_n_ho_wo_k.mData.data());
}
