#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_convolution_backward_data_implicit_gemm_v4r1_nchw_kcyx_nkhw.hpp"

namespace launcher {

using namespace ck;

template <typename GridwiseOp, index_t GemmId, typename... Xs>
__global__ void run_gridwise_convolution_backward_data_v4r1(Xs... xs)
{
    GridwiseOp::template Run<GemmId>(xs...);
}

template <typename T,
          typename InDesc,
          typename WeiDesc,
          typename OutDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_convolution_backward_data_implicit_gemm_v4r1_nchw_kcyx_nkhw(InDesc in_nchw_desc,
                                                                        Tensor<T>& in_nchw,
                                                                        WeiDesc wei_kcyx_desc,
                                                                        const Tensor<T>& wei_kcyx,
                                                                        OutDesc out_nkhw_desc,
                                                                        const Tensor<T>& out_nkhw,
                                                                        ConvStrides,
                                                                        ConvDilations,
                                                                        InLeftPads,
                                                                        InRightPads,
                                                                        std::size_t nrepeat)
{
    constexpr index_t N = out_nkhw_desc.GetLengths()[0];
    constexpr index_t K = out_nkhw_desc.GetLengths()[1];
    constexpr index_t C = wei_kcyx_desc.GetLengths()[1];

    constexpr index_t Hi = in_nchw_desc.GetLengths()[2];
    constexpr index_t Wi = in_nchw_desc.GetLengths()[3];

    constexpr index_t Ho = out_nkhw_desc.GetLengths()[2];
    constexpr index_t Wo = out_nkhw_desc.GetLengths()[3];

    constexpr index_t Y = wei_kcyx_desc.GetLengths()[2];
    constexpr index_t X = wei_kcyx_desc.GetLengths()[3];

    constexpr index_t ConvStrideH = ConvStrides{}[0];
    constexpr index_t ConvStrideW = ConvStrides{}[1];

    constexpr index_t ConvDilationH = ConvDilations{}[0];
    constexpr index_t ConvDilationW = ConvDilations{}[1];

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_kcyx_device_buf(data_sz * wei_kcyx.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_kcyx_device_buf.ToDevice(wei_kcyx.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

#if 1
    // BlockSize = 256, each thread hold 64 data
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock              = 128;
    constexpr index_t GemmNPerBlock              = 128;
    constexpr index_t GemmKPerBlock              = 8;
    constexpr index_t GemmMPerThread             = 4;
    constexpr index_t GemmNPerThread             = 4;
    constexpr index_t GemmKPerThread             = 1;
    constexpr index_t GemmMLevel0Cluster         = 4;
    constexpr index_t GemmNLevel0Cluster         = 4;
    constexpr index_t GemmMLevel1Cluster         = 4;
    constexpr index_t GemmNLevel1Cluster         = 4;
    constexpr index_t GemmThreadGemmDataPerReadM = 4;
    constexpr index_t GemmThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 1
    // BlockSize = 256, each thread hold 64 data
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock              = 128;
    constexpr index_t GemmNPerBlock              = 128;
    constexpr index_t GemmKPerBlock              = 16;
    constexpr index_t GemmMPerThread             = 4;
    constexpr index_t GemmNPerThread             = 4;
    constexpr index_t GemmKPerThread             = 1;
    constexpr index_t GemmMLevel0Cluster         = 4;
    constexpr index_t GemmNLevel0Cluster         = 4;
    constexpr index_t GemmMLevel1Cluster         = 4;
    constexpr index_t GemmNLevel1Cluster         = 4;
    constexpr index_t GemmThreadGemmDataPerReadM = 4;
    constexpr index_t GemmThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<8, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<8, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#endif

    constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
    constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

    constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
    constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

    constexpr index_t YDot = math::integer_divide_ceil(Y, YTilda);
    constexpr index_t XDot = math::integer_divide_ceil(X, XTilda);

    constexpr index_t HTilda = Ho + math::integer_divide_ceil(ConvDilationH * (Y - 1), ConvStrideH);
    constexpr index_t WTilda = Wo + math::integer_divide_ceil(ConvDilationW * (X - 1), ConvStrideW);

    constexpr index_t HTildaLeft = math::integer_divide_floor(
        math::max(0, InLeftPads{}[0] - ConvDilationH * (YTilda - 1)), ConvStrides{}[0]);
    constexpr index_t WTildaLeft = math::integer_divide_floor(
        math::max(0, InLeftPads{}[1] - ConvDilationW * (XTilda - 1)), ConvStrides{}[1]);

    constexpr index_t HTildaRight = math::min(
        HTilda, math::integer_divide_ceil(InLeftPads{}[0] + Hi - 1, ConvStrides{}[0]) + 1);
    constexpr index_t WTildaRight = math::min(
        WTilda, math::integer_divide_ceil(InLeftPads{}[1] + Wi - 1, ConvStrides{}[1]) + 1);

    constexpr index_t HTildaSlice = HTildaRight - HTildaLeft;
    constexpr index_t WTildaSlice = WTildaRight - WTildaLeft;

    constexpr index_t GemmM = C;
    constexpr index_t GemmN = N * HTildaSlice * WTildaSlice;

    constexpr index_t GridSize = math::integer_divide_ceil(GemmM, GemmMPerBlock) *
                                 math::integer_divide_ceil(GemmN, GemmNPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(index_t i = 0; i < nrepeat; ++i)
    {
        using GridwiseConvBwdData = GridwiseConvolutionBackwardDataImplicitGemm_v4r1_nchw_kcyx_nkhw<
            GridSize,
            BlockSize,
            T,
            T,
            decltype(in_nchw_desc),
            decltype(wei_kcyx_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            ConvDilations,
            InLeftPads,
            InRightPads,
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerThread,
            GemmNPerThread,
            GemmKPerThread,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmABlockCopyDstDataPerWrite_GemmM,
            GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN,
            GemmCThreadCopyDstDataPerWrite_GemmN1>;

        KernelTimer timer;
        timer.Start();

        static_for<0, GridwiseConvBwdData::GetNumberOfGemm(), 1>{}([&](auto gemm_id_) {
            constexpr index_t gemm_id = decltype(gemm_id_){};

            constexpr auto gemm_sizes        = GridwiseConvBwdData::GetGemmSize(gemm_id);
            constexpr index_t gemm_k         = gemm_sizes.At(2);
            constexpr bool is_gemm_not_empty = gemm_k > 0;

            // only compile and run if GEMM is no empty
            static_if<is_gemm_not_empty>{}([&](auto fwd) {
                launch_kernel(
                    run_gridwise_convolution_backward_data_v4r1<GridwiseConvBwdData,
                                                                fwd(gemm_id),
                                                                T* const __restrict__,
                                                                const T* const __restrict__,
                                                                const T* const __restrict__>,
                    dim3(GridSize),
                    dim3(BlockSize),
                    0,
                    0,
                    static_cast<T*>(in_nchw_device_buf.GetDeviceBuffer()),
                    static_cast<T*>(wei_kcyx_device_buf.GetDeviceBuffer()),
                    static_cast<T*>(out_nkhw_device_buf.GetDeviceBuffer()));
            });
        });

        timer.End();
        float time = timer.GetElapsedTime();

        printf("Elapsed time : %f ms, %f TFlop/s\n",
               time,
               (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                   (std::size_t(1000) * 1000 * 1000) / time);
        usleep(std::min(time * 1000, float(10000)));
    }

    in_nchw_device_buf.FromDevice(in_nchw.mData.data());
}

} // namespace launcher
