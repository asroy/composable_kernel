#pragma once
#include <unistd.h>

#define MIOPEN_USE_FP16 0
#define MIOPEN_USE_BFP16 0
#define MIOPEN_USE_FP32 1

#define __HIP_PLATFORM_HCC__ 1

#include "float_types.h"
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_convolution_kernel_wrapper.hpp"
#include "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw.hpp"
#include "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer.hpp"
#include "gridwise_convolution_implicit_gemm_v4_fp16_bfp16_nchw_kcyx_nkhw_lds_double_buffer.hpp"

#define CK_PARAM_TUNABLE_K_PER_BLOCK 64

using namespace ck;

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations>
void device_convolution_implicit_gemm_v4_nchw_kcyx_nkhw(InDesc,
                                                        const Tensor<T>& in_nchw,
                                                        WeiDesc,
                                                        const Tensor<T>& wei_kcyx,
                                                        OutDesc,
                                                        Tensor<T>& out_nkhw,
                                                        ConvStrides,
                                                        ConvDilations,
                                                        index_t nrepeat)
{

    
    // read params: problem decription

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc  = InDesc{};
    constexpr auto wei_kcyx_desc = WeiDesc{};
    constexpr auto out_nkhw_desc = OutDesc{};

    constexpr index_t Hi = in_nchw_desc.GetLength(I2);
    constexpr index_t Wi = in_nchw_desc.GetLength(I3);

    constexpr index_t N  = out_nkhw_desc.GetLength(I0);
    constexpr index_t Ho = out_nkhw_desc.GetLength(I2);
    constexpr index_t Wo = out_nkhw_desc.GetLength(I3);

    constexpr index_t K = wei_kcyx_desc.GetLength(I0);
    constexpr index_t C = wei_kcyx_desc.GetLength(I1);
    constexpr index_t Y = wei_kcyx_desc.GetLength(I2);
    constexpr index_t X = wei_kcyx_desc.GetLength(I3);

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_kcyx_device_buf(data_sz * wei_kcyx.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_kcyx_device_buf.ToDevice(wei_kcyx.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

    constexpr index_t N1 = 2;
    constexpr index_t N2 = 4;

    constexpr index_t B = (N * Ho * Wo) / (N1 * N2);

    constexpr index_t BPerBlock = 16;
    constexpr index_t KPerBlock = K % 128 == 0 ? 128 : (K % 64 == 0 ? 64 : 32);
    constexpr index_t BlockSize = K % 128 == 0 ? 256 : (K % 64 == 0 ? 128 : 64);

#if MIOPEN_USE_FP16 == 1
    // ES set to 4 as dot4 operator is supported on fp16 in MI100
    constexpr index_t ES = 4;
#elif MIOPEN_USE_BFP16 == 1
    // ES set to 2 as dot2 operator is supported on bfp16 in MI100
    constexpr index_t ES = 2;
#else
// do nothing
#endif

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

#if MIOPEN_USE_FP32 == 1
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]
#elif MIOPEN_USE_FP16 == 1 || MIOPEN_USE_BFP16 == 1
    // ES - E dimension is folded into 2 dimensions E and ES
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2, 4>; // [E, N1, N2, B, ES]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1, 3, 2, 4>; // [E, N1, N2, B, ES]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3, 4>; // [E, N1, B, N2, ES]

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0, 2>; // [K, E, ES]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0, 2>; // [K, E, ES]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1, 2>; // [E, K, ES]
#endif

#if CK_PARAM_TUNABLE_K_PER_BLOCK == 32

    constexpr index_t EPerBlock = 4;

    constexpr index_t GemmMLevel0Cluster = 1;

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 1;

#if MIOPEN_USE_FP32 == 1
    // all_of(X_Per_Block % (X_Sub_Length * X_Cluster_Length) == 0)
    // accumulate(X_Cluster_Lengths, multiply) == BlockSize
    using InBlockCopySubLengths_E_N1_B_N2     = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2 = Sequence<4, 1, 16, 1>;
    using WeiBlockCopySubLengths_E_K          = Sequence<2, 1>;
    using WeiBlockCopyClusterLengths_E_K      = Sequence<2, 32>;
#elif MIOPEN_USE_FP16 == 1 || MIOPEN_USE_BFP16 == 1
    using InBlockCopySubLengths_E_N1_B_N2_ES     = Sequence<1, 2, 1, 4, ES>;
    using InBlockCopyClusterLengths_E_N1_B_N2_ES = Sequence<4, 1, 16, 1, 1>;
    using WeiBlockCopySubLengths_E_K_ES          = Sequence<2, 1, ES>;
    using WeiBlockCopyClusterLengths_E_K_ES      = Sequence<2, 32, 1>;
#endif // MIOPEN_USE_FP32 == 1

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 1;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;

#elif CK_PARAM_TUNABLE_K_PER_BLOCK == 64

    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmMLevel0Cluster = 2;

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 1;

#if MIOPEN_USE_FP32 == 1
    using InBlockCopySubLengths_E_N1_B_N2           = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2       = Sequence<8, 1, 16, 1>;
    using WeiBlockCopySubLengths_E_K                = Sequence<4, 1>;
    using WeiBlockCopyClusterLengths_E_K            = Sequence<2, 64>;
#elif MIOPEN_USE_FP16 == 1 || MIOPEN_USE_BFP16 == 1
    // ES - E dimension is folded into 2 dimensions E and ES
    using InBlockCopySubLengths_E_N1_B_N2_ES     = Sequence<1, 2, 1, 4, ES>;
    using InBlockCopyClusterLengths_E_N1_B_N2_ES = Sequence<8, 1, 16, 1, 1>;
    using WeiBlockCopySubLengths_E_K_ES          = Sequence<4, 1, ES>;
    using WeiBlockCopyClusterLengths_E_K_ES      = Sequence<2, 64, 1>;
#endif // MIOPEN_USE_FP32 == 1

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 1;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;

#elif CK_PARAM_TUNABLE_K_PER_BLOCK == 128
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmMLevel0Cluster = 4;

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 1;

#if MIOPEN_USE_FP32 == 1
    using InBlockCopySubLengths_E_N1_B_N2           = Sequence<1, 1, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2       = Sequence<8, 2, 16, 1>;
    using WeiBlockCopySubLengths_E_K                = Sequence<4, 1>;
    using WeiBlockCopyClusterLengths_E_K            = Sequence<2, 128>;
#elif MIOPEN_USE_FP16 == 1 || MIOPEN_USE_BFP16 == 1
    // ES - E dimension is folded into 2 dimensions E and ES
    using InBlockCopySubLengths_E_N1_B_N2_ES     = Sequence<1, 1, 1, 4, ES>;
    using InBlockCopyClusterLengths_E_N1_B_N2_ES = Sequence<8, 2, 16, 1, 1>;
    using WeiBlockCopySubLengths_E_K_ES          = Sequence<4, 1, ES>;
    using WeiBlockCopyClusterLengths_E_K_ES      = Sequence<2, 128, 1>;
#endif // MIOPEN_USE_FP32 == 1

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 1;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#else
    static_assert(false, "wrong! Only kperblock could be 32/64/128 not supported");
#endif // CK_PARAM_TUNABLE_K_PER_BLOCK == 32


    constexpr index_t GridSize =
        ((B + BPerBlock - 1) / BPerBlock) * ((K + KPerBlock - 1) / KPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(index_t i = 0; i < nrepeat; ++i)
    {
        constexpr auto gridwise_conv =
#if MIOPEN_USE_FP32 == 1
        GridwiseConvolutionImplicitGemm_v4_nchw_kcyx_nkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            FLOAT,
            FLOAT_ACCUM,
            decltype(in_nchw_desc),
            decltype(wei_kcyx_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            ConvDilations,
            BPerBlock,
            KPerBlock,
            EPerBlock,
            N1,
            N2,
            GemmMPerThreadSubC,
            GemmNPerThreadSubC,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmKPerThreadLoop,
            GemmDataPerReadA,
            GemmDataPerReadB,
            InBlockCopySubLengths_E_N1_B_N2,
            InBlockCopyClusterLengths_E_N1_B_N2,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopySrcDataPerRead_B,
            InBlockCopyDstDataPerWrite_N2,
            WeiBlockCopySubLengths_E_K,
            WeiBlockCopyClusterLengths_E_K,
            WeiBlockCopyThreadClusterArrangeOrder,
            WeiBlockCopySrcAccessOrder,
            WeiBlockCopyDstAccessOrder,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_K>{};
#elif MIOPEN_USE_FP16 == 1 || MIOPEN_USE_BFP16 == 1
        GridwiseConvolutionImplicitGemm_v4_fp16_bfp16_nchw_kcyx_nkhw_lds_double_buffer<
            GridSize,
            BlockSize,
            half,
            float,
            decltype(in_nchw_desc),
            decltype(wei_kcyx_desc),
            decltype(out_nkhw_desc),
            ConvStrides,
            ConvDilations,
            BPerBlock,
            KPerBlock,
            EPerBlock,
            N1,
            N2,
            ES,
            GemmMPerThreadSubC,
            GemmNPerThreadSubC,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmKPerThreadLoop,
            GemmDataPerReadA,
            GemmDataPerReadB,
            InBlockCopySubLengths_E_N1_B_N2_ES,
            InBlockCopyClusterLengths_E_N1_B_N2_ES,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopySrcDataPerRead_B,
            InBlockCopyDstDataPerWrite_N2,
            WeiBlockCopySubLengths_E_K_ES,
            WeiBlockCopyClusterLengths_E_K_ES,
            WeiBlockCopyThreadClusterArrangeOrder,
            WeiBlockCopySrcAccessOrder,
            WeiBlockCopyDstAccessOrder,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_K>{};
#endif 

        float time = launch_kernel(run_gridwise_convolution_kernel<decltype(gridwise_conv), T>,
                                   dim3(GridSize),
                                   dim3(BlockSize),
                                   0,
                                   static_cast<T*>(in_nchw_device_buf.GetDeviceBuffer()),
                                   static_cast<T*>(wei_kcyx_device_buf.GetDeviceBuffer()),
                                   static_cast<T*>(out_nkhw_device_buf.GetDeviceBuffer()));

        printf("Elapsed time : %f ms, %f TFlop/s\n",
               time,
               (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                   (std::size_t(1000) * 1000 * 1000) / time);
        usleep(std::min(time * 1000, float(10000)));
    }

    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
