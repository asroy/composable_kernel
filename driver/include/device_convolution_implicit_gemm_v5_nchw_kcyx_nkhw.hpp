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
//#include "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw.hpp"
#include "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_lds_double_buffer.hpp"
#include "gridwise_convolution_implicit_gemm_v4r4_xdlops_fp16_bfp16_nchw_kcyx_nkhw_lds_double_buffer.hpp"

#define CK_ENABLE_XDLOPS 0
#define CK_PARAM_PROBLEM_DIRECTION 2
#define CK_PARAM_EPACK_LENGTH 1
#define CK_PARAM_TUNABLE_BLOCK_SIZE 64
#define CK_PARAM_TUNABLE_K_PER_BLOCK 32
#define CK_PARAM_TUNABLE_B_PER_BLOCK 64
#define CK_PARAM_TUNABLE_E_PER_BLOCK 8
#define CK_PARAM_DEPENDENT_GRID_SIZE 2
#define CK_PARAM_GEMM_M_PER_WAVE 32
#define CK_PARAM_GEMM_N_PER_WAVE 64
#define CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_E 8
#define CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_B 8
#define CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E 4
#define CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K 16
#define CK_PARAM_PROBLEM_CONV_DILATION_W  1
#define CK_PARAM_PROBLEM_CONV_DILATION_H 1
#define CK_PARAM_PROBLEM_CONV_STRIDE_H 1
#define CK_PARAM_PROBLEM_CONV_STRIDE_W 1
#define CK_PARAM_IN_BLOCK_COPY_DATA_PER_ACCESS_B 1
#define CK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E 2
#define CK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_K 2
#define CK_PARAM_OUT_THREAD_COPY_DATA_PER_ACCESS_B 1


using namespace ck;

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations>
void device_convolution_implicit_gemm_v5_nchw_kcyx_nkhw(InDesc,
                                                        const Tensor<T>& in_nchw,
                                                        WeiDesc,
                                                        const Tensor<T>& wei_kcyx,
                                                        OutDesc,
                                                        Tensor<T>& out_nkhw,
                                                        ConvStrides,
                                                        ConvDilations,
                                                        index_t nrepeat)
{


    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc_org  = InDesc{};
    constexpr auto wei_kcyx_desc_org = WeiDesc{};
    constexpr auto out_nkhw_desc_org = OutDesc{};

    constexpr index_t Hi = in_nchw_desc_org.GetLength(I2);
    constexpr index_t Wi = in_nchw_desc_org.GetLength(I3);

    constexpr index_t N  = out_nkhw_desc_org.GetLength(I0);
    constexpr index_t Ho = out_nkhw_desc_org.GetLength(I2);
    constexpr index_t Wo = out_nkhw_desc_org.GetLength(I3);

    constexpr index_t K = wei_kcyx_desc_org.GetLength(I0);
    constexpr index_t C = wei_kcyx_desc_org.GetLength(I1);
    constexpr index_t Y = wei_kcyx_desc_org.GetLength(I2);
    constexpr index_t X = wei_kcyx_desc_org.GetLength(I3);

    constexpr index_t ConvStrideH = CK_PARAM_PROBLEM_CONV_STRIDE_H;
    constexpr index_t ConvStrideW = CK_PARAM_PROBLEM_CONV_STRIDE_W;

    constexpr index_t ConvDilationH = CK_PARAM_PROBLEM_CONV_DILATION_H;
    constexpr index_t ConvDilationW = CK_PARAM_PROBLEM_CONV_DILATION_W;

    // read params: tunable params
    constexpr index_t BlockSize = CK_PARAM_TUNABLE_BLOCK_SIZE;

    constexpr index_t BPerBlock = CK_PARAM_TUNABLE_B_PER_BLOCK;
    constexpr index_t KPerBlock = CK_PARAM_TUNABLE_K_PER_BLOCK;
    constexpr index_t EPerBlock = CK_PARAM_TUNABLE_E_PER_BLOCK;

    // read params: dependent params
    constexpr index_t GridSize = CK_PARAM_DEPENDENT_GRID_SIZE;

// calculate dependent params amd heuristic params
#if CK_PARAM_PROBLEM_DIRECTION == 2
    // In the WrW direction the filter is the output, while the output image is the input being
    // convolved with the (original) input image. This requires that the tensordescriptors be
    // swapped
    // To reuse the fwd kernel for this operation we need to swap the n and c dimension of the
    // input descriptor, the n and k dimension of the output descriptor
    // This change is necessary so that reduction dimensions are consistent with the requirement
    // of the wrw convolution when used in a fwd context
    printf("backward weight is executed\n");

    // constexpr auto tmp_in_nchw_desc =
    //     make_ConstantTensorDescriptor_packed(Sequence<N, C, Hi, Wi>{});
    // constexpr auto tmp_wei_kcyx_desc = make_ConstantTensorDescriptor_packed(Sequence<K, C, Y, X>{});
    // constexpr auto tmp_out_nkhw_desc =
    //     make_ConstantTensorDescriptor_packed(Sequence<N, K, Ho, Wo>{});
    // constexpr auto in_nchw_desc = tmp_in_nchw_desc.ReorderGivenNew2Old(Sequence<1, 0, 2, 3>{});
    // // wei and out are swapped in the solver
    // constexpr auto wei_kcyx_desc = tmp_out_nkhw_desc.ReorderGivenNew2Old(Sequence<1, 0, 2, 3>{});
    // constexpr auto out_nkhw_desc = tmp_wei_kcyx_desc.ReorderGivenNew2Old(Sequence<1, 0, 2, 3>{});
    constexpr auto dir           = ImplicitGemmDirection::BackwardWeight;

    constexpr auto in_nchw_desc  = make_ConstantTensorDescriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto wei_kcyx_desc = make_ConstantTensorDescriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto out_nkhw_desc = make_ConstantTensorDescriptor_packed(Sequence<N, K, Ho, Wo>{});

    // swap stride and dilation
    // using ConvDilations = Sequence<ConvStrideH, ConvStrideW>;
    // using ConvStrides   = Sequence<ConvDilationH, ConvDilationW>;
#else
    printf("forward data is executed\n");
    // calculate dependent params amd heuristic params
    constexpr auto in_nchw_desc  = make_ConstantTensorDescriptor_packed(Sequence<N, C, Hi, Wi>{});
    constexpr auto wei_kcyx_desc = make_ConstantTensorDescriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto out_nkhw_desc = make_ConstantTensorDescriptor_packed(Sequence<N, K, Ho, Wo>{});

    constexpr auto dir  = ImplicitGemmDirection::ForwardData;
    // using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    // using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;
#endif // CK_PARAM_PROBLEM_DIRECTION == 2

    constexpr index_t InBlockCopyClusterLengths_E = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_E;
    constexpr index_t InBlockCopyClusterLengths_B = CK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_B;

    constexpr index_t InBlockCopySubLengths_E = EPerBlock / InBlockCopyClusterLengths_E;
    constexpr index_t InBlockCopySubLengths_B = BPerBlock / InBlockCopyClusterLengths_B;

    constexpr index_t WeiBlockCopyClusterLengths_E = CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E;
    constexpr index_t WeiBlockCopyClusterLengths_K = CK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K;

    constexpr index_t WeiBlockCopySubLengths_E = EPerBlock / WeiBlockCopyClusterLengths_E;
    constexpr index_t WeiBlockCopySubLengths_K = KPerBlock / WeiBlockCopyClusterLengths_K;

    constexpr index_t EPack = CK_PARAM_EPACK_LENGTH;

#if MIOPEN_USE_FP32
    printf("fp32 is executed\n");
    using InBlockCopySubLengths_E_B = Sequence<InBlockCopySubLengths_E, InBlockCopySubLengths_B>;
    using InBlockCopyClusterLengths_E_B =
        Sequence<InBlockCopyClusterLengths_E, InBlockCopyClusterLengths_B>;

    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1>; // [E, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1>; // [E, B]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, B]

    using WeiBlockCopySubLengths_E_K = Sequence<WeiBlockCopySubLengths_E, WeiBlockCopySubLengths_K>;
    using WeiBlockCopyClusterLengths_E_K =
        Sequence<WeiBlockCopyClusterLengths_E, WeiBlockCopyClusterLengths_K>;

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]
#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
    using InBlockCopySubLengths_E_B =
        Sequence<InBlockCopySubLengths_E, InBlockCopySubLengths_B, EPack>;
    using InBlockCopyClusterLengths_E_B =
        Sequence<InBlockCopyClusterLengths_E, InBlockCopyClusterLengths_B, 1>;

    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 2>; // [E, B, EPack]
    using InBlockCopySrcAccessOrder            = Sequence<0, 1, 2>; // [E, B, EPack]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2>; // [E, B, EPack]

    using WeiBlockCopySubLengths_E_K =
        Sequence<WeiBlockCopySubLengths_E, WeiBlockCopySubLengths_K, EPack>;
    using WeiBlockCopyClusterLengths_E_K =
        Sequence<WeiBlockCopyClusterLengths_E, WeiBlockCopyClusterLengths_K, 1>;

    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0, 2>; // [K, E, EPack]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0, 2>; // [K, E, EPack]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1, 2>; // [E, K, EPack]
#endif

    constexpr index_t InBlockCopyDataPerAccess_B    = CK_PARAM_IN_BLOCK_COPY_DATA_PER_ACCESS_B;
    constexpr index_t WeiBlockCopySrcDataPerRead_E  = CK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = CK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_K;
    constexpr index_t OutThreadCopyDataPerAccess_B  = CK_PARAM_OUT_THREAD_COPY_DATA_PER_ACCESS_B;

    constexpr auto GemmMPerWave        = CK_PARAM_GEMM_M_PER_WAVE;
    constexpr auto GemmNPerWave        = CK_PARAM_GEMM_N_PER_WAVE;
    constexpr auto GemmMWaves          = KPerBlock / GemmMPerWave;
    constexpr auto GemmNWaves          = BPerBlock / GemmNPerWave;
    constexpr index_t GemmDataPerReadA = 1;
    constexpr index_t GemmDataPerReadB = 1;
        
    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_kcyx_device_buf(data_sz * wei_kcyx.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_kcyx_device_buf.ToDevice(wei_kcyx.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

// #if MIOPEN_USE_FP16 == 1
//     // ES set to 4 as dot4 operator is supported on fp16 in MI100
//     constexpr index_t ES = 4;
// #elif MIOPEN_USE_BFP16 == 1
//     // ES set to 2 as dot2 operator is supported on bfp16 in MI100
//     constexpr index_t ES = 2;
// #else
// // do nothing
// #endif

    // constexpr index_t GridSize =
    //     ((B + BPerBlock - 1) / BPerBlock) * ((K + KPerBlock - 1) / KPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(index_t i = 0; i < nrepeat; ++i)
    {
        constexpr auto gridwise_conv =
#if MIOPEN_USE_FP32 == 1
        GridwiseConvolutionImplicitGemm_v4r4_xdlops_nchw_kcyx_nkhw_lds_double_buffer<
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
            EPack,
            GemmMPerWave,
            GemmNPerWave,
            GemmMWaves,
            GemmNWaves,
            GemmDataPerReadA,
            GemmDataPerReadB,
            false,
            InBlockCopySubLengths_E_B,
            InBlockCopyClusterLengths_E_B,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopyDataPerAccess_B,
            WeiBlockCopySubLengths_E_K,
            WeiBlockCopyClusterLengths_E_K,
            WeiBlockCopyThreadClusterArrangeOrder,
            WeiBlockCopySrcAccessOrder,
            WeiBlockCopyDstAccessOrder,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_K,
            OutThreadCopyDataPerAccess_B,
            dir>{};
#elif MIOPEN_USE_FP16 == 1 || MIOPEN_USE_BFP16 == 1
        GridwiseConvolutionImplicitGemm_v4r4_xdlops_fp16_bfp16_nchw_kcyx_nkhw_lds_double_buffer<
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
            EPack,
            GemmMPerWave,
            GemmNPerWave,
            GemmMWaves,
            GemmNWaves,
            GemmDataPerReadA,
            GemmDataPerReadB,
            false,
            InBlockCopySubLengths_E_B,
            InBlockCopyClusterLengths_E_B,
            InBlockCopyThreadClusterArrangeOrder,
            InBlockCopySrcAccessOrder,
            InBlockCopyDstAccessOrder,
            InBlockCopyDataPerAccess_B,
            WeiBlockCopySubLengths_E_K,
            WeiBlockCopyClusterLengths_E_K,
            WeiBlockCopyThreadClusterArrangeOrder,
            WeiBlockCopySrcAccessOrder,
            WeiBlockCopyDstAccessOrder,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_K,
            OutThreadCopyDataPerAccess_B,
            dir>{};
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
