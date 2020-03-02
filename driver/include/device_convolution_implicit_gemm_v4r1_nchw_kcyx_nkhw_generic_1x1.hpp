#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "convolution_common.hpp"
#include "gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer_generic_1x1.hpp"

template <typename T,
          typename InDesc,
          typename WeiDesc,
          typename OutDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads>
void device_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_generic_1x1(InDesc,
                                                                      const Tensor<T>& in_nchw,
                                                                      WeiDesc,
                                                                      const Tensor<T>& wei_kcyx,
                                                                      OutDesc,
                                                                      Tensor<T>& out_nkhw,
                                                                      ConvStrides,
                                                                      ConvDilations,
                                                                      LeftPads,
                                                                      RightPads,
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

    constexpr index_t N  = out_nkhw_desc.GetLength(I0);
    constexpr index_t K  = out_nkhw_desc.GetLength(I1);
    constexpr index_t Ho = out_nkhw_desc.GetLength(I2);
    constexpr index_t Wo = out_nkhw_desc.GetLength(I3);

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_kcyx_device_buf(data_sz * wei_kcyx.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_kcyx_device_buf.ToDevice(wei_kcyx.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

    constexpr index_t Hi = in_nchw_desc.GetLength(I2);
    constexpr index_t Wi = in_nchw_desc.GetLength(I3);
    constexpr index_t C  = in_nchw_desc.GetLength(I1);
    constexpr index_t Y  = wei_kcyx_desc.GetLength(I2);
    constexpr index_t X  = wei_kcyx_desc.GetLength(I3);

    using TensorLengths_Hi_Wi_Y_X_Ho_Wo = Sequence<Hi, Wi, Y, X, Ho, Wo>;

#if 0
    // BlockSize = 256, EperBlock = 8, each thread hold 64 data
    constexpr index_t BlockSize = 256;

    constexpr index_t BPerBlock = 16;
    constexpr index_t KPerBlock = 128;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 1, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<8, 2, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 128>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 1
    // BlockSize = 256, EPerBlock = 16, each thread hold 64 data
    constexpr index_t BlockSize = 256;

    constexpr index_t BPerBlock = 16;
    constexpr index_t KPerBlock = 128;
    constexpr index_t EPerBlock = 16;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<16, 1, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 2>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<4, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 2;
#elif 1
    // BlockSize = 256, EPerBlock = 16, each thread hold 64 data
    // for 1x1
    constexpr index_t BlockSize = 256;

    constexpr index_t BPerBlock = 16;
    constexpr index_t KPerBlock = 128;
    constexpr index_t EPerBlock = 16;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<4, 1, 1, 2>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<4, 2, 16, 2>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 2;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 2>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<4, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 2;
#elif 1
    // BlockSize = 64, each thread hold 64 data
    constexpr index_t BlockSize = 64;

    constexpr index_t BPerBlock = 8;
    constexpr index_t KPerBlock = 64;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 2;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<8, 1, 8, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 2>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 32>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 0
    // BlockSize = 256, blockwise-GEMM 64x128, each thread hold 32 data
    constexpr index_t BlockSize = 256;

    constexpr index_t BPerBlock = 16;
    constexpr index_t KPerBlock = 64;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThreadSubC = 2;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 2;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 1, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<8, 2, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<2, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<4, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 2;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#endif

    constexpr index_t N1 = GemmNRepeat;
    constexpr index_t N2 = GemmNPerThreadSubC;

    constexpr index_t B = (N * Ho * Wo) / (N1 * N2);

    constexpr index_t GridSize =
        ((B + BPerBlock - 1) / BPerBlock) * ((K + KPerBlock - 1) / KPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    static_assert(LeftPads{}[0] == RightPads{}[0] && LeftPads{}[1] == RightPads{}[1],
                  "currently assume Pads X&Y in left&right is the same");

    constexpr auto gridwise_conv =
        GridwiseConvolutionImplicitGemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer_generic_1x1<
            BlockSize,
            T,
            T,
            ConvolutionDirection::Forward,
            BPerBlock,
            KPerBlock,
            EPerBlock,
            GemmNRepeat,
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

    for(index_t i = 0; i < nrepeat; ++i)
    {
        float time =
            launch_and_time_kernel(run_gridwise_operation<decltype(gridwise_conv),
                                                          const T* const __restrict__,
                                                          const T* const __restrict__,
                                                          T* const __restrict__,
                                                          index_t,
                                                          index_t,
                                                          index_t,
                                                          index_t,
                                                          index_t,
                                                          index_t,
                                                          index_t,
                                                          index_t,
                                                          index_t,
                                                          index_t,
                                                          index_t,
                                                          index_t,
                                                          index_t>,
                                   dim3(GridSize),
                                   dim3(BlockSize),
                                   0,
                                   0,
                                   gridwise_conv,
                                   const_cast<const T* const __restrict__>(
                                       static_cast<T*>(in_nchw_device_buf.GetDeviceBuffer())),
                                   const_cast<const T* const __restrict__>(
                                       static_cast<T*>(wei_kcyx_device_buf.GetDeviceBuffer())),
                                   const_cast<T* const __restrict__>(
                                       static_cast<T*>(out_nkhw_device_buf.GetDeviceBuffer())),
                                   Hi,
                                   Wi,
                                   N,
                                   K,
                                   C,
                                   Ho,
                                   Wo,
                                   ConvStrides{}[0],
                                   ConvStrides{}[1],
                                   ConvDilations{}[0],
                                   ConvDilations{}[1],
                                   LeftPads{}[0],
                                   LeftPads{}[1]);

        printf("Elapsed time : %f ms, %f TFlop/s\n",
               time,
               (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                   (std::size_t(1000) * 1000 * 1000) / time);
        usleep(std::min(time * 1000, float(10000)));
    }

    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
