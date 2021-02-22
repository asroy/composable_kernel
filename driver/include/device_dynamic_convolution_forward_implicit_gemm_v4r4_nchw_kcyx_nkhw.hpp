#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_dynamic_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"

template <typename T>
__host__ __device__ constexpr auto sequence_to_tuple_of_number(const T& x)
{
    using namespace ck;

    return generate_tuple(
        [&](auto i) {
            constexpr index_t tmp = T::At(i);
            return Number<tmp>{};
        },
        T::Size());
}

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_dynamic_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw(InDesc,
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

    using TDevice = typename conditional<is_same<half_float::half, T>::value, half_t, T>::type;

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_kcyx_device_buf(data_sz * wei_kcyx.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_kcyx_device_buf.ToDevice(wei_kcyx.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

#if 0
    // run-time variables
    const auto in_n_c_hi_wi_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(to_multi_index(InDesc::GetLengths()));
    const auto wei_k_c_y_x_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(to_multi_index(WeiDesc::GetLengths()));
    const auto out_n_k_ho_wo_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(to_multi_index(OutDesc::GetLengths()));

    const auto conv_strides   = to_multi_index(ConvStrides{});
    const auto conv_dilations = to_multi_index(ConvDilations{});
    const auto in_left_pads   = to_multi_index(InLeftPads{});
    const auto in_right_pads  = to_multi_index(InRightPads{});
#else
    // compile-time variables
    const auto in_n_c_hi_wi_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
        sequence_to_tuple_of_number(InDesc::GetLengths()));
    const auto wei_k_c_y_x_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
        sequence_to_tuple_of_number(WeiDesc::GetLengths()));
    const auto out_n_k_ho_wo_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
        sequence_to_tuple_of_number(OutDesc::GetLengths()));

    const auto conv_strides   = sequence_to_tuple_of_number(ConvStrides{});
    const auto conv_dilations = sequence_to_tuple_of_number(ConvDilations{});
    const auto in_left_pads   = sequence_to_tuple_of_number(InLeftPads{});
    const auto in_right_pads  = sequence_to_tuple_of_number(InRightPads{});
#endif

#if 0
    // cdata = 64, BlockSize = 256, 128x128x2
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<1, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 256, 128x128x4
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 2;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<2, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x8
    // b thread copy 4x1
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x8
    // b thread copy 2x2
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 2;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<2, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<4, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x16
    // GemmBBlockCopySrcDataPerRead_GemmN = 4
    // GemmCThreadCopyDstDataPerWrite_GemmN1 = 4
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 16;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 64>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<8, 32>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 4;
#endif

    constexpr auto conv_driver =
#if 1
        DriverDynamicConvolutionForwardImplicitGemm_v4r4_nchw_kcyx_nkhw_pad
#elif 0
        DriverDynamicConvolutionForwardImplicitGemm_v4r4_nchw_kcyx_nkhw_no_pad
#elif 1
        DriverDynamicConvolutionForwardImplicitGemm_v4r4_nchw_kcyx_nkhw_1x1
#endif
        <BlockSize,
         TDevice,
         TDevice,
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
         GemmABlockTransferThreadSliceLengths_GemmK_GemmM,
         GemmABlockTransferThreadClusterLengths_GemmK_GemmM,
         GemmABlockTransferSrcScalarPerVector_GemmK,
         GemmABlockTransferDstScalarPerVector_GemmM,
         GemmBBlockTransferThreadSliceLengths_GemmK_GemmN,
         GemmBBlockTransferThreadClusterLengths_GemmK_GemmN,
         GemmBBlockTransferSrcScalarPerVector_GemmN,
         GemmBBlockTransferDstScalarPerVector_GemmN,
         GemmCThreadTransferDstScalarPerVector_GemmN1>{};

    conv_driver.Run(wei_k_c_y_x_desc,
                    in_n_c_hi_wi_desc,
                    out_n_k_ho_wo_desc,
                    conv_strides,
                    conv_dilations,
                    in_left_pads,
                    in_right_pads,
                    static_cast<TDevice*>(wei_kcyx_device_buf.GetDeviceBuffer()),
                    static_cast<TDevice*>(in_nchw_device_buf.GetDeviceBuffer()),
                    static_cast<TDevice*>(out_nkhw_device_buf.GetDeviceBuffer()));

    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
