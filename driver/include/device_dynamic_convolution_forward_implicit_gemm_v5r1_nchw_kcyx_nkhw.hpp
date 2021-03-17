#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_dynamic_convolution_forward_implicit_gemm_v5r1_nchw_kcyx_nkhw.hpp"

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_dynamic_convolution_forward_implicit_gemm_v5r1_nchw_kcyx_nkhw(InDesc,
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
    std::cout << "device_dynamic_convolution_forward_implicit_gemm_v5r1_nchw_kcyx_nkhw"
              << std::endl;

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

    // cdata = 16, BlockSize = 64, 16x64x4
    constexpr index_t BlockSize = 128;

    constexpr index_t KPerBlock   = 16;
    constexpr index_t HPerBlock   = 8;
    constexpr index_t WPerBlock   = 8;
    constexpr index_t CYXPerBlock = 4;

    constexpr index_t KPerThread   = 8;
    constexpr index_t HPerThread   = 1;
    constexpr index_t WPerThread   = 1;
    constexpr index_t CYXPerThread = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;

    constexpr auto conv_driver =
        DriverDynamicConvolutionForwardImplicitGemm_v5r1_nchw_kcyx_nkhw_pad<
            BlockSize,
            TDevice,
            TDevice,
            KPerBlock,
            HPerBlock,
            WPerBlock,
            CYXPerBlock,
            KPerThread,
            HPerThread,
            WPerThread,
            CYXPerThread,
            GemmABlockTransferThreadSliceLengths_GemmK_GemmM,
            GemmABlockTransferThreadClusterLengths_GemmK_GemmM,
            GemmABlockTransferSrcScalarPerVector_GemmK,
            GemmABlockTransferDstScalarPerVector_GemmM,
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
