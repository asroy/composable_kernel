#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_dynamic_convolution_forward_implicit_gemm_v4r4_chwn_cyxk_khwn.hpp"

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_dynamic_convolution_forward_implicit_gemm_v4r4_chwn_cyxk_khwn(InDesc,
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
    std::cout << "device_dynamic_convolution_forward_implicit_gemm_v4r4_chwn_cyxk_khwn"
              << std::endl;

    using namespace ck;

    using TDevice = typename conditional<is_same<half_float::half, T>::value, half_t, T>::type;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto N = OutDesc::GetLengths()[I0];
    constexpr auto K = OutDesc::GetLengths()[I1];
    constexpr auto C = WeiDesc::GetLengths()[I1];

    constexpr auto Hi = InDesc::GetLengths()[I2];
    constexpr auto Wi = InDesc::GetLengths()[I3];

    constexpr auto Ho = OutDesc::GetLengths()[I2];
    constexpr auto Wo = OutDesc::GetLengths()[I3];

    constexpr auto Y = WeiDesc::GetLengths()[I2];
    constexpr auto X = WeiDesc::GetLengths()[I3];

#if 0
    // run-time variables
    constexpr auto in_n_hi_wi_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(N, Hi, Wi, C));
    constexpr auto wei_k_y_x_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(K, Y, X, C));
    constexpr auto out_n_ho_wo_k_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(N, Ho, Wo, K));

    const auto conv_strides   = to_multi_index(ConvStrides{});
    const auto conv_dilations = to_multi_index(ConvDilations{});
    const auto in_left_pads   = to_multi_index(InLeftPads{});
    const auto in_right_pads  = to_multi_index(InRightPads{});
#else
    // compile-time variables
    constexpr auto in_c_hi_wi_n_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(C, Hi, Wi, N));
    constexpr auto wei_c_y_x_k_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(C, Y, X, K));
    constexpr auto out_k_ho_wo_n_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, Ho, Wo, N));

    const auto conv_strides   = sequence_to_tuple_of_number(ConvStrides{});
    const auto conv_dilations = sequence_to_tuple_of_number(ConvDilations{});
    const auto in_left_pads   = sequence_to_tuple_of_number(InLeftPads{});
    const auto in_right_pads  = sequence_to_tuple_of_number(InRightPads{});
#endif

    Tensor<float> in_chwn(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<C, Hi, Wi, N>{})));
    Tensor<float> wei_cyxk(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<C, Y, X, K>{})));
    Tensor<float> out_khwn(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<K, Ho, Wo, N>{})));

    auto f_nchw2chwn = [&](auto c, auto hi, auto wi, auto n) {
        in_chwn(c, hi, wi, n) = in_nchw(n, c, hi, wi);
    };

    auto f_kcyx2cyxk = [&](auto c, auto y, auto x, auto k) {
        wei_cyxk(c, y, x, k) = wei_kcyx(k, c, y, x);
    };

    auto f_nkhw2khwn = [&](auto k, auto ho, auto wo, auto n) {
        out_khwn(k, ho, wo, n) = out_nkhw(n, k, ho, wo);
    };

    make_ParallelTensorFunctor(f_nchw2chwn, C, Hi, Wi, N)(std::thread::hardware_concurrency());
    make_ParallelTensorFunctor(f_kcyx2cyxk, C, Y, X, K)(std::thread::hardware_concurrency());
    make_ParallelTensorFunctor(f_nkhw2khwn, K, Ho, Wo, N)(std::thread::hardware_concurrency());

    std::size_t data_sz = sizeof(T);

    DeviceMem in_chwn_device_buf(data_sz * in_chwn.mDesc.GetElementSpace());
    DeviceMem wei_cyxk_device_buf(data_sz * wei_cyxk.mDesc.GetElementSpace());
    DeviceMem out_khwn_device_buf(data_sz * out_khwn.mDesc.GetElementSpace());

    in_chwn_device_buf.ToDevice(in_chwn.mData.data());
    wei_cyxk_device_buf.ToDevice(wei_cyxk.mData.data());
    out_khwn_device_buf.ToDevice(out_khwn.mData.data());

    // cdata = 16, BlockSize = 64, 16x64x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 2;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = (GemmMPerBlock) / (GemmMPerThread * GemmMLevel0Cluster * 2);
    constexpr index_t GemmNLevel1Cluster = (GemmNPerBlock) / (GemmNPerThread * GemmNLevel0Cluster * 2);

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<1, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<4, 16>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 1;

    constexpr auto conv_driver =
        DriverDynamicConvolutionForwardImplicitGemm_v4r4_chwn_cyxk_khwn_pad
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
         GemmCThreadTransferDstScalarPerVector_GemmM1>{};

    conv_driver.Run(wei_c_y_x_k_desc,
                    in_c_hi_wi_n_desc,
                    out_k_ho_wo_n_desc,
                    conv_strides,
                    conv_dilations,
                    in_left_pads,
                    in_right_pads,
                    static_cast<TDevice*>(wei_cyxk_device_buf.GetDeviceBuffer()),
                    static_cast<TDevice*>(in_chwn_device_buf.GetDeviceBuffer()),
                    static_cast<TDevice*>(out_khwn_device_buf.GetDeviceBuffer()));

    out_khwn_device_buf.FromDevice(out_khwn.mData.data());

    auto f_khwn2nkhw = [&](auto n, auto k, auto ho, auto wo) {
        out_nkhw(n, k, ho, wo) = out_khwn(k, ho, wo, n);
    };

    make_ParallelTensorFunctor(f_khwn2nkhw, N, K, Ho, Wo)(std::thread::hardware_concurrency());
}
