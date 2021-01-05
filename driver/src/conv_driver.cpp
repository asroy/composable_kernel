#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
#include "device_convolution_forward_implicit_gemm_v4r1_nchw_kcyx_nkhw.hpp"
#include "device_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"
#include "device_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw.hpp"
#include "device_dynamic_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"
#include "device_dummy_static_transform.hpp"
#include "device_dummy_dynamic_transform_v1.hpp"
#include "device_dummy_dynamic_transform.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;

    // 1x1, 56x56
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t HI = 56;
    constexpr index_t WI = 56;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;

    auto in_nchw_desc  = make_native_tensor_descriptor_packed(Sequence<N, C, HI, WI>{});
    auto wei_kcyx_desc = make_native_tensor_descriptor_packed(Sequence<K, C, Y, X>{});
    auto out_nkhw_desc = get_convolution_output_default_4d_tensor_descriptor(
        in_nchw_desc, wei_kcyx_desc, ConvStrides{}, ConvDilations{}, LeftPads{}, RightPads{});

    ostream_tensor_descriptor(in_nchw_desc, std::cout << "in_nchw_desc: ");
    ostream_tensor_descriptor(wei_kcyx_desc, std::cout << "wei_kcyx_desc: ");
    ostream_tensor_descriptor(out_nkhw_desc, std::cout << "out_nkhw_desc: ");
    print_array("LeftPads", to_multi_index(LeftPads{}));
    print_array("RightPads", to_multi_index(RightPads{}));
    print_array("ConvStrides", to_multi_index(ConvStrides{}));
    print_array("ConvDilations", to_multi_index(ConvDilations{}));

#if 1
    using in_data_t  = float;
    using out_data_t = float;
#else
    using in_data_t  = half_float::half;
    using out_data_t = half_float::half;
#endif

    Tensor<in_data_t> in_nchw(make_HostTensorDescriptor(in_nchw_desc));
    Tensor<in_data_t> wei_kcyx(make_HostTensorDescriptor(wei_kcyx_desc));
    Tensor<out_data_t> out_nkhw_host(make_HostTensorDescriptor(out_nkhw_desc));
    Tensor<out_data_t> out_nkhw_device(make_HostTensorDescriptor(out_nkhw_desc));

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(argc != 3)
    {
        printf("arg1: do_verification, arg2: nrepeat\n");
        exit(1);
    }

    bool do_verification = atoi(argv[1]);
    index_t nrepeat      = atoi(argv[2]);

    if(do_verification)
    {
#if 0
        in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
#elif 0
        in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_3{}, num_thread);
#elif 0
        in_nchw.GenerateTensorValue(GeneratorTensor_3{}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
#elif 1
        in_nchw.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
#elif 0
        in_nchw.GenerateTensorValue(GeneratorTensor_2{1, 5}, num_thread);

        auto gen_wei = [](auto... is) {
            return GeneratorTensor_2{1, 5}(is...) * GeneratorTensor_Checkboard{}(is...);
        };
        wei_kcyx.GenerateTensorValue(gen_wei, num_thread);
#endif
    }

    gridwise_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw(in_nchw_desc,
                                                                          in_nchw,
                                                                          wei_kcyx_desc,
                                                                          wei_kcyx,
                                                                          out_nkhw_desc,
                                                                          out_nkhw_device,
                                                                          ConvStrides{},
                                                                          ConvDilations{},
                                                                          LeftPads{},
                                                                          RightPads{},
                                                                          nrepeat);
    if(do_verification)
    {
#if 0
        if(Y == 3 && X == 3 && ConvStrides{}[0] == 1 && ConvStrides{}[1] == 1 &&
           ConvDilations{}[0] == 1 && ConvDilations{}[1] == 1)
        {
            host_winograd_3x3_convolution(
                in_nchw, wei_kcyx, out_nkhw_host, LeftPads{}, RightPads{});
        }
        else
#endif
        {
            host_direct_convolution(in_nchw,
                                    wei_kcyx,
                                    out_nkhw_host,
                                    ConvStrides{},
                                    ConvDilations{},
                                    LeftPads{},
                                    RightPads{});
        }
        check_error(out_nkhw_host, out_nkhw_device);

#if 0
        LogRange(std::cout << "in_nchw : ", in_nchw.mData, ",") << std::endl;
        LogRange(std::cout << "wei_kcyx: ", wei_kcyx.mData, ",") << std::endl;
        LogRange(std::cout << "out_nkhw_host  : ", out_nkhw_host.mData, ",") << std::endl;
        LogRange(std::cout << "out_nkhw_device: ", out_nkhw_device.mData, ",") << std::endl;
#endif
    }
}
