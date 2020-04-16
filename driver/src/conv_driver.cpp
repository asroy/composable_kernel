#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "config.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "print_array.hpp"
#include "print_sequence.hpp"
#include "device.hpp"
#include "tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
//#include "device_convolution_direct_v2_nchw_kcyx_nkhw.hpp"
//#include "device_convolution_implicit_gemm_v1_chwn_cyxk_khwn.hpp"
//#include "device_convolution_implicit_gemm_v1_chwn_cyxk_khwn_padded.hpp"
//#include "device_convolution_implicit_gemm_v1_nchw_cyxk_nkhw.hpp"
//#include "device_convolution_implicit_gemm_v2_chwn_cyxk_khwn.hpp"
//#include "device_convolution_implicit_gemm_v3_nchw_cyxk_nkhw.hpp"
#include "device_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw.hpp"
#include "device_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"
#include "device_convolution_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;

    // 1x1, 14x14
    constexpr index_t N  = 64;
    constexpr index_t C  =  1024;
    constexpr index_t HI =  14;
    constexpr index_t WI =  14;
    constexpr index_t K  =  1024;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;

    auto in_nchw_desc  = make_ConstantTensorDescriptor_packed(Sequence<N, C, HI, WI>{});
    auto wei_kcyx_desc = make_ConstantTensorDescriptor_packed(Sequence<K, C, Y, X>{});
    auto out_nkhw_desc = get_convolution_output_default_4d_tensor_descriptor_deprecated(
        in_nchw_desc, wei_kcyx_desc, ConvStrides{}, ConvDilations{}, LeftPads{}, RightPads{});

    ostream_ConstantTensorDescriptor(in_nchw_desc, std::cout << "in_nchw_desc: ");
    ostream_ConstantTensorDescriptor(wei_kcyx_desc, std::cout << "wei_kcyx_desc: ");
    ostream_ConstantTensorDescriptor(out_nkhw_desc, std::cout << "out_nkhw_desc: ");
    print_sequence("LeftPads", LeftPads{});
    print_sequence("RightPads", RightPads{});
    print_sequence("ConvStrides", ConvStrides{});
    print_sequence("ConvDilations", ConvDilations{});

    using in_data_t  = float;
    using out_data_t = float;
    Tensor<in_data_t> in_nchw(make_TensorDescriptor(in_nchw_desc));
    Tensor<in_data_t> wei_kcyx(make_TensorDescriptor(wei_kcyx_desc));
    Tensor<out_data_t> out_nkhw_host(make_TensorDescriptor(out_nkhw_desc));
    Tensor<out_data_t> out_nkhw_device(make_TensorDescriptor(out_nkhw_desc));

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

#if 0
    device_convolution_direct_v2_nchw_kcyx_nkhw
        (in_nchw_desc, in_nchw, wei_kcyx_desc, wei_kcyx, out_nkhw_desc, out_nkhw_device, nrepeat);
#elif 0
    device_convolution_implicit_gemm_v1_chwn_cyxk_khwn(
        in_nchw_desc, in_nchw, wei_kcyx_desc, wei_kcyx, out_nkhw_desc, out_nkhw_device, nrepeat);
#elif 0
    device_convolution_implicit_gemm_v1_chwn_cyxk_khwn_padded(in_nchw_desc,
                                                              in_nchw,
                                                              wei_kcyx_desc,
                                                              wei_kcyx,
                                                              out_nkhw_desc,
                                                              out_nkhw_device,
                                                              LeftPads{},
                                                              RightPads{},
                                                              nrepeat);
#elif 0
    device_convolution_implicit_gemm_v1_nchw_cyxk_nkhw(
        in_nchw_desc, in_nchw, wei_kcyx_desc, wei_kcyx, out_nkhw_desc, out_nkhw_device, nrepeat);
#elif 0
    device_convolution_implicit_gemm_v2_chwn_cyxk_khwn(
        in_nchw_desc, in_nchw, wei_kcyx_desc, wei_kcyx, out_nkhw_desc, out_nkhw_device, nrepeat);
#elif 0
    device_convolution_implicit_gemm_v3_nchw_cyxk_nkhw(
        (in_nchw_desc, in_nchw, wei_kcyx_desc, wei_kcyx, out_nkhw_desc, out_nkhw_device, nrepeat);
#elif 0
    device_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw(in_nchw_desc,
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
#elif 1
    device_convolution_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw(in_nchw_desc,
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
#endif

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
