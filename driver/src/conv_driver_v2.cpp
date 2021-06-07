#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
#include "device_dynamic_convolution_forward_implicit_gemm_v4r5_nchw_kcyx_nkhw.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    if(argc != 20)
    {
        printf("arg1: do_verification, arg2: do_log, arg3: init_method, arg4: nrepeat\n");
        printf("arg5 to arg19: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        exit(1);
    }

    const bool do_verification = atoi(argv[1]);
    const int init_method      = atoi(argv[2]);
    const bool do_log          = atoi(argv[3]);
    const int nrepeat          = atoi(argv[4]);

    const index_t N  = atoi(argv[5]);
    const index_t K  = atoi(argv[6]);
    const index_t C  = atoi(argv[7]);
    const index_t Y  = atoi(argv[8]);
    const index_t X  = atoi(argv[9]);
    const index_t Hi = atoi(argv[10]);
    const index_t Wi = atoi(argv[11]);

    const auto conv_strides   = make_tuple(atoi(argv[12]), atoi(argv[13]));
    const auto conv_dilations = make_tuple(atoi(argv[14]), atoi(argv[15]));
    const auto in_left_pads   = make_tuple(atoi(argv[16]), atoi(argv[17]));
    const auto in_right_pads  = make_tuple(atoi(argv[18]), atoi(argv[19]));

    const auto YEff = (Y - I1) * conv_dilations[I0] + I1;
    const auto XEff = (X - I1) * conv_dilations[I1] + I1;

    const auto Ho = (Hi + in_left_pads[I0] + in_right_pads[I0] - YEff) / conv_strides[I0] + I1;
    const auto Wo = (Wi + in_left_pads[I1] + in_right_pads[I1] - XEff) / conv_strides[I1] + I1;

#if 1
    using in_data_t                  = float;
    constexpr index_t in_vector_size = 1;
    using acc_data_t                 = float;
    using out_data_t                 = float;
#elif 0
    using in_data_t                  = float;
    constexpr index_t in_vector_size = 1;
    using acc_data_t                 = float;
    using out_data_t                 = int8_t;
#elif 1
    using in_data_t                  = int8_t;
    constexpr index_t in_vector_size = 16;
    using acc_data_t                 = int32_t;
    using out_data_t                 = int8_t;
#endif

    Tensor<in_data_t> in_nchw(HostTensorDescriptor(std::initializer_list<index_t>{N, C, Hi, Wi}));
    Tensor<in_data_t> wei_kcyx(HostTensorDescriptor(std::initializer_list<index_t>{K, C, Y, X}));
    Tensor<out_data_t> out_nkhw_host(
        HostTensorDescriptor(std::initializer_list<index_t>{N, K, Ho, Wo}));
    Tensor<out_data_t> out_nkhw_device(
        HostTensorDescriptor(std::initializer_list<index_t>{N, K, Ho, Wo}));

    ostream_HostTensorDescriptor(in_nchw.mDesc, std::cout << "in_nchw_desc: ");
    ostream_HostTensorDescriptor(wei_kcyx.mDesc, std::cout << "wei_kcyx_desc: ");
    ostream_HostTensorDescriptor(out_nkhw_host.mDesc, std::cout << "out_nkhw_desc: ");

    print_array("InLeftPads", in_left_pads);
    print_array("InRightPads", in_right_pads);
    print_array("ConvStrides", conv_strides);
    print_array("ConvDilations", conv_dilations);

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(do_verification)
    {
        switch(init_method)
        {
        case 0:
            in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            wei_kcyx.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            break;
        case 1:
            in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            wei_kcyx.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            break;
        case 2:
            in_nchw.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            wei_kcyx.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            break;
        case 3:
            in_nchw.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            wei_kcyx.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            break;
        default:
            in_nchw.GenerateTensorValue(GeneratorTensor_2{1, 5}, num_thread);

            auto gen_wei = [](auto... is) {
                return GeneratorTensor_2{1, 5}(is...) * GeneratorTensor_Checkboard{}(is...);
            };
            wei_kcyx.GenerateTensorValue(gen_wei, num_thread);
        }
    }

#if 1
    device_dynamic_convolution_forward_implicit_gemm_v4r5_nchw_kcyx_nkhw<in_data_t,
                                                                         in_vector_size,
                                                                         acc_data_t,
                                                                         out_data_t>(
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, C, Hi, Wi)),
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, C, Y, X)),
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, K, Ho, Wo)),
        conv_strides,
        conv_dilations,
        in_left_pads,
        in_right_pads,
        in_nchw,
        wei_kcyx,
        out_nkhw_device,
        nrepeat);
#endif

    if(do_verification)
    {
        host_direct_convolution(in_nchw,
                                wei_kcyx,
                                out_nkhw_host,
                                conv_strides,
                                conv_dilations,
                                in_left_pads,
                                in_right_pads);

        check_error(out_nkhw_host, out_nkhw_device);

        if(do_log)
        {
            LogRange(std::cout << "in_nchw : ", in_nchw.mData, ",") << std::endl;
            LogRange(std::cout << "wei_kcyx: ", wei_kcyx.mData, ",") << std::endl;
            LogRange(std::cout << "out_nkhw_host  : ", out_nkhw_host.mData, ",") << std::endl;
            LogRange(std::cout << "out_nkhw_device: ", out_nkhw_device.mData, ",") << std::endl;
        }
    }
}
