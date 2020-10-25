#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
#include "host_col2im.hpp"
#include "device_col2im_eb_nchw.hpp"
#include "device_dynamic_col2im_gemmkgemmn_nchw.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;

#if 1
    // 3x3, 71x71
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t HI = 71;
    constexpr index_t WI = 71;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 1
    // 1x1, 8x8
    constexpr index_t N  = 128;
    constexpr index_t C  = 1536;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 73x73
    constexpr index_t N  = 128;
    constexpr index_t C  = 160;
    constexpr index_t HI = 73;
    constexpr index_t WI = 73;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 35x35
    constexpr index_t N  = 128;
    constexpr index_t C  = 96;
    constexpr index_t HI = 35;
    constexpr index_t WI = 35;
    constexpr index_t K  = 96;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 3x3, 71x71
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t HI = 71;
    constexpr index_t WI = 71;
    constexpr index_t K  = 192;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 7x1, 17x17
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t HI = 17;
    constexpr index_t WI = 17;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 7;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<3, 0>;
    using RightPads = Sequence<3, 0>;
#elif 1
    // 1x7, 17x17
    constexpr index_t N  = 128;
    constexpr index_t C  = 128;
    constexpr index_t HI = 17;
    constexpr index_t WI = 17;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 7;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 3>;
    using RightPads = Sequence<0, 3>;
#elif 0
    // 3x3, 299x299 stride=2
    constexpr index_t N  = 128;
    constexpr index_t C  = 3;
    constexpr index_t HI = 299;
    constexpr index_t WI = 299;
    constexpr index_t K  = 32;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 147x147
    constexpr index_t N  = 128;
    constexpr index_t C  = 32;
    constexpr index_t HI = 147;
    constexpr index_t WI = 147;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 3x3, 149x149
    constexpr index_t N  = 128;
    constexpr index_t C  = 32;
    constexpr index_t HI = 149;
    constexpr index_t WI = 149;
    constexpr index_t K  = 32;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 17x17, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t HI = 17;
    constexpr index_t WI = 17;
    constexpr index_t K  = 192;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 35x35
    constexpr index_t N  = 128;
    constexpr index_t C  = 384;
    constexpr index_t HI = 35;
    constexpr index_t WI = 35;
    constexpr index_t K  = 96;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 35x35, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 256;
    constexpr index_t HI = 35;
    constexpr index_t WI = 35;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x3, 8x8
    constexpr index_t N  = 128;
    constexpr index_t C  = 384;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 448;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 1>;
    using RightPads = Sequence<0, 1>;
#elif 0
    // 3x1, 8x8
    constexpr index_t N  = 128;
    constexpr index_t C  = 448;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 512;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 0>;
    using RightPads = Sequence<1, 0>;
#elif 0
    // 3x3, 147x147
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t HI = 147;
    constexpr index_t WI = 147;
    constexpr index_t K  = 96;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 7x1, 73x73
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t HI = 73;
    constexpr index_t WI = 73;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 7;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<3, 0>;
    using RightPads = Sequence<3, 0>;
#elif 0
    // 3x3, 73x73
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t HI = 73;
    constexpr index_t WI = 73;
    constexpr index_t K  = 96;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 14x14, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 1024;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 2048;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 14x14
    constexpr index_t N  = 128;
    constexpr index_t C  = 1024;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 14x14, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 1024;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 512;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 1
    // 3x3, 28x28
    constexpr index_t N  = 128;
    constexpr index_t C  = 192;
    constexpr index_t HI = 28;
    constexpr index_t WI = 28;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 3x3, 14x14
    constexpr index_t N  = 128;
    constexpr index_t C  = 256;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 1x1, 56x56, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 256;
    constexpr index_t HI = 56;
    constexpr index_t WI = 56;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 7x7, 230x230 stride=2
    constexpr index_t N  = 128;
    constexpr index_t C  = 3;
    constexpr index_t HI = 230;
    constexpr index_t WI = 230;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 7;
    constexpr index_t X  = 7;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 28x28, stride = 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t HI = 28;
    constexpr index_t WI = 28;
    constexpr index_t K  = 1024;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 28x28, stride 2
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t HI = 28;
    constexpr index_t WI = 28;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<2, 2>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1, 7x7
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 2048;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 7x7
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 512;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 1x1, 56x56
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t HI = 56;
    constexpr index_t WI = 56;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3, 56x56
    constexpr index_t N  = 128;
    constexpr index_t C  = 64;
    constexpr index_t HI = 56;
    constexpr index_t WI = 56;
    constexpr index_t K  = 64;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#endif

    constexpr auto img_nchw_desc = make_native_tensor_descriptor_packed(Sequence<N, C, HI, WI>{});
    constexpr auto wei_kcyx_desc = make_native_tensor_descriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto out_nkhw_desc = get_convolution_output_default_4d_tensor_descriptor(
        img_nchw_desc, wei_kcyx_desc, ConvStrides{}, ConvDilations{}, LeftPads{}, RightPads{});

    constexpr index_t HO = out_nkhw_desc.GetLengths()[2];
    constexpr index_t WO = out_nkhw_desc.GetLengths()[3];

    constexpr auto col_eb_desc =
        make_native_tensor_descriptor_packed(Sequence<C * Y * X, N * HO * WO>{});

    using FilterSizes = Sequence<Y, X>;
    using OutputSizes = Sequence<HO, WO>;

    ostream_tensor_descriptor(col_eb_desc, std::cout << "col_eb_desc: ");
    ostream_tensor_descriptor(img_nchw_desc, std::cout << "img_nchw_desc: ");
    print_array("FilterSizes", FilterSizes{});
    print_array("OutputSizes", OutputSizes{});
    print_array("LeftPads", LeftPads{});
    print_array("LeftPads", LeftPads{});
    print_array("RightPads", RightPads{});
    print_array("ConvStrides", ConvStrides{});
    print_array("ConvDilations", ConvDilations{});

    Tensor<float> col_eb(make_HostTensorDescriptor(col_eb_desc));
    Tensor<float> img_nchw_host(make_HostTensorDescriptor(img_nchw_desc));
    Tensor<float> img_nchw_device(make_HostTensorDescriptor(img_nchw_desc));

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(argc != 3)
    {
        printf("arg1: do_verification, arg2: nrepeat\n");
        exit(1);
    }

    bool do_verification = atoi(argv[1]);
    std::size_t nrepeat  = atoi(argv[2]);

    if(do_verification)
    {
#if 0
        col_eb.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
#else
        col_eb.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
#endif
    }

#if 0
    device_col2im_eb_nchw(col_eb_desc,
                          col_eb,
                          img_nchw_desc,
                          img_nchw_device,
                          FilterSizes{},
                          OutputSizes{},
                          ConvStrides{},
                          ConvDilations{},
                          LeftPads{},
                          RightPads{},
                          nrepeat);
#elif 1
    device_dynamic_col2im_gemmkgemmn_nchw(col_eb_desc,
                                          col_eb,
                                          img_nchw_desc,
                                          img_nchw_device,
                                          FilterSizes{},
                                          OutputSizes{},
                                          ConvStrides{},
                                          ConvDilations{},
                                          LeftPads{},
                                          RightPads{},
                                          nrepeat);
#endif

    if(do_verification)
    {
        host_col2im(col_eb,
                    img_nchw_host,
                    FilterSizes{},
                    OutputSizes{},
                    ConvStrides{},
                    ConvDilations{},
                    LeftPads{},
                    RightPads{});

        check_error(img_nchw_host, img_nchw_device);

#if 0
        LogRange(std::cout << "col_eb : ", col_eb.mData, ",") << std::endl;
        LogRange(std::cout << "img_nchw_host : ", img_nchw_host.mData, ",") << std::endl;
        LogRange(std::cout << "img_nchw_device : ", img_nchw_device.mData, ",") << std::endl;
#endif
    }
}
