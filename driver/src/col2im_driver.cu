#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "config.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "print_array.hpp"
#include "print_sequence.hpp"
#include "device.hpp"
#include "tensor_generator.hpp"
#include "device_tensor.hpp"
#include "conv_common.hpp"
#include "host_col2im.hpp"
//#include "device_col2im.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;

#if 1
    constexpr index_t N  = 1;
    constexpr index_t C  = 1;
    constexpr index_t HI = 17;
    constexpr index_t WI = 17;
    constexpr index_t K  = 1;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<1, 1>;
    using RightPads = Sequence<1, 1>;
#elif 0
    // 3x3, 34x34
    constexpr index_t N  = 64;
    constexpr index_t C  = 256;
    constexpr index_t HI = 34;
    constexpr index_t WI = 34;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 3;
    constexpr index_t X  = 3;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 8x8 image
    // cudnn@V100 68%, ck@V100 72%, ck@P100 52%, ck@VII 42%
    constexpr index_t N  = 64;
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
    // 1x1 filter, 8x8 image
    // cudnn@V100 77%, ck@V100 76%, ck@P100 79%, ck@VII 51%
    constexpr index_t N  = 128;
    constexpr index_t C  = 2048;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 7x7 image
    // cudnn@V100 82%, ck@V100 76%, ck@P100 67%, ck@VII 64%
    constexpr index_t N  = 128;
    constexpr index_t C  = 832;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 8x8 image
    // cudnn@V100 83%, ck@V100 75%, ck@P100 78%, ck@VII 65%
    constexpr index_t N  = 128;
    constexpr index_t C  = 1280;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 14x14 image
    // cudnn@V100 62%, ck@V100 68%, ck@P100 70%, ck@VII 50%
    constexpr index_t N  = 128;
    constexpr index_t C  = 512;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 8x8 image
    // cudnn@V100 74%, ck@V100 57%, ck@P100 78%, ck@VII 61%
    constexpr index_t N  = 64;
    constexpr index_t C  = 1536;
    constexpr index_t HI = 8;
    constexpr index_t WI = 8;
    constexpr index_t K  = 384;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 28x28 image
    // cudnn@V100 86%, ck@V100 84%, ck@P100 80%, ck@VII 69%
    constexpr index_t N  = 128;
    constexpr index_t C  = 256;
    constexpr index_t HI = 28;
    constexpr index_t WI = 28;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 7x7 image
    // cudnn@V100 71%, ck@V100 55%, ck@P100 70%, ck@VII 62%
    constexpr index_t N  = 128;
    constexpr index_t C  = 832;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 256;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 17x17 input
    // cudnn@V100 81%, ck@V100 76%, ck@P100 70%, ck@VII 76%
    constexpr index_t N  = 128;
    constexpr index_t C  = 768;
    constexpr index_t HI = 17;
    constexpr index_t WI = 17;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 14x14 image
    // cudnn@V100 73%, ck@V100 71%, ck@P100 70%, ck@VII 64%
    constexpr index_t N  = 128;
    constexpr index_t C  = 528;
    constexpr index_t HI = 14;
    constexpr index_t WI = 14;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 1x1 filter, 14x14 image
    // cudnn@V100 73%, ck@V100 72%, ck@P100 79%, ck@VII 75%
    constexpr index_t N  = 128;
    constexpr index_t C  = 528;
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
    // 1x1 filter, 7x7 image
    // cudnn@V100 49%, ck@V100 50%, ck@P100 61%, ck@VII 52%
    constexpr index_t N  = 128;
    constexpr index_t C  = 832;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 1;
    constexpr index_t X  = 1;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<0, 0>;
    using RightPads = Sequence<0, 0>;
#elif 0
    // 3x3 filter, 2x2 stride, 35x35 input, 17x17 output
    // cudnn@V100 90%, ck@V100 93%, ck@P100 83%, ck@VII 81%
    constexpr index_t N  = 128;
    constexpr index_t C  = 288;
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
    // 5x5 filter, 2x2 pad, 7x7 input
    constexpr index_t N  = 128;
    constexpr index_t C  = 48;
    constexpr index_t HI = 7;
    constexpr index_t WI = 7;
    constexpr index_t K  = 128;
    constexpr index_t Y  = 5;
    constexpr index_t X  = 5;

    using ConvStrides   = Sequence<1, 1>;
    using ConvDilations = Sequence<1, 1>;

    using LeftPads  = Sequence<2, 2>;
    using RightPads = Sequence<2, 2>;
#elif 0
    // 7x1 filter, 3x0 pad, 17x17 input
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
    // 1x7 filter, 0x3 pad, 17x17 input
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
#endif

    constexpr auto in_nchw_desc  = make_native_tensor_descriptor_packed(Sequence<N, C, HI, WI>{});
    constexpr auto wei_kcyx_desc = make_native_tensor_descriptor_packed(Sequence<K, C, Y, X>{});
    constexpr auto out_nkhw_desc = get_convolution_output_default_4d_tensor_descriptor(
        in_nchw_desc, wei_kcyx_desc, ConvStrides{}, ConvDilations{}, LeftPads{}, RightPads{});

    constexpr index_t HO = out_nkhw_desc.GetLengths()[2];
    constexpr index_t WO = out_nkhw_desc.GetLengths()[3];

    auto in_eb_desc = make_native_tensor_descriptor_packed(Sequence<C * Y * X, N * HO * WO>{});

    using FilterSizes = Sequence<Y, X>;
    using OutputSizes = Sequence<HO, WO>;

    ostream_ConstantTensorDescriptor(in_nchw_desc, std::cout << "in_nchw_desc: ");
    ostream_ConstantTensorDescriptor(in_eb_desc, std::cout << "in_eb_desc: ");
    print_sequence("FilterSizes", FilterSizes{});
    print_sequence("OutputSizes", OutputSizes{});
    print_sequence("LeftPads", LeftPads{});
    print_sequence("LeftPads", LeftPads{});
    print_sequence("RightPads", RightPads{});
    print_sequence("ConvStrides", ConvStrides{});
    print_sequence("ConvDilations", ConvDilations{});

    Tensor<float> in_eb(make_TensorDescriptor(in_eb_desc));
    Tensor<float> in_nchw_host(make_TensorDescriptor(in_nchw_desc));
    Tensor<float> in_nchw_device(make_TensorDescriptor(in_nchw_desc));

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
#if 1
        in_eb.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
#else
        in_eb.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
#endif
    }

#if 0
    device_col2im(in_eb_desc,
                  in_eb,
                  in_nchw_desc,
                  in_nchw_device,
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
        host_col2im(in_eb,
                    in_nchw_host,
                    FilterSizes{},
                    OutputSizes{},
                    ConvStrides{},
                    ConvDilations{},
                    LeftPads{},
                    RightPads{});

        check_error(in_nchw_host, in_nchw_device);

#if 1
        LogRange(std::cout << "in_eb : ", in_eb.mData, ",") << std::endl;
        LogRange(std::cout << "in_nchw_host : ", in_nchw_host.mData, ",") << std::endl;
        LogRange(std::cout << "in_nchw_device : ", in_nchw_device.mData, ",") << std::endl;
#endif
    }
}
