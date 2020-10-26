#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "tensor_descriptor_helper.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_dynamic_copy_gemmkgemmn.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;

#if 1
    constexpr index_t GemmK = 8;
    constexpr index_t GemmN = 128;
#endif

    constexpr auto src_gemmk_gemmn_desc =
        make_native_tensor_descriptor_packed(Sequence<GemmK, GemmN>{});
    constexpr auto dst_gemmk_gemmn_desc =
        make_native_tensor_descriptor_packed(Sequence<GemmK, GemmN>{});
    ostream_tensor_descriptor(src_gemmk_gemmn_desc, std::cout << "src_gemmk_gemmn_desc: ");
    ostream_tensor_descriptor(dst_gemmk_gemmn_desc, std::cout << "dst_gemmk_gemmn_desc: ");

    Tensor<float> src_gemmk_gemmn(make_HostTensorDescriptor(src_gemmk_gemmn_desc));
    Tensor<float> dst_gemmk_gemmn_device(make_HostTensorDescriptor(dst_gemmk_gemmn_desc));

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
        src_gemmk_gemmn.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
#else
        src_gemmk_gemmn.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
#endif
    }

    device_dynamic_copy(
        src_gemmk_gemmn_desc, src_gemmk_gemmn, dst_gemmk_gemmn_desc, dst_gemmk_gemmn_device);

    if(do_verification)
    {
        check_error(src_gemmk_gemmn, dst_gemmk_gemmn_device);

#if 0
        LogRange(std::cout << "src_gemmk_gemmn : ", src_gemmk_gemmn.mData, ",") << std::endl;
        LogRange(std::cout << "dst_gemmk_gemmn_device : ", dst_gemmk_gemmn_device.mData, ",") << std::endl;
#endif
    }
}
