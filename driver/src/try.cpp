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
#include "device_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw.hpp"
#include "device_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"

int main(int argc, char* argv[])
{
    using namespace ck;

    auto idx1 = std::array<index_t, 2>{{1, 0}};
    auto idx2 = Array<index_t, 2>{{1, 0}};
    auto idx3 = MultiIndex<2>{{1, 0}};

    auto idx0 = MultiIndex<2>{{1, 0}};

    print_array("idx2", idx2);
    print_array("idx3", idx2);
}
