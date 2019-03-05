#pragma once
#include "ConstantTensorDescriptor.hip.hpp"

template <class Float, class Desc, class F>
__device__ void threadwise_4d_tensor_pointwise_operation_unary(Desc, Float* __restrict__ p, F f)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto desc = Desc{};

#if 0
    if(threadIdx.x == 0)
    {
        print_ConstantTensorDescriptor(desc, "threadwise_4d_tensor_op_unary: ");
    }
#endif

    for(unsigned did0 = 0; did0 < desc.GetLength(I0); ++did0)
    {
        for(unsigned did1 = 0; did1 < desc.GetLength(I1); ++did1)
        {
            for(unsigned did2 = 0; did2 < desc.GetLength(I2); ++did2)
            {
                for(unsigned did3 = 0; did3 < desc.GetLength(I3); ++did3)
                {
                    const unsigned dindex = desc.Get1dIndex(did0, did1, did2, did3);

                    f(p[dindex]);
                }
            }
        }
    }
}

template <class Float, class Desc>
__device__ void threadwise_4d_tensor_set_zero(Desc, Float* __restrict__ p)
{
    auto f_set_zero = [](Float& v) { v = Float(0); };

    threadwise_4d_tensor_pointwise_operation_unary<Float, Desc, decltype(f_set_zero)>(
        Desc{}, p, f_set_zero);
}
