#ifndef CK_THREADWISE_GENERIC_TENSOR_OP_HPP
#define CK_THREADWISE_GENERIC_TENSOR_OP_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"

namespace ck {
template <class Float, class TensorDesc>
__device__ void threadwise_generic_tensor_set_zero(TensorDesc, Float* __restrict__ p)
{
    ford<decltype(TensorDesc::GetLengths())>{}(
        [&](auto idx) { p[TensorDesc::CalculateOffset(idx)] = static_cast<Float>(0); });
}

} // namespace ck
#endif
