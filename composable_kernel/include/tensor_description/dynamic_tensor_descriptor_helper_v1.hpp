#ifndef CK_DYNAMIC_TENSOR_DESCRIPTOR_HELPER_V1_HPP
#define CK_DYNAMIC_TENSOR_DESCRIPTOR_HELPER_V1_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor_v1.hpp"

namespace ck {

template <typename Lengths, typename Strides>
__host__ __device__ constexpr auto make_dynamic_native_tensor_descriptor_v1(const Lengths& lengths,
                                                                            const Strides& strides)
{
    static_assert(Lengths::Size() == Strides::Size(), "wrong! Size not the same");

    return DynamicNativeTensorDescriptor_v1<Lengths::Size()>(lengths, strides);
}

template <typename LowTensorDescriptor,
          typename Transforms,
          typename LowDimensionIds,
          typename UpDimensionIds>
__host__ __device__ constexpr auto
transform_dynamic_tensor_descriptor_v1(const LowTensorDescriptor& low_tensor_desc,
                                       const Transforms& transforms,
                                       LowDimensionIds,
                                       UpDimensionIds)
{
    return DynamicTransformedTensorDescriptor_v1<LowTensorDescriptor,
                                                 Transforms,
                                                 LowDimensionIds,
                                                 UpDimensionIds>{low_tensor_desc, transforms};
}

} // namespace ck
#endif
