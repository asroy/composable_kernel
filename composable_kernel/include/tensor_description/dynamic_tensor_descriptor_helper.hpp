#ifndef CK_DYNAMIC_TENSOR_DESCRIPTOR_HELPER_HPP
#define CK_DYNAMIC_TENSOR_DESCRIPTOR_HELPER_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"

namespace ck {

template <typename Lengths, typename Strides>
__host__ __device__ constexpr auto make_dynamic_native_tensor_descriptor(const Lengths& lengths,
                                                                         const Strides& strides)
{
    static_assert(Lengths::GetSize() == Strides::GetSize(), "wrong! Size not the same");

    return DynamicNativeTensorDescriptor<Lengths::GetSize()>(lengths, strides);
}

template <typename LowTensorDescriptor,
          typename Transforms,
          typename LowDimensions,
          typename UpDimensions>
__host__ __device__ constexpr auto
transform_dynamic_tensor_descriptor(const LowTensorDescriptor& low_tensor_desc,
                                    const Transforms& transforms,
                                    LowDimensions,
                                    UpDimensions)
{
    return DynamicTransformedTensorDescriptor<LowTensorDescriptor,
                                              Transforms,
                                              LowDimensions,
                                              UpDimensions>(low_tensor_desc, transforms);
}

} // namespace ck
#endif
