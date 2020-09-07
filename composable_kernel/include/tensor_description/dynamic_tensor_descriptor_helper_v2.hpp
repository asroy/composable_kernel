#ifndef CK_DYNAMIC_TENSOR_DESCRIPTOR_HELPER_V2_HPP
#define CK_DYNAMIC_TENSOR_DESCRIPTOR_HELPER_V2_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor_v2.hpp"

namespace ck {

template <typename LowerTensorDescriptor,
          typename Transforms,
          typename LowerVisibleDimensionLowerVisibleIdss,
          typename UpperVisibleDimensionUpperVisibleIdss>
__host__ __device__ constexpr auto
transform_dynamic_tensor_descriptor_v2(const LowerTensorDescriptor& low_tensor_desc,
                                       const Transforms& transforms,
                                       LowerVisibleDimensionLowerVisibleIdss,
                                       UpperVisibleDimensionUpperVisibleIdss)
{
    // convert lower visible dimension idss (tuple of sequences) to hidden dimension idss (tuple of sequences)
    constexpr auto low_visible_dimension_hidden_idss = transform_tuples(
        // convert lower visible dimension ids (a sequence) to hidden dimension ids (a sequence)
        [](auto low_visible_dim_ids) {
            return transform_sequences(
                // convert lower visible dimension id to hidden dimension id
                [](auto low_visible_dim_id) {
                    return low_tensor_desc.GetVisibleDimensionIds()[low_visible_dim_id];
                },
                low_visible_dim_ids);
        },
        LowerVisibleDimensionLowerVisibleIdss{});

    constexpr auto up_visible_dims_

    const auto all_transforms = merge_tuples(old_tensor_desc.GetTransforms(), new_transforms);
    constexpr auto all_low_dim_idss =
        merge_tuples(old_tensor_desc.GetLowerDimensionIdss(), new_low_dim_idss);
    constexpr auto all_up_dim_idss =
        merge_tuples(old_tensor_desc.GetUpperDimensionIdss(), new_up_dim_idss);

    constexpr auto new_visible_dim_ids = new_up_dim_idss

        return DynamicTensorDescriptor_v2<decltype(all_transforms),
                                          decltype(all_low_dim_idss),
                                          decltype(all_up_dim_idss),
                                          decltype(new_visible_dim_ids)>{
            all_transforms, old_tensor_desc.GetElementSpaceSize()};
}

} // namespace ck
#endif
