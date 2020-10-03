#ifndef CK_DYNAMIC_TENSOR_DESCRIPTOR_HELPER_HPP
#define CK_DYNAMIC_TENSOR_DESCRIPTOR_HELPER_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"

namespace ck {

template <index_t N>
__host__ __device__ constexpr auto
make_dynamic_native_tensor_descriptor_packed(const MultiIndex<N>& lengths)
{

    const auto transforms              = make_tuple(DynamicUnMerge<N>{lengths});
    constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});
    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});
    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

    const index_t element_space_size =
        container_reduce(lengths, math::multiplies<index_t>{}, index_t{1});

    return DynamicTensorDescriptor<decltype(transforms),
                                   decltype(low_dim_hidden_idss),
                                   decltype(up_dim_hidden_idss),
                                   decltype(visible_dim_hidden_ids)>{transforms,
                                                                     element_space_size};
}

template <index_t N>
__host__ __device__ constexpr auto
make_dynamic_native_tensor_descriptor(const MultiIndex<N>& lengths, const MultiIndex<N>& strides)
{
    const auto transforms              = make_tuple(DynamicEmbed<N>{lengths, strides});
    constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});
    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});
    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

    index_t element_space_size = 1;

    static_for<0, N, 1>{}([&](auto i) { element_space_size += (lengths[i] - 1) * strides[i]; });

    return DynamicTensorDescriptor<decltype(transforms),
                                   decltype(low_dim_hidden_idss),
                                   decltype(up_dim_hidden_idss),
                                   decltype(visible_dim_hidden_ids)>{transforms,
                                                                     element_space_size};
}

} // namespace ck
#endif
