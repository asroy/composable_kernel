#ifndef CK_DYNAMIC_TENSOR_DESCRIPTOR_HELPER_HPP
#define CK_DYNAMIC_TENSOR_DESCRIPTOR_HELPER_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"

namespace ck {

/*
 * These functions create tensor descriptor at runtime. If they are not constexpr, you will
 * likely see usage of scratch memory during construction of these tensor descriptors. So
 * it's better to call these functions on host and then pass the constructed tensor descritpors
 * to GPU. If the tensor descritpors being constructed are constexpr, then you can call these
 * functions on GPU without worrying about scratch memory usage.
 */

template <index_t N>
__host__ __device__ constexpr auto
make_dynamic_naive_tensor_descriptor(const MultiIndex<N>& lengths, const MultiIndex<N>& strides)
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

template <index_t N>
__host__ __device__ constexpr auto
make_dynamic_naive_tensor_descriptor_packed(const MultiIndex<N>& lengths)
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
make_dynamic_naive_tensor_descriptor_aligned(const MultiIndex<N>& lengths, index_t align)
{
    auto strides = make_zero_multi_index<N>();

    strides(Number<N - 1>{}) = 1;
    strides(Number<N - 2>{}) = math::lcm(lengths[Number<N - 1>{}], align);

    static_for<N - 3, -1, -1>{}([&](auto i) {
        constexpr auto i_p1 = i + Number<1>{};
        strides(i)          = strides(i_p1) * lengths(i_p1);
    });

    return make_dynamic_naive_tensor_descriptor<N>(lengths, strides);
}

} // namespace ck
#endif
