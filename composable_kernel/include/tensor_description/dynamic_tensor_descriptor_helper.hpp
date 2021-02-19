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

template <typename... Lengths,
          typename... Strides,
          typename std::enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
__host__ __device__ constexpr auto
make_dynamic_naive_tensor_descriptor_v2(const Tuple<Lengths...>& lengths,
                                        const Tuple<Strides...>& strides)
{
    constexpr index_t N = sizeof...(Lengths);

    const auto transforms =
        make_tuple(DynamicEmbed<N, Tuple<Lengths...>, Tuple<Strides...>>{lengths, strides});

    constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});

    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});

    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

    // recursive function for reduction
    auto f = [&](auto fs, auto i, auto acc_old) {
        auto acc_new = acc_old + (lengths[i] - Number<1>{}) * strides[i];

        if constexpr(i.value < N - 1)
        {
            return fs(fs, i + Number<1>{}, acc_new);
        }
        else
        {
            return acc_new;
        }
    };

    const auto element_space_size = f(f, Number<0>{}, Number<1>{});

    return DynamicTensorDescriptor<remove_cv_t<decltype(transforms)>,
                                   remove_cv_t<decltype(low_dim_hidden_idss)>,
                                   remove_cv_t<decltype(up_dim_hidden_idss)>,
                                   remove_cv_t<decltype(visible_dim_hidden_ids)>,
                                   remove_cv_t<decltype(element_space_size)>>{transforms,
                                                                              element_space_size};
}

// Lengths... can be:
//   1) index_t, which is known at run-time
//   2) Number<>, which is known at compile-time
template <typename... Lengths>
__host__ __device__ constexpr auto
make_dynamic_naive_tensor_descriptor_packed_v2(const Tuple<Lengths...>& lengths)
{
    constexpr index_t N = sizeof...(Lengths);

    const auto transforms = make_tuple(DynamicUnMerge<N, false, Tuple<Lengths...>>{lengths});

    constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});

    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});

    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

    const auto element_space_size = container_reduce(lengths, math::multiplies_v2{}, Number<1>{});

    return DynamicTensorDescriptor<remove_cv_t<decltype(transforms)>,
                                   remove_cv_t<decltype(low_dim_hidden_idss)>,
                                   remove_cv_t<decltype(up_dim_hidden_idss)>,
                                   remove_cv_t<decltype(visible_dim_hidden_ids)>,
                                   remove_cv_t<decltype(element_space_size)>>{transforms,
                                                                              element_space_size};
}

template <typename... Lengths, typename Align>
__host__ __device__ constexpr auto
make_dynamic_naive_tensor_descriptor_aligned_v2(const Tuple<Lengths...>& lengths, Align align)
{
    constexpr index_t N = sizeof...(Lengths);

    auto strides = generate_tuple(
        [&](auto i) {
            if constexpr(i.value == N - 1)
            {
                return Number<1>{};
            }
            else if constexpr(i.value == N - 2)
            {
                return math::lcm(lengths[Number<N - 1>{}], align);
            }
            else
            {
                return container_reduce(lengths,
                                        math::multiplies_v2{},
                                        math::lcm(lengths[Number<N - 1>{}], align),
                                        i,
                                        Number<N - 2>{},
                                        Number<1>{});
            }
        },
        Number<N>{});

    return make_dynamic_naive_tensor_descriptor_v2(lengths, strides);
}

} // namespace ck
#endif
