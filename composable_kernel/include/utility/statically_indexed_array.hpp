#ifndef CK_STATICALLY_INDEXED_ARRAY_HPP
#define CK_STATICALLY_INDEXED_ARRAY_HPP

#include "functional2.hpp"
#include "sequence.hpp"
#include "tuple.hpp"

namespace ck {

namespace detail {

template <typename T, index_t NSize>
__host__ __device__ constexpr auto generate_same_type_tuple()
{
    return generate_tuple([](auto) -> T { return T{}; }, Number<NSize>{});
}

template <typename T, index_t NSize>
using same_type_tuple = decltype(generate_same_type_tuple<T, NSize>());

} // namespace detail

template <typename T, index_t NSize>
using StaticallyIndexedArray = detail::same_type_tuple<T, NSize>;

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_statically_indexed_array(const X& x, const Xs&... xs)
{
    return StaticallyIndexedArray<X, sizeof...(Xs) + 1>(x, static_cast<X>(xs)...);
}

// make empty StaticallyIndexedArray
template <typename X>
__host__ __device__ constexpr auto make_statically_indexed_array()
{
    return StaticallyIndexedArray<X, 0>();
}

template <typename TData, index_t NSize, typename Reduce>
__host__ __device__ constexpr auto
reverse_exclusive_scan_on_array(const StaticallyIndexedArray<TData, NSize>& x, Reduce f, TData init)
{
    StaticallyIndexedArray<TData, NSize> y;

    TData r = init;

    static_for<NSize - 1, 0, -1>{}([&](auto i) {
        y(i) = r;
        r    = f(r, x[i]);
    });

    y(Number<0>{}) = r;

    return y;
}

template <typename TData, index_t NSize, typename Reduce>
__host__ __device__ constexpr auto
reverse_inclusive_scan_on_array(const StaticallyIndexedArray<TData, NSize>& x, Reduce f, TData init)
{
    StaticallyIndexedArray<TData, NSize> y;

    TData r = init;

    static_for<NSize - 1, 0, -1>{}([&](auto i) {
        r    = f(r, x[i]);
        y(i) = r;
    });

    r              = f(r, x[Number<0>{}]);
    y(Number<0>{}) = r;

    return y;
}

} // namespace ck
#endif
