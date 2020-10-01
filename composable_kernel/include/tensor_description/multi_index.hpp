#ifndef CK_MULTI_INDEX_HPP
#define CK_MULTI_INDEX_HPP

#include "common_header.hpp"

namespace ck {

#if CK_USE_DYNAMICALLY_INDEXED_MULTI_INDEX
template <index_t N>
using MultiIndex = Array<index_t, N>;

template <typename... Xs>
__host__ __device__ constexpr auto make_multi_index(Xs&&... xs)
{
    return make_array<index_t>(index_t{xs}...);
}

template <index_t NSize>
__host__ __device__ constexpr auto make_zero_multi_index()
{
    return unpack([](auto... xs) { return make_multi_index(xs...); },
                  typename uniform_sequence_gen<NSize, 0>::type{});
}

template <typename T>
__host__ __device__ constexpr auto to_multi_index(const T& x)
{
    return unpack([](auto... ys) { return make_multi_index(ys...); }, x);
}

template <index_t NSize, typename X>
__host__ __device__ constexpr auto operator+=(MultiIndex<NSize>& y, const X& x)
{
    static_assert(X::Size() == NSize, "wrong! size not the same");
    static_for<0, NSize, 1>{}([&](auto i) { y(i) += x[i]; });
    return y;
}

template <index_t NSize, typename X>
__host__ __device__ constexpr auto operator-=(MultiIndex<NSize>& y, const X& x)
{
    static_assert(X::Size() == NSize, "wrong! size not the same");
    static_for<0, NSize, 1>{}([&](auto i) { y(i) -= x[i]; });
    return y;
}

template <index_t NSize, typename T>
__host__ __device__ constexpr auto operator+(const MultiIndex<NSize>& a, const T& b)
{
    using type = MultiIndex<NSize>;
    static_assert(T::Size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a[i] + b[i]; });
    return r;
}

template <index_t NSize, typename T>
__host__ __device__ constexpr auto operator-(const MultiIndex<NSize>& a, const T& b)
{
    using type = MultiIndex<NSize>;
    static_assert(T::Size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a[i] - b[i]; });
    return r;
}

template <index_t NSize, typename T>
__host__ __device__ constexpr auto operator*(const MultiIndex<NSize>& a, const T& b)
{
    using type = MultiIndex<NSize>;
    static_assert(T::Size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a[i] * b[i]; });
    return r;
}

#else
template <index_t N>
using MultiIndex = StaticallyIndexedArray<index_t, N>;

template <typename... Xs>
__host__ __device__ constexpr auto make_multi_index(Xs&&... xs)
{
    return make_statically_indexed_array<index_t>(index_t{xs}...);
}

template <index_t NSize>
__host__ __device__ constexpr auto make_zero_multi_index()
{
    return unpack([](auto... xs) { return make_multi_index(xs...); },
                  typename uniform_sequence_gen<NSize, 0>::type{});
}

template <typename T>
__host__ __device__ constexpr auto to_multi_index(const T& x)
{
    return unpack([](auto... ys) { return make_multi_index(ys...); }, x);
}

// Here should use MultiIndex<NSize>, instead of Tuple<Ys...>, although the former
// is the alias of the latter. This is because compiler cannot infer the NSize if
// using MultiIndex<NSize>
// TODO: how to fix this?
template <typename... Ys, typename X>
__host__ __device__ constexpr auto operator+=(Tuple<Ys...>& y, const X& x)
{
    static_assert(X::Size() == sizeof...(Ys), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Ys);
    static_for<0, NSize, 1>{}([&](auto i) { y(i) += x[i]; });
    return y;
}

template <typename... Ys, typename X>
__host__ __device__ constexpr auto operator-=(Tuple<Ys...>& y, const X& x)
{
    static_assert(X::Size() == sizeof...(Ys), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Ys);
    static_for<0, NSize, 1>{}([&](auto i) { y(i) -= x[i]; });
    return y;
}

template <typename... Xs, typename Y>
__host__ __device__ constexpr auto operator+(const Tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = x[i] + y[i]; });
    return r;
}

template <typename... Xs, typename Y>
__host__ __device__ constexpr auto operator-(const Tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = x[i] - y[i]; });
    return r;
}

template <typename... Xs, typename Y>
__host__ __device__ constexpr auto operator*(const Tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::Size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = x[i] * y[i]; });
    return r;
}

#endif
} // namespace ck
#endif
