#ifndef CK_ARRAY_HELPER_HPP
#define CK_ARRAY_HELPER_HPP

#include "sequence.hpp"
#include "sequence_helper.hpp"
#include "array.hpp"
#include "array_helper.hpp"
#include "tuple.hpp"
#include "tuple_helper.hpp"
#include "statically_indexed_array.hpp"
#include "array_element_picker.hpp"

namespace ck {

template <typename TData, index_t NSize>
__host__ __device__ constexpr auto container_push_back(const Array<TData, NSize>& a, const TData& x)
{
    Array<TData, NSize + 1> r;

    static_for<0, NSize, 1>{}([&r, &a ](auto i) constexpr { r(i) = a[i]; });

    r(Number<NSize>{}) = x;

    return r;
}

template <typename TData, index_t NSize, index_t... IRs>
__host__ __device__ constexpr auto
container_reorder_given_new2old(const Array<TData, NSize>& old_array, Sequence<IRs...> /*new2old*/)
{
    static_assert(NSize == sizeof...(IRs), "wrong! size not consistent");

    static_assert(is_valid_sequence_map<Sequence<IRs...>>{}, "wrong! invalid reorder map");

    return make_array(old_array[Number<IRs>{}]...);
}

template <typename TData, index_t NSize, index_t... IRs>
__host__ __device__ constexpr auto
container_reorder_given_old2new(const Array<TData, NSize>& old_array, Sequence<IRs...> old2new)
{
    return container_reorder_given_new2old(
        old_array, typename sequence_map_inverse<decltype(old2new)>::type{});
}

template <typename... Ts, index_t... IRs>
__host__ __device__ constexpr auto container_reorder_given_new2old(const Tuple<Ts...>& old_tuple,
                                                                   Sequence<IRs...> /*new2old*/)
{
    static_assert(sizeof...(Ts) == sizeof...(IRs), "wrong! size not consistent");

    static_assert(is_valid_sequence_map<Sequence<IRs...>>{}, "wrong! invalid reorder map");

    return make_tuple(old_tuple[Number<IRs>{}]...);
}

template <typename... Ts, index_t... IRs>
__host__ __device__ constexpr auto container_reorder_given_old2new(const Tuple<Ts...>& old_tuple,
                                                                   Sequence<IRs...> old2new)
{
    return container_reorder_given_new2old(
        old_tuple, typename sequence_map_inverse<decltype(old2new)>::type{});
}

template <typename TData, typename Container, typename Reduce>
__host__ __device__ constexpr TData container_reduce(const Container& a, Reduce f, TData init)
{
    // static_assert(is_same<typename Arr::data_type, TData>::value, "wrong! different data type");
    static_assert(Container::Size() > 0, "wrong");

    TData result = init;

    static_for<0, Container::Size(), 1>{}([&](auto I) { result = f(result, a[I]); });

    return result;
}

template <typename TData, index_t NSize, typename Reduce>
__host__ __device__ constexpr auto
container_reverse_inclusive_scan(const Array<TData, NSize>& x, Reduce f, TData init)
{
    Array<TData, NSize> y;

    TData r = init;

    static_for<NSize - 1, 0, -1>{}([&](auto i) {
        r    = f(r, x[i]);
        y(i) = r;
    });

    r              = f(r, x[Number<0>{}]);
    y(Number<0>{}) = r;

    return y;
}

template <typename TData, index_t NSize, typename Reduce>
__host__ __device__ constexpr auto
container_reverse_exclusive_scan(const Array<TData, NSize>& x, Reduce f, TData init)
{
    Array<TData, NSize> y;

    TData r = init;

    static_for<NSize - 1, 0, -1>{}([&](auto i) {
        y(i) = r;
        r    = f(r, x[i]);
    });

    y(Number<0>{}) = r;

    return y;
}

template <typename TData, index_t NSize, typename Reduce>
__host__ __device__ constexpr auto container_reverse_exclusive_scan(
    const StaticallyIndexedArray<TData, NSize>& x, Reduce f, TData init)
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
__host__ __device__ constexpr auto container_reverse_inclusive_scan(
    const StaticallyIndexedArray<TData, NSize>& x, Reduce f, TData init)
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

template <typename X, typename... Ys>
__host__ __device__ constexpr auto container_cat(const X& x, const Ys&... ys)
{
    return container_cat(x, container_cat(ys...));
}

template <typename T, index_t NX, index_t NY>
__host__ __device__ constexpr auto container_cat(const Array<T, NX>& ax, const Array<T, NY>& ay)
{
    return unpack2(
        [&](auto&&... zs) { return make_array(std::forward<decltype(zs)>(zs)...); }, ax, ay);
}

template <typename... X, typename... Y>
__host__ __device__ constexpr auto container_cat(const Tuple<X...>& tx, const Tuple<Y...>& ty)
{
    return unpack2(
        [&](auto&&... zs) { return make_tuple(std::forward<decltype(zs)>(zs)...); }, tx, ty);
}

template <typename Container>
__host__ __device__ constexpr auto container_cat(const Container& x)
{
    return x;
}

} // namespace ck
#endif
