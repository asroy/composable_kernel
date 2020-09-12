#ifndef CK_ARRAY_HELPER_HPP
#define CK_ARRAY_HELPER_HPP

#include "array.hpp"

namespace ck {

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_array(const X& x, const Xs&... xs)
{
    return Array<X, sizeof...(xs) + 1>{{x, xs...}};
}

template <typename Arr, typename Picks>
__host__ __device__ constexpr auto pick_array_element(Arr& a, Picks)
{
    return ArrayElementPicker<Arr, Picks>(a);
}

template <typename T>
__host__ __device__ constexpr auto to_array(const T& x)
{
    Array<typename T::data_type, T::Size()> y;

    static_for<0, T::Size(), 1>{}([&](auto i) { y.At(i) = x.At(i); });

    return y;
}

template <typename TData, index_t NSize>
__host__ __device__ constexpr auto make_zero_array()
{
    constexpr auto zero_sequence = typename uniform_sequence_gen<NSize, 0>::type{};
    constexpr auto zero_array    = to_array(zero_sequence);
    return zero_array;
}

template <typename TData, index_t NSize, index_t... IRs>
__host__ __device__ constexpr auto reorder_array_given_new2old(const Array<TData, NSize>& old_array,
                                                               Sequence<IRs...> /*new2old*/)
{
    static_assert(NSize == sizeof...(IRs), "NSize not consistent");

    static_assert(is_valid_sequence_map<Sequence<IRs...>>{}, "wrong! invalid reorder map");

    return Array<TData, NSize>{old_array[IRs]...};
}

template <typename TData, index_t NSize, typename MapOld2New>
struct lambda_reorder_array_given_old2new
{
    const Array<TData, NSize>& old_array;
    Array<TData, NSize>& new_array;

    __host__ __device__ constexpr lambda_reorder_array_given_old2new(
        const Array<TData, NSize>& old_array_, Array<TData, NSize>& new_array_)
        : old_array(old_array_), new_array(new_array_)
    {
    }

    template <index_t IOldDim>
    __host__ __device__ constexpr void operator()(Number<IOldDim>) const
    {
        TData old_data = old_array[IOldDim];

        constexpr index_t INewDim = MapOld2New::At(Number<IOldDim>{});

        new_array(Number<INewDim>{}) = old_data;
    }
};

template <typename TData, index_t NSize, index_t... IRs>
__host__ __device__ constexpr auto reorder_array_given_old2new(const Array<TData, NSize>& old_array,
                                                               Sequence<IRs...> /*old2new*/)
{
    Array<TData, NSize> new_array;

    static_assert(NSize == sizeof...(IRs), "NSize not consistent");

    static_assert(is_valid_sequence_map<Sequence<IRs...>>::value, "wrong! invalid reorder map");

    static_for<0, NSize, 1>{}(
        lambda_reorder_array_given_old2new<TData, NSize, Sequence<IRs...>>(old_array, new_array));

    return new_array;
}

template <typename TData, index_t NSize, typename ExtractSeq>
__host__ __device__ constexpr auto extract_array(const Array<TData, NSize>& old_array, ExtractSeq)
{
    Array<TData, ExtractSeq::GetSize()> new_array;

    constexpr index_t new_size = ExtractSeq::GetSize();

    static_assert(new_size <= NSize, "wrong! too many extract");

    static_for<0, new_size, 1>{}([&](auto I) { new_array(I) = old_array[ExtractSeq::At(I)]; });

    return new_array;
}

// emulate constepxr lambda for array
template <typename F, typename X, typename Y, typename Z>
struct lambda_array_math
{
    const F& f;
    const X& x;
    const Y& y;
    Z& z;

    __host__ __device__ constexpr lambda_array_math(const F& f_, const X& x_, const Y& y_, Z& z_)
        : f(f_), x(x_), y(y_), z(z_)
    {
    }

    template <index_t IDim_>
    __host__ __device__ constexpr void operator()(Number<IDim_>) const
    {
        constexpr auto IDim = Number<IDim_>{};
        z(IDim)             = f(x[IDim], y[IDim]);
    }
};

// Array = Sequence - Array
template <typename TData, index_t NSize, index_t... Is>
__host__ __device__ constexpr auto operator-(Sequence<Is...> a, Array<TData, NSize> b)
{
    static_assert(sizeof...(Is) == NSize, "wrong! size not the same");

    Array<TData, NSize> result;

    auto f = math::minus<index_t>{};

    static_for<0, NSize, 1>{}(
        lambda_array_math<decltype(f), decltype(a), decltype(b), decltype(result)>(
            f, a, b, result));

    return result;
}

// Array = Array * TData
template <typename TData, index_t NSize>
__host__ __device__ constexpr auto operator*(TData v, Array<TData, NSize> a)
{
    Array<TData, NSize> result;

    for(index_t i = 0; i < NSize; ++i)
    {
        result(i) = a[i] * v;
    }

    return result;
}

template <typename TData, typename Arr, typename Reduce>
__host__ __device__ constexpr TData reduce_on_array(const Arr& a, Reduce f, TData init)
{
    static_assert(is_same<typename Arr::data_type, TData>::value, "wrong! different data type");
    static_assert(Arr::Size() > 0, "wrong");

    TData result = init;

    static_for<0, Arr::Size(), 1>{}([&](auto I) { result = f(result, a[I]); });

    return result;
}

template <typename TData, index_t NSize, typename Reduce>
__host__ __device__ constexpr auto
reverse_inclusive_scan_on_array(const Array<TData, NSize>& x, Reduce f, TData init)
{
    Array<TData, NSize> y;

    TData r = init;

#pragma unroll
    for(index_t i = NSize - 1; i >= 0; --i)
    {
        r    = f(r, x[i]);
        y(i) = r;
    }

    return y;
}

template <typename TData, index_t NSize, typename Reduce>
__host__ __device__ constexpr auto
reverse_exclusive_scan_on_array(const Array<TData, NSize>& x, Reduce f, TData init)
{
    Array<TData, NSize> y;

    TData r = init;

#pragma unroll
    for(index_t i = NSize - 1; i > 0; --i)
    {
        y(i) = r;
        r    = f(r, x[i]);
    }

    y(0) = r;

    return y;
}

template <typename X, typename... Ys>
__host__ __device__ constexpr auto merge_arrays(const X& x, const Ys&... ys)
{
    return merge_arrays(x, merge_arrays(ys...));
}

template <typename T, index_t NX, index_t NY>
__host__ __device__ constexpr auto merge_arrays(const Array<T, NX>& x, const Array<T, NY>& y)
{
    Array<T, NX + NY> z;

    for(index_t i = 0; i < NX; ++i)
    {
        z(i) = x[i];
    }

    for(index_t i = 0; i < NY; ++i)
    {
        z(i + NX) = y[i];
    }

    return z;
}

template <typename X>
__host__ __device__ constexpr auto merge_arrays(const X& x)
{
    return x;
}

} // namespace ck
#endif
