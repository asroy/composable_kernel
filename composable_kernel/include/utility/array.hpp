#ifndef CK_ARRAY_HPP
#define CK_ARRAY_HPP

#include "functional2.hpp"
#include "sequence.hpp"

namespace ck {

template <typename TData, index_t NSize>
struct Array
{
    using type      = Array<TData, NSize>;
    using data_type = TData;

    // hack: add extra element to allow empty array
    // TODO: implement empty Array
    TData mData[NSize + 1] = {0};

    __host__ __device__ static constexpr index_t Size() { return NSize; }

    // TODO: remove
    __host__ __device__ static constexpr index_t GetSize() { return Size(); }

    template <index_t I>
    __host__ __device__ constexpr const TData& At(Number<I>) const
    {
        static_assert(I < NSize, "wrong!");

        return mData[I];
    }

    template <index_t I>
    __host__ __device__ constexpr TData& At(Number<I>)
    {
        static_assert(I < NSize, "wrong!");

        return mData[I];
    }

    __host__ __device__ constexpr const TData& At(index_t i) const { return mData[i]; }

    __host__ __device__ constexpr TData& At(index_t i) { return mData[i]; }

    template <typename I>
    __host__ __device__ constexpr const TData& operator[](I i) const
    {
        return At(i);
    }

    template <typename I>
    __host__ __device__ constexpr TData& operator()(I i)
    {
        return At(i);
    }

    template <typename T>
    __host__ __device__ constexpr auto operator=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });

        return *this;
    }

    template <typename T>
    __host__ __device__ constexpr auto operator+=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) += a[i]; });

        return *this;
    }

    template <typename T>
    __host__ __device__ constexpr auto operator-=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) -= a[i]; });

        return *this;
    }

    template <typename T>
    __host__ __device__ constexpr auto operator+(const T& a) const
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        type r;

        static_for<0, Size(), 1>{}([&](auto i) { r(i) = operator[](i) + a[i]; });

        return r;
    }

    template <typename T>
    __host__ __device__ constexpr auto operator-(const T& a) const
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        type r;

        static_for<0, Size(), 1>{}([&](auto i) { r(i) = operator[](i) - a[i]; });

        return r;
    }

    template <typename T>
    __host__ __device__ constexpr auto operator*(const T& a) const
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        type r;

        static_for<0, Size(), 1>{}([&](auto i) { r(i) = operator[](i) * a[i]; });

        return r;
    }

    struct lambda_PushBack // emulate constexpr lambda
    {
        const Array<TData, NSize>& old_array;
        Array<TData, NSize + 1>& new_array;

        __host__ __device__ constexpr lambda_PushBack(const Array<TData, NSize>& old_array_,
                                                      Array<TData, NSize + 1>& new_array_)
            : old_array(old_array_), new_array(new_array_)
        {
        }

        template <index_t I>
        __host__ __device__ constexpr void operator()(Number<I>) const
        {
            new_array(Number<I>{}) = old_array[I];
        }
    };

    __host__ __device__ constexpr auto PushBack(TData x) const
    {
        Array<TData, NSize + 1> new_array;

        static_for<0, NSize, 1>{}(lambda_PushBack(*this, new_array));

        new_array(Number<NSize>{}) = x;

        return new_array;
    }

    template <index_t NAppend>
    __host__ __device__ constexpr auto Append(const Array<TData, NAppend>& xs) const
    {
        Array<TData, NSize + NAppend> r;

        static_for<0, NSize, 1>{}([&r, this ](auto i) constexpr { r(i) = (*this)[i]; });

        static_for<0, NAppend, 1>{}([&r, &xs ](auto i) constexpr { r(NSize + i) = xs[i]; });

        return r;
    }
};

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_array(const X& x, const Xs&... xs)
{
    return Array<X, sizeof...(xs) + 1>{{x, xs...}};
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

} // namespace ck
#endif
