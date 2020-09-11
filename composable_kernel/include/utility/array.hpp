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

#if 0
    __host__ __device__ explicit constexpr Array(TData x)
        : mData{x}
    {}

    __host__ __device__ explicit constexpr Array()
        : Array(TData{0})
    {}

    template <typename... Xs>
    __host__ __device__ constexpr Array(Xs... xs)
        : mData{static_cast<TData>(xs)...}
    {
        static_assert(sizeof...(Xs) == NSize, "wrong! size");
    }
#endif

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
};

// Arr: Array
// Picks: Sequence<...>
template <typename Arr, typename Picks>
struct ArrayElementPicker
{
    using type      = ArrayElementPicker;
    using data_type = typename Arr::data_type;

    __host__ __device__ constexpr ArrayElementPicker() = delete;

    __host__ __device__ explicit constexpr ArrayElementPicker(Arr& array) : mArray{array}
    {
        constexpr index_t imax = reduce_on_sequence(Picks{}, math::maxer<index_t>{}, Number<0>{});

        static_assert(imax < Arr::Size(), "wrong! exceeding # array element");
    }

    __host__ __device__ static constexpr auto Size() { return Picks::Size(); }

    template <index_t I>
    __host__ __device__ constexpr const data_type& At(Number<I>) const
    {
        static_assert(I < Size(), "wrong!");

        constexpr auto IP = Picks{}[I];
        return mArray[IP];
    }

    template <index_t I>
    __host__ __device__ constexpr data_type& At(Number<I>)
    {
        static_assert(I < Size(), "wrong!");

        constexpr auto IP = Picks{}[I];
        return mArray(IP);
    }

    __host__ __device__ constexpr const data_type& operator[](index_t i) const
    {
        index_t ip = Picks{}[i];
        return mArray[ip];
    }

    __host__ __device__ constexpr data_type& operator()(index_t i)
    {
        index_t ip = Picks{}[i];
        return mArray(ip);
    }

    template <typename T>
    __host__ __device__ constexpr auto operator=(const T& a)
    {
        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });

        return *this;
    }

    template <typename T>
    __host__ __device__ constexpr auto operator+=(const T& a)
    {
        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) += a[i]; });

        return *this;
    }

    template <typename T>
    __host__ __device__ constexpr auto operator-=(const T& a)
    {
        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) -= a[i]; });

        return *this;
    }

    private:
    Arr& mArray;
};

} // namespace ck
#endif
