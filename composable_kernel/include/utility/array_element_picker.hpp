#ifndef CK_ARRAY_ELEMENT_PICKER_HPP
#define CK_ARRAY_ELEMENT_PICKER_HPP

#include "functional2.hpp"
#include "sequence.hpp"

namespace ck {

// Arr: Array or StaticallyIndexedArray
// Picks: Sequence<...>
template <typename Arr, typename Picks>
struct ArrayElementPicker
{
    using type = ArrayElementPicker;
#if 0
    using data_type = typename Arr::data_type;
#endif

    __host__ __device__ constexpr ArrayElementPicker() = delete;

    __host__ __device__ explicit constexpr ArrayElementPicker(Arr& array) : mArray{array}
    {
        constexpr index_t imax = reduce_on_sequence(Picks{}, math::maxer<index_t>{}, Number<0>{});

        static_assert(imax < Arr::Size(), "wrong! exceeding # array element");
    }

    __host__ __device__ static constexpr auto Size() { return Picks::Size(); }

    template <index_t I>
    __host__ __device__ constexpr const auto& At(Number<I> i) const
    {
        static_assert(I < Size(), "wrong!");

        constexpr auto IP = Picks{}[i];
        return mArray[IP];
    }

    template <index_t I>
    __host__ __device__ constexpr auto& At(Number<I> i)
    {
        static_assert(I < Size(), "wrong!");

        constexpr auto IP = Picks{}[i];
        return mArray(IP);
    }

    template <index_t I>
    __host__ __device__ constexpr const auto& operator[](Number<I> i) const
    {
        return At(i);
    }

    template <index_t I>
    __host__ __device__ constexpr auto& operator()(Number<I> i)
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

    private:
    Arr& mArray;
};

template <typename Arr, typename Picks, typename X>
__host__ __device__ constexpr auto operator+=(ArrayElementPicker<Arr, Picks>& y, const X& x)
{
    using Y                 = ArrayElementPicker<Arr, Picks>;
    constexpr index_t nsize = Y::Size();

    static_assert(nsize == X::Size(), "wrong! size not the same");

    static_for<0, nsize, 1>{}([&](auto i) { y(i) += x[i]; });

    return y;
}

template <typename Arr, typename Picks, typename X>
__host__ __device__ constexpr auto operator-=(ArrayElementPicker<Arr, Picks>& y, const X& x)
{
    using Y                 = ArrayElementPicker<Arr, Picks>;
    constexpr index_t nsize = Y::Size();

    static_assert(nsize == X::Size(), "wrong! size not the same");

    static_for<0, nsize, 1>{}([&](auto i) { y(i) -= x[i]; });

    return y;
}

} // namespace ck
#endif
