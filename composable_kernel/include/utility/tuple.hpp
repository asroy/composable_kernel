#ifndef CK_TUPLE_HPP
#define CK_TUPLE_HPP

#include "integral_constant.hpp"
#include "sequence.hpp"
#include "type.hpp"

namespace ck {

namespace detail {

template <index_t>
struct TupleElementKey
{
};

template <typename Key, typename Data>
struct TupleElement
{
    __host__ __device__ explicit constexpr TupleElement() = default;

    template <typename UData>
    __host__ __device__ explicit constexpr TupleElement(const TupleElement<Key, UData>& te)
        : mData(static_cast<const UData&>(te.mData))
    {
    }

    template <typename UData>
    __host__ __device__ explicit constexpr TupleElement(TupleElement<Key, UData>&& te)
        : mData(static_cast<UData&&>(te.mData))
    {
    }

    template <typename T>
    __host__ __device__ explicit constexpr TupleElement(T&& v) : mData(std::forward<T>(v))
    {
    }

    Data mData;
};

template <typename Key, typename Data>
__host__ __device__ constexpr const Data& get_tuple_element(const TupleElement<Key, Data>& x)
{
    return static_cast<const Data&>(x.mData);
}

template <typename Key, typename Data>
__host__ __device__ constexpr Data& get_tuple_element(TupleElement<Key, Data>& x)
{
    return x.mData;
}

// TODO: not sure the use of reference is correct
template <typename Key, typename Data>
__host__ __device__ constexpr Data&& get_tuple_element(TupleElement<Key, Data>&& x)
{
    return static_cast<Data&&>(x.mData);
}

template <typename Indices, typename... Xs>
struct TupleImpl;

template <index_t... Is, typename... Xs>
struct TupleImpl<Sequence<Is...>, Xs...> : TupleElement<TupleElementKey<Is>, Xs>...
{
#if 1
    __host__ __device__ explicit constexpr TupleImpl() = default;

    template <typename... Ys, typename std::enable_if<sizeof...(Ys) >= 1, bool>::type = false>
    __host__ __device__ explicit constexpr TupleImpl(Ys&&... ys)
        : TupleElement<TupleElementKey<Is>, Xs>(std::forward<Ys>(ys))...
    {
        static_assert(sizeof...(Is) == sizeof...(Xs) && sizeof...(Is) == sizeof...(Ys),
                      "wrong! inconsistent size");
    }
#else
    __host__ __device__ explicit constexpr TupleImpl() = default;

    template <typename Y,
              typename std::enable_if<sizeof...(Is) == 1 && sizeof...(Xs) == 1 &&
                                          !is_same<remove_cv_t<Y>, TupleImpl>::value,
                                      bool>::type = false>
    __host__ __device__ explicit constexpr TupleImpl(Y&& y)
        : TupleElement<TupleElementKey<Is>, Xs>(std::forward<Y>(y))...
    {
    }

    template <typename... Ys, typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
    __host__ __device__ explicit constexpr TupleImpl(Ys&&... ys)
        : TupleElement<TupleElementKey<Is>, Xs>(std::forward<Ys>(ys))...
    {
        static_assert(sizeof...(Is) == sizeof...(Xs) && sizeof...(Is) == sizeof...(Ys),
                      "wrong! inconsistent size");
    }
#endif

    __host__ __device__ static constexpr index_t Size() { return sizeof...(Xs); }

    template <index_t I>
    __host__ __device__ constexpr const auto& GetElementByKey(TupleElementKey<I>) const
    {
        return get_tuple_element<TupleElementKey<I>>(*this);
    }

    template <index_t I>
    __host__ __device__ constexpr auto& GetElementByKey(TupleElementKey<I>)
    {
        return get_tuple_element<TupleElementKey<I>>(*this);
    }
};

} // namespace detail

template <typename... Xs>
struct Tuple : detail::TupleImpl<typename arithmetic_sequence_gen<0, sizeof...(Xs), 1>::type, Xs...>
{
    using base =
        detail::TupleImpl<typename arithmetic_sequence_gen<0, sizeof...(Xs), 1>::type, Xs...>;

    __host__ __device__ constexpr Tuple() = default;

    template <typename Y,
              typename std::enable_if<sizeof...(Xs) == 1 && !is_same<remove_cv_t<Y>, Tuple>::value,
                                      bool>::type = false>
    __host__ __device__ explicit constexpr Tuple(Y&& y) : base(std::forward<Y>(y))
    {
    }

    template <typename... Ys,
              typename std::enable_if<sizeof...(Ys) == sizeof...(Xs) && sizeof...(Ys) >= 2,
                                      bool>::type = false>
    __host__ __device__ explicit constexpr Tuple(Ys&&... ys) : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ static constexpr index_t Size() { return sizeof...(Xs); }

    template <index_t I>
    __host__ __device__ constexpr const auto& At(Number<I>) const
    {
        static_assert(I < base::Size(), "wrong! out of range");
        return base::GetElementByKey(detail::TupleElementKey<I>{});
    }

    template <index_t I>
    __host__ __device__ constexpr auto& At(Number<I>)
    {
        static_assert(I < base::Size(), "wrong! out of range");
        return base::GetElementByKey(detail::TupleElementKey<I>{});
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
};

template <typename... Xs>
__host__ __device__ constexpr auto make_tuple(Xs&&... xs)
{
    return Tuple<remove_cv_t<remove_reference_t<Xs>>...>(std::forward<Xs>(xs)...);
}

} // namespace ck
#endif
