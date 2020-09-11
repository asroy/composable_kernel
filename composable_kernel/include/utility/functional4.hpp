#ifndef CK_FUNCTIONAL4_HPP
#define CK_FUNCTIONAL4_HPP

#include "sequence.hpp"
#include "tuple.hpp"
#include "array.hpp"

namespace ck {

namespace detail {

template <typename Indices>
struct unpack_impl;

template <index_t... Is>
struct unpack_impl<Sequence<Is...>>
{
    template <typename F, typename X>
    __host__ __device__ constexpr auto operator()(F f, const X& x) const
    {
        return f(x.At(Number<Is>{})...);
    }
};

template <typename Seq0, typename Seq1>
struct unpack2_impl;

// TODO: remove this, after properly implementing unpack that takes any number of containers
template <index_t... Is, index_t... Js>
struct unpack2_impl<Sequence<Is...>, Sequence<Js...>>
{
    template <typename F, typename X, typename Y>
    __host__ __device__ constexpr auto operator()(F f, const X& x, const Y& y) const
    {
        return f(x.At(Number<Is>{})..., y.At(Number<Js>{})...);
    }
};

} // namespace detail

template <typename F, typename X>
__host__ __device__ constexpr auto unpack(F f, const X& x)
{
    return detail::unpack_impl<typename arithmetic_sequence_gen<0, X::Size(), 1>::type>{}(f, x);
}

// TODO: properly implement unpack that takes any number of containers
template <typename F, typename X, typename Y>
__host__ __device__ constexpr auto unpack(F f, const X& x, const Y& y)
{
    return detail::unpack2_impl<typename arithmetic_sequence_gen<0, X::Size(), 1>::type,
                                typename arithmetic_sequence_gen<0, Y::Size(), 1>::type>{}(f, x, y);
}

} // namespace ck
#endif
