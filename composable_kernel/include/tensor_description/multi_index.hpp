#ifndef CK_MULTI_INDEX_HPP
#define CK_MULTI_INDEX_HPP

#include "common_header.hpp"

namespace ck {

#if 1 // debug
template <index_t N>
using MultiIndex = Array<index_t, N>;

template <typename... Xs>
__host__ __device__ constexpr auto make_multi_index(Xs... xs)
{
    return MultiIndex<sizeof...(Xs)>{{static_cast<index_t>(xs)...}};
}

#else
template <index_t N>
using MultiIndex = StaticallyIndexedArray<index_t, N>;

template <typename... Xs>
__host__ __device__ constexpr auto make_multi_index(Xs... xs)
{
    return MultiIndex<sizeof...(Xs)>(static_cast<index_t>(xs)... r);
}
#endif

template <index_t NSize>
__host__ __device__ constexpr auto make_zero_multi_index()
{
    return unpack([](auto... xs) { return make_multi_index(xs...); },
                  typename uniform_sequence_gen<NSize, 0>::type{});
}

} // namespace ck
#endif
