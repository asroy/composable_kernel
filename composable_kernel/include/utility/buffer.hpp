#ifndef CK_BUFFER_HPP
#define CK_BUFFER_HPP

#include "float_type.hpp"

namespace ck {

template <typename T, index_t N>
struct StaticBuffer : public vector_type_maker<T, N>::type
{
    using base = typename vector_type_maker<T, N>::type;

    __host__ __device__ constexpr StaticBuffer() : base{} {}
};

template <typename T, index_t N>
__host__ __device__ constexpr auto make_static_buffer(Number<N>)
{
    return StaticBuffer<T, N>{};
}

} // namespace ck
#endif
