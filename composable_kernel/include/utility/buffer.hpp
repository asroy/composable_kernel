#ifndef CK_BUFFER_HPP
#define CK_BUFFER_HPP

#include "float_type.hpp"

namespace ck {

template <
    typename ScalarType,
    index_t N,
    typename std::enable_if<is_same<typename scalar_type<ScalarType>::type, ScalarType>::value,
                            bool>::type = false>
struct StaticBuffer : public vector_type<ScalarType, N>
{
    using base = vector_type<ScalarType, N>;

    __host__ __device__ constexpr StaticBuffer() : base{} {}

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return false; }
};

template <typename T, index_t N>
__host__ __device__ constexpr auto make_static_buffer(Number<N>)
{
    using scalar_t                      = typename scalar_type<T>::type;
    constexpr index_t scalar_per_vector = scalar_type<T>::vector_size;

    return StaticBuffer<scalar_t, N * scalar_per_vector>{};
}

template <
    typename ScalarType,
    typename std::enable_if<is_same<typename scalar_type<ScalarType>::type, ScalarType>::value,
                            bool>::type = false>
struct DynamicBuffer
{
    template <typename T>
    struct PointerWrapper
    {
        T* p_;

        __host__ __device__ constexpr const T& operator[](index_t i) const { return p_[i]; }

        __host__ __device__ constexpr T& operator()(index_t i) { return p_[i]; }
    };

    ScalarType* p_scalar_;

    __host__ __device__ constexpr DynamicBuffer(ScalarType* p_scalar) : p_scalar_{p_scalar} {}

    template <typename X,
              typename std::enable_if<
                  is_same<typename scalar_type<remove_cv_t<remove_reference_t<X>>>::type,
                          ScalarType>::value,
                  bool>::type = false>
    __host__ __device__ constexpr const auto AsType() const
    {
        return PointerWrapper<X>{reinterpret_cast<X*>(p_scalar_)};
    }

    template <typename X,
              typename std::enable_if<
                  is_same<typename scalar_type<remove_cv_t<remove_reference_t<X>>>::type,
                          ScalarType>::value,
                  bool>::type = false>
    __host__ __device__ constexpr auto AsType()
    {
        return PointerWrapper<X>{reinterpret_cast<X*>(p_scalar_)};
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return false; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return true; }
};

template <typename T>
__host__ __device__ constexpr auto make_dynamic_buffer(T* p)
{
    using scalar_t                      = typename scalar_type<T>::type;
    constexpr index_t scalar_per_vector = scalar_type<T>::vector_size;

    return DynamicBuffer<scalar_t>{p};
}

} // namespace ck
#endif
