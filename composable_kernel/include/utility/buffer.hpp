#ifndef CK_BUFFER_HPP
#define CK_BUFFER_HPP

#include "statically_indexed_array.hpp"

namespace ck {

template <typename T, index_t N>
struct StaticBuffer : public StaticallyIndexedArray<T, N>
{
    using type = T;
    using base = StaticallyIndexedArray<T, N>;

    __host__ __device__ constexpr StaticBuffer() : base{} {}

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return false; }
};

template <typename T, index_t N>
__host__ __device__ constexpr auto make_static_buffer(Number<N>)
{
    return StaticBuffer<T, N>{};
}

template <typename T>
struct DynamicBuffer
{
    using type = T;

    T* p_data_;

    __host__ __device__ constexpr DynamicBuffer(T* p_data) : p_data_{p_data} {}

    __host__ __device__ constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    __host__ __device__ constexpr T& operator()(index_t i) { return p_data_[i]; }

    template <typename X,
              typename std::enable_if<
                  is_same<typename scalar_type<remove_cv_t<remove_reference_t<X>>>::type,
                          typename scalar_type<remove_cv_t<remove_reference_t<T>>>::type>::value,
                  bool>::type = false>
    __host__ __device__ constexpr const auto Get(index_t i) const
    {
#if !CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
        return *reinterpret_cast<const X*>(&p_data_[i]);
#else
        return *reinterpret_cast<const X*>(&p_data_[i]);
#endif
    }

    template <typename X,
              typename std::enable_if<
                  is_same<typename scalar_type<remove_cv_t<remove_reference_t<X>>>::type,
                          typename scalar_type<remove_cv_t<remove_reference_t<T>>>::type>::value,
                  bool>::type = false>
    __host__ __device__ void Set(index_t i, const X& x)
    {
#if !CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
        *reinterpret_cast<X*>(&p_data_[i]) = x;
#else
        if constexpr(is_same<typename scalar_type<remove_cv_t<remove_reference_t<T>>>::type,
                             int8_t>::value)
        {
            static_assert(is_same<remove_cv_t<remove_reference_t<T>>, int8x16_t>::value &&
                              is_same<remove_cv_t<remove_reference_t<X>>, int8x16_t>::value,
                          "wrong! not implemented for this combination, please add implementation");

            if constexpr(is_same<remove_cv_t<remove_reference_t<T>>, int8x16_t>::value &&
                         is_same<remove_cv_t<remove_reference_t<X>>, int8x16_t>::value)
            {
#if 0
                *reinterpret_cast<int32x4_t*>(&p_data_[i]) = as_type<int32x4_t>(x);
#else
                *reinterpret_cast<int32x4_t*>(&p_data_[i]) =
                    *reinterpret_cast<const int32x4_t*>(&x);
#endif
            }
        }
        else
        {
            *reinterpret_cast<X*>(&p_data_[i]) = x;
        }
#endif
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return false; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return true; }
};

template <typename T>
__host__ __device__ constexpr auto make_dynamic_buffer(T* p)
{
    return DynamicBuffer<T>{p};
}

} // namespace ck
#endif
