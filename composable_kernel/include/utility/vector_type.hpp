#ifndef CK_VECTOR_TYPE_HPP
#define CK_VECTOR_TYPE_HPP

#include "cuda_fp16.h"
#include "config.hpp"
#include "integral_constant.hpp"

namespace ck {

template <class T, index_t N>
struct vector_type
{
    T vector[N];
};

template <>
struct vector_type<float, 1>
{
    using MemoryType = float;

    __host__ __device__ static constexpr index_t GetSize() { return 1; }

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, float s, Number<I>)
    {
        static_assert(I < 1, "wrong");
        *(reinterpret_cast<float*>(&v) + I) = s;
    }
};

template <>
struct vector_type<float, 2>
{
    using MemoryType = float2_t;

    __host__ __device__ static constexpr index_t GetSize() { return 2; }

    union Data
    {
        MemoryType vector;
        float scalar[2];
    };

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, float s, Number<I>)
    {
        static_assert(I < 2, "wrong");
        *(reinterpret_cast<float*>(&v) + I) = s;
    }

};

template <>
struct vector_type<float, 4>
{
    using MemoryType = float4_t;

    __host__ __device__ static constexpr index_t GetSize() { return 4; }

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, float s, Number<I>)
    {
        static_assert(I < 4, "wrong");
        *(reinterpret_cast<float*>(&v) + I) = s;
    }
};

template <>
struct vector_type<half, 1>
{
    using MemoryType = half;

    __host__ __device__ static constexpr index_t GetSize() { return 1; }

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, half s, Number<I>)
    {
        static_assert(I < 1, "wrong");
        *(reinterpret_cast<half*>(&v) + I) = s;
    }
};

template <>
struct vector_type<half, 2>
{
    using MemoryType = half2;

    union Data
    {
        MemoryType vector;
        half scalar[2];
    };

    __host__ __device__ static constexpr index_t GetSize() { return 2; }

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, half s, Number<I>)
    {
        static_assert(I < 2, "wrong");
        *(reinterpret_cast<half*>(&v) + I) = s;
    }

};

template <>
struct vector_type<half, 4>
{
    typedef struct MemoryType
    {
        half2 vector[2];
    } MemoryType;

    __host__ __device__ static constexpr index_t GetSize() { return 4; }

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, half s, Number<I>)
    {
        static_assert(I < 4, "wrong");
        *(reinterpret_cast<half*>(&v) + I) = s;
    }
};

template <>
struct vector_type<ushort, 1>
{
    using MemoryType = ushort;

    __host__ __device__ static constexpr index_t GetSize() { return 1; }

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, ushort s, Number<I>)
    {
        static_assert(I < 1, "wrong");
        *(reinterpret_cast<ushort*>(&v) + I) = s;
    }
};

template <>
struct vector_type<ushort, 2>
{
    using MemoryType = ushort2;

    union Data
    {
        MemoryType vector;
        half scalar[2];
    };

    __host__ __device__ static constexpr index_t GetSize() { return 2; }

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, ushort s, Number<I>)
    {
        static_assert(I < 2, "wrong");
        *(reinterpret_cast<ushort*>(&v) + I) = s;
    }

};

template <>
struct vector_type<ushort, 4>
{
    typedef struct MemoryType
    {
        ushort2 vector[2];
    } MemoryType;

    __host__ __device__ static constexpr index_t GetSize() { return 4; }

    template <index_t I>
    __host__ __device__ static void SetScalar(MemoryType& v, ushort s, Number<I>)
    {
        static_assert(I < 4, "wrong");
        *(reinterpret_cast<ushort*>(&v) + I) = s;
    }
};

} // namespace ck

#endif
