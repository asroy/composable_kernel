#ifndef CK_FLOAT_TYPE_AMD_HPP
#define CK_FLOAT_TYPE_AMD_HPP

namespace ck {

// For some reason, HIP compiler need this definition to generate optimal ISA
// float
typedef float float2_t __attribute__((ext_vector_type(2)));
typedef float float4_t __attribute__((ext_vector_type(4)));
typedef float float32_t __attribute__((ext_vector_type(32)));

// float16
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));

// bfloat16
typedef ushort ushort2_t __attribute__((ext_vector_type(2)));
typedef ushort ushort4_t __attribute__((ext_vector_type(4)));

// data type conversion
template <typename T>
struct type_convert
{
    template <typename X>
    __device__ T operator()(X x) const
    {
        return static_cast<T>(x);
    }
};

template <>
template <>
__device__ float type_convert<float>::operator()<ushort>(ushort x) const
{
    return bfloat16_to_float(x);
}

template <>
template <>
__device__ ushort type_convert<ushort>::operator()<float>(float x) const
{
    return float_to_bfloat16(x);
}

template <typename T>
struct inner_product_with_conversion
{
    static constexpr auto convert = type_convert<T>();

    __device__ T operator()(float a, float b) const { return convert(a) * convert(b); }

    __device__ T operator()(half2_t a, half2_t b) const
    {
        const half* p_a_half = reinterpret_cast<const half*>(&a);
        const half* p_b_half = reinterpret_cast<const half*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += convert(p_a_half[v]) * convert(p_b_half[v]);
        }

        return acc;
    }

    __device__ T operator()(half4_t a, half4_t b) const
    {
        const half* p_a_half = reinterpret_cast<const half*>(&a);
        const half* p_b_half = reinterpret_cast<const half*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += convert(p_a_half[v]) * convert(p_b_half[v]);
        }
        return acc;
    }

    __device__ T operator()(ushort2_t a, ushort2_t b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += convert(p_a_bfloat16[v]) * convert(p_b_bfloat16[v]);
        }

        return acc;
    }

    __device__ T operator()(ushort4_t a, ushort4_t b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += convert(p_a_bfloat16[v]) * convert(p_b_bfloat16[v]);
        }
        return acc;
    }
};

} // namespace ck
#endif
