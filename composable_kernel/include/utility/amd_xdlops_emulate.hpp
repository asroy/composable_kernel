#ifndef CK_AMD_XDLOPS_HPP
#define CK_AMD_XDLOPS_HPP

namespace ck {

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x1f32(const float&, const float&, float32_t*);

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<64, 64>(const float& reg_a, const float& reg_b, float32_t* reg_c)
{
    auto reg_c_ = reinterpret_cast<float_t*>(reg_c);
    for(index_t i = 0; i < 32; i++)
    {
        reg_c_[i] += reg_a * reg_b;
        reg_c_[i+32] = reg_c[i];
    }
}

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<32, 64>(const float& reg_a, const float& reg_b, float32_t* reg_c)
{
    auto reg_c_ = reinterpret_cast<float_t*>(reg_c);
    for(index_t i = 0; i < 16; i++)
    {
        reg_c_[i] += reg_a * reg_b;
        reg_c_[i+16] = reg_c[i];
    }
}

template <>
__device__ void gcnasm_mfma_f32_32x32x1f32<64, 32>(const float& reg_a, const float& reg_b, float32_t* reg_c)
{
    auto reg_c_ = reinterpret_cast<float_t*>(reg_c);
    for(index_t i = 0; i < 16; i++)
    {
        reg_c_[i] += reg_a * reg_b;
        reg_c_[i+16] = reg_c[i];
    }
}

__device__ void gcnasm_mfma_f32_32x32x2f32(const float& reg_a, const float& reg_b, float16_t* reg_c)
{
    auto reg_c_ = reinterpret_cast<float_t*>(reg_c);
    for(index_t i = 0; i < 16; i++)
    {
        reg_c_[i] += reg_a * reg_b;
    }
}

__device__ void gcnasm_mfma_f32_16x16x4f32(const float& reg_a, const float& reg_b, float4_t* reg_c)
{
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_16x16x1f32(const float&, const float&, float16_t*);

template <>
__device__ void gcnasm_mfma_f32_16x16x1f32<16, 64>(const float& reg_a, const float& reg_b, float16_t* reg_c)
{
}

template <>
__device__ void gcnasm_mfma_f32_16x16x1f32<64, 16>(const float& reg_a, const float& reg_b, float16_t* reg_c)
{
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x1f32(const float& reg_a, const float& reg_b, float4_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_4x4x1f32<4, 64>(const float& reg_a, const float& reg_b, float4_t* reg_c)
{
}

template <>
__device__ void gcnasm_mfma_f32_4x4x1f32<8, 64>(const float& reg_a, const float& reg_b, float4_t* reg_c)
{
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x4f16(const half4_t&,
                                           const half4_t&,
                                           float32_t*);
template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<64, 64>(const half4_t& reg_a, const half4_t& reg_b, float32_t* reg_c)
{
}

template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<32, 64>(const half4_t& reg_a, const half4_t& reg_b, float32_t* reg_c)
{
}

template <>
__device__ void gcnasm_mfma_f32_32x32x4f16<64, 32>(const half4_t& reg_a, const half4_t& reg_b, float32_t* reg_c)
{
}

__device__ void gcnasm_mfma_f32_32x32x8f16(const half4_t& reg_a, const half4_t& reg_b, float16_t* reg_c)
{
}

__device__ void gcnasm_mfma_f32_16x16x16f16(const half4_t& reg_a, const half4_t& reg_b, float4_t* reg_c)
{
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_16x16x4f16(const half4_t& reg_a, const half4_t& reg_b, float16_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_16x16x4f16<16, 64>(const half4_t& reg_a, const half4_t& reg_b, float16_t* reg_c)
{
}

template <>
__device__ void gcnasm_mfma_f32_16x16x4f16<64, 16>(const half4_t& reg_a, const half4_t& reg_b, float16_t* reg_c)

{
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x4f16(const half4_t& reg_a, const half4_t& reg_b, float4_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_4x4x4f16<4, 64>(const half4_t& reg_a, const half4_t& reg_b, float4_t* reg_c)
{
}

template <>
__device__ void gcnasm_mfma_f32_4x4x4f16<8, 64>(const half4_t& reg_a, const half4_t& reg_b, float4_t* reg_c)
{
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_32x32x2bf16(const ushort2_t&, const ushort2_t&, float32_t*);

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<64, 64>(const ushort2_t& reg_a, const ushort2_t& reg_b, float32_t* reg_c)
{
}

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<32, 64>(const ushort2_t& reg_a, const ushort2_t& reg_b, float32_t* reg_c)
{
}

template <>
__device__ void gcnasm_mfma_f32_32x32x2bf16<64, 32>(const ushort2_t& reg_a, const ushort2_t& reg_b, float32_t* reg_c)
{
}

__device__ void gcnasm_mfma_f32_32x32x4bf16(const ushort2_t& reg_a, const ushort2_t& reg_b, float16_t* reg_c)
{
}

__device__ void gcnasm_mfma_f32_16x16x8bf16(const ushort2_t& reg_a, const ushort2_t& reg_b, float4_t* reg_c)
{
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_16x16x2bf16(const ushort2_t& reg_a, const ushort2_t& reg_b, float16_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_16x16x2bf16<16, 64>(const ushort2_t& reg_a, const ushort2_t& reg_b, float16_t* reg_c)
{
}

template <>
__device__ void gcnasm_mfma_f32_16x16x2bf16<64, 16>(const ushort2_t& reg_a, const ushort2_t& reg_b, float16_t* reg_c)
{
}

template <index_t MPerWave, index_t NPerWave>
__device__ void gcnasm_mfma_f32_4x4x2bf16(const ushort2_t& reg_a, const ushort2_t& reg_b, float4_t* reg_c);

template <>
__device__ void gcnasm_mfma_f32_4x4x2bf16<4, 64>(const ushort2_t& reg_a, const ushort2_t& reg_b, float4_t* reg_c)
{
}

template <>
__device__ void gcnasm_mfma_f32_4x4x2bf16<8, 64>(const ushort2_t& reg_a, const ushort2_t& reg_b, float4_t* reg_c)
{
}
// clang-format on

}
#endif
