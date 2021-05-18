#ifndef CK_XDLOPS_GEMM_HPP
#define CK_XDLOPS_GEMM_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "math.hpp"
#include "amd_xdlops.hpp"

namespace ck {

enum struct mfma_instr
{
    // fp32
    mfma_f32_32x32x1xf32 = 0,
    mfma_f32_16x16x1xf32,
    mfma_f32_4x4x1xf32,
    mfma_f32_32x32x2xf32, // k reduction
    mfma_f32_16x16x4xf32, // k reduction
                          // fp16
    mfma_f32_32x32x4f16,
    mfma_f32_16x16x4f16,
    mfma_f32_4x4x4f16,
    mfma_f32_32x32x8f16,  // k reduction
    mfma_f32_16x16x16f16, // k reduction
                          // bfp16
    mfma_f32_32x32x2bf16,
    mfma_f32_16x16x2bf16,
    mfma_f32_4x4x2bf16,
    mfma_f32_32x32x4bf16, // k reduction
    mfma_f32_16x16x8bf16, // k reduction
};

template <mfma_instr instr>
struct mfma_info;

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x1xf32>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 2;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 1;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 1;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        return intrin_mfma_f32_32x32x1f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x2xf32>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 2;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 1;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        return intrin_mfma_f32_32x32x2f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x4xf32>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 4;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 1;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const float*>(a);
        const auto p_b = reinterpret_cast<const float*>(b);

        return intrin_mfma_f32_16x16x4f32(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x1xf32>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 4;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 1;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 1;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const float*>(a);
        const auto p_b = reinterpret_cast<const float*>(b);

        return intrin_mfma_f32_16x16x1f32<MPerXdlops, NPerXdlops>(p_a, p_b, reg_c);
    }
};

// treat 4x4x1 as a single-blk 4x64 mfma
template <>
struct mfma_info<mfma_instr::mfma_f32_4x4x1xf32>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 64;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = 1;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = 4;
    static constexpr index_t m               = 4;
    static constexpr index_t n               = 64;
    static constexpr index_t k               = 1;
    static constexpr index_t cycles          = 8;
    static constexpr index_t k_base          = 1;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const float*>(a);
        const auto p_b = reinterpret_cast<const float*>(b);

        return intrin_mfma_f32_4x4x1f32<MPerXdlops, NPerXdlops>::run(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x4f16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 2;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 4;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 4;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const half4_t*>(a);
        const auto p_b = reinterpret_cast<const half4_t*>(b);

        return intrin_mfma_f32_32x32x4f16<MPerXdlops, NPerXdlops, AStride, BStride>::run(
            p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x8f16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 8;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 4;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const half4_t*>(a);
        const auto p_b = reinterpret_cast<const half4_t*>(b);

        return intrin_mfma_f32_32x32x8f16(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x16f16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 16;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 4;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const half4_t*>(a);
        const auto p_b = reinterpret_cast<const half4_t*>(b);

        return intrin_mfma_f32_16x16x16f16(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x4f16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 4;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 4;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 4;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const half4_t*>(a);
        const auto p_b = reinterpret_cast<const half4_t*>(b);

        return intrin_mfma_f32_16x16x4f16<MPerXdlops, NPerXdlops>(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_4x4x4f16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 64;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = 1;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = 4;
    static constexpr index_t m               = 4;
    static constexpr index_t n               = 64;
    static constexpr index_t k               = 4;
    static constexpr index_t cycles          = 8;
    static constexpr index_t k_base          = 4;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const half4_t*>(a);
        const auto p_b = reinterpret_cast<const half4_t*>(b);

        return intrin_mfma_f32_4x4x4f16<MPerXdlops, NPerXdlops>::run(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x2bf16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 2;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 2;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 2;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const ushort2_t*>(a);
        const auto p_b = reinterpret_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_32x32x2bf16<MPerXdlops, NPerXdlops, AStride, BStride>::run(
            p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x4bf16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 4;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 32;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 32;
    static constexpr index_t n               = 32;
    static constexpr index_t k               = 4;
    static constexpr index_t cycles          = 64;
    static constexpr index_t k_base          = 2;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const ushort2_t*>(a);
        const auto p_b = reinterpret_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_32x32x4bf16(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x8bf16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 8;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 2;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const ushort2_t*>(a);
        const auto p_b = reinterpret_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_16x16x8bf16(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x2bf16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 16;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = wave_size / num_threads_blk;
    static constexpr index_t num_output_blks = 4;
    static constexpr index_t num_regs_xdlops = num_regs_blk * num_output_blks;
    static constexpr index_t m               = 16;
    static constexpr index_t n               = 16;
    static constexpr index_t k               = 2;
    static constexpr index_t cycles          = 32;
    static constexpr index_t k_base          = 2;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const ushort2_t*>(a);
        const auto p_b = reinterpret_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_16x16x2bf16<MPerXdlops, NPerXdlops>(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_4x4x2bf16>
{
    static constexpr index_t group_size      = 4;
    static constexpr index_t num_groups_blk  = 1;
    static constexpr index_t num_regs_blk    = group_size * num_groups_blk;
    static constexpr index_t num_threads_blk = 64;
    static constexpr index_t wave_size       = 64;
    static constexpr index_t num_input_blks  = 1;
    static constexpr index_t num_output_blks = 1;
    static constexpr index_t num_regs_xdlops = 4;
    static constexpr index_t m               = 4;
    static constexpr index_t n               = 64;
    static constexpr index_t k               = 2;
    static constexpr index_t cycles          = 8;
    static constexpr index_t k_base          = 2;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = reinterpret_cast<const ushort2_t*>(a);
        const auto p_b = reinterpret_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_4x4x2bf16<MPerXdlops, NPerXdlops>::run(p_a, p_b, reg_c);
    }
};

template <mfma_instr instr,
          index_t MPerXdlops_,
          index_t NPerXdlops_,
          index_t MRepeats_,
          index_t NRepeats_,
          class OutputVecType_>
struct xdlops_info
{
    static constexpr auto mfma_type = mfma_info<instr>{};

    static constexpr index_t MPerXdlops = MPerXdlops_;
    static constexpr index_t NPerXdlops = NPerXdlops_;
    static constexpr index_t MRepeats   = MRepeats_;
    static constexpr index_t NRepeats   = NRepeats_;

    static constexpr bool IsABroadcast() { return NPerXdlops >= MPerXdlops; }

    static constexpr bool IsKReduction()
    {
        return (mfma_type.num_output_blks == 1) && (mfma_type.num_input_blks > 1);
    }

    static constexpr auto OutputVecType = OutputVecType_{};
};

template <class data_type, index_t MPerWave, index_t NPerWave, index_t KPerWave>
struct XdlopsGemm
{
    struct MatrixIndex
    {
        index_t row;
        index_t col;
    };

    __device__ static constexpr index_t GetNumBlksPerXdlops()
    {
        return (MPerXdlops * NPerXdlops) / (mfma_type.m * mfma_type.n);
    }

    __host__ __device__ constexpr XdlopsGemm()
    {
        static_assert(NPerXdlops == 4 || NPerXdlops == 8 || NPerXdlops == 16 || NPerXdlops == 32 ||
                          NPerXdlops == 64,
                      "Only support GemmNPerXdlops == 4, 8, 16, 32 or 64 for xdlops");

        static_assert(MPerXdlops == 4 || MPerXdlops == 8 || MPerXdlops == 16 || MPerXdlops == 32 ||
                          MPerXdlops == 64,
                      "Only support GemmMPerXdlops == 4, 8, 16, 32 or 64 for xdlops");

        static_assert(mfma_type.num_threads_blk == mfma_type.n, "n != num_threads_blk");
        static_assert(mfma_type.num_regs_blk * mfma_type.num_input_blks == mfma_type.m,
                      "m != num_input_blks * num_regs_blk");
        static_assert(mfma_type.num_output_blks == mfma_type.num_input_blks ||
                          mfma_type.num_output_blks == 1,
                      "incorrect num_output_blks");
        static_assert(mfma_type.num_regs_blk * mfma_type.wave_size == mfma_type.m * mfma_type.n,
                      "num_regs_blk incorrect");

        static_assert(mfma_type.k % mfma_type.k_base == 0, "k and k_base is inconsistent!");
    }

    __device__ static constexpr index_t GetRegSizePerXdlops()
    {
        return MPerXdlops * NPerXdlops / mfma_type.wave_size;
    }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
    {
        static_assert(is_same<data_type, float>::value || is_same<data_type, half_t>::value ||
                          is_same<data_type, ushort>::value,
                      "base data_type must be float, half, ushort!");

        static_for<0, KPerWave, mfma_type.k_base>{}([&](auto k_i) {
            mfma_type.template run<MPerXdlops, NPerXdlops>(
                p_a_wave[Number<k_i>{}], p_b_wave[Number<k_i>{}], p_c_thread);
        });
    }

    __device__ static MatrixIndex GetBeginOfThreadBlk(index_t i)
    {
        const index_t xdlops_i = i / GetNumBlksPerXdlops();
        const index_t j        = i % GetNumBlksPerXdlops();

        const index_t m_i = xdlops_i / NRepeats;
        const index_t n_i = xdlops_i % NRepeats;

        const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
        const index_t blk_id = laneId / mfma_type.num_threads_blk;
        const index_t blk_td = laneId % mfma_type.num_threads_blk;

        index_t col_blk = j % mfma_type.num_output_blks;
        index_t row_blk = j / mfma_type.num_output_blks;

        static_if<!IsABroadcast>{}([&](auto) {
            col_blk = j / mfma_type.num_output_blks;
            row_blk = j % mfma_type.num_output_blks;
        });

        index_t col = col_blk * mfma_type.n + blk_td + n_i * NPerXdlops;
        index_t row = row_blk * mfma_type.m + blk_id * mfma_type.group_size + m_i * MPerXdlops;

        return MatrixIndex{row, col};
    }

    __device__ void SetZeroXdlopsRegs() const {}

    template <class FloatC>
    __device__ void ReadXdlopsRegs(FloatC* const __restrict__) const
    {
    }

    template <class data_type_  = data_type,
              index_t MPerWave_ = MPerWave,
              index_t NPerWave_ = NPerWave>
    static constexpr auto GetXdlopsInfo();

    template <>
    static constexpr auto GetXdlopsInfo<float, 128, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x1xf32, 64, 64, 2, 1, c_vec32_4_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 64, 128>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x1xf32, 64, 64, 1, 2, c_vec32_4_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 64, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x1xf32, 64, 64, 1, 1, c_vec32_2_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 64, 32>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x1xf32, 64, 32, 1, 1, c_vec32_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 32, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x1xf32, 32, 64, 1, 1, c_vec32_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 64, 16>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x1xf32, 64, 16, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 16, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x1xf32, 16, 64, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 8, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x1xf32, 8, 64, 1, 1, c_vec4_2_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 4, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x1xf32, 4, 64, 1, 1, c_vec4_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 32, 32>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2xf32, 32, 32, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 16, 16>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x4xf32, 16, 16, 1, 1, c_vec4_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 128, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x4f16, 64, 64, 2, 1, c_vec32_4_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 64, 128>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x4f16, 64, 64, 1, 2, c_vec32_4_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 64, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x4f16, 64, 64, 1, 1, c_vec32_2_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 64, 32>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x4f16, 64, 32, 1, 1, c_vec32_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 32, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x4f16, 32, 64, 1, 1, c_vec32_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 64, 16>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x4f16, 64, 16, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 16, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x4f16, 16, 64, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 8, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x4f16, 8, 64, 1, 1, c_vec4_2_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 4, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x4f16, 4, 64, 1, 1, c_vec4_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 32, 32>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x8f16, 32, 32, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 16, 16>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x16f16, 16, 16, 1, 1, c_vec4_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 128, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2bf16, 64, 64, 2, 1, c_vec32_4_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 64, 128>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2bf16, 64, 64, 1, 2, c_vec32_4_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 64, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2bf16, 64, 64, 1, 1, c_vec32_2_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 64, 32>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2bf16, 64, 32, 1, 1, c_vec32_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 32, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2bf16, 32, 64, 1, 1, c_vec32_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 64, 16>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x2bf16, 64, 16, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 16, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x2bf16, 16, 64, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 8, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x2bf16, 8, 64, 1, 1, c_vec4_2_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 4, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x2bf16, 4, 64, 1, 1, c_vec4_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 32, 32>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x4bf16, 32, 32, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 16, 16>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x8bf16, 16, 16, 1, 1, c_vec4_1_t>{};
    }

    static constexpr index_t MRepeats   = GetXdlopsInfo().MRepeats;
    static constexpr index_t NRepeats   = GetXdlopsInfo().NRepeats;
    static constexpr index_t MPerXdlops = GetXdlopsInfo().MPerXdlops;
    static constexpr index_t NPerXdlops = GetXdlopsInfo().NPerXdlops;

    static constexpr bool IsKReduction = GetXdlopsInfo().IsKReduction();
    static constexpr bool IsABroadcast = GetXdlopsInfo().IsABroadcast();

    static constexpr auto mfma_type = GetXdlopsInfo().mfma_type;

    struct OutputLayout
    {
        __host__ __device__ static constexpr index_t M1() { return mfma_type.num_groups_blk; }
        __host__ __device__ static constexpr index_t M0() { return mfma_type.group_size; }
        __host__ __device__ static constexpr index_t N1() { return mfma_type.num_input_blks; }
        __host__ __device__ static constexpr index_t N0() { return mfma_type.num_threads_blk; }

        __device__ static constexpr index_t GetBlkSize() { return mfma_type.num_regs_blk; }

        __device__ static constexpr index_t GetNumBlks()
        {
            return GetNumBlksPerXdlops() * MRepeats * NRepeats;
        }

        __device__ static constexpr auto CreateOutputVecZero()
        {
            return GetXdlopsInfo().OutputVecType.CreateVecZero();
        }
    };

    __host__ __device__ static constexpr auto GetOutputLayout() { return OutputLayout{}; }
};

} // namespace ck
#endif
