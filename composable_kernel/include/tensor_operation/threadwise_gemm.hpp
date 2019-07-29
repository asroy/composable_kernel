#ifndef CK_THREADWISE_GEMM_HPP
#define CK_THREADWISE_GEMM_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "float_types.h"

namespace ck {

template <class Float, class Matrix>
__device__ void threadwise_matrix_set_zero(Matrix, Float* __restrict__ p_thread)
{
    for(index_t i = 0; i < Matrix::NRow(); ++i)
    {
        for(index_t j = 0; j < Matrix::NCol(); ++j)
        {
            const index_t id = Matrix::GetOffsetFromMultiIndex(i, j);
            p_thread[id]     = Float(0);
        }
    }
}

template <class Float,
          class SrcMatrix,
          class DstMatrix,
          index_t NRow,
          index_t NCol,
          index_t DataPerRead>
__device__ void threadwise_matrix_copy(SrcMatrix,
                                       const Float* __restrict__ p_src,
                                       DstMatrix,
                                       Float* __restrict__ p_dst,
                                       Sequence<NRow, NCol>,
                                       Number<DataPerRead>)
{
    static_assert(NCol % DataPerRead == 0, "wrong! should be NCol % == DataPerRead == 0");

    constexpr auto src_mtx = SrcMatrix{};
    constexpr auto dst_mtx = DstMatrix{};

    // Depending upon datatype i.e float/half/bfloat16, carry out data movement
    // in appropriate vectorized form
    // float - 4, half - 4, bfloat16 - 2
    static_if<std::is_same<Float, float>::value>{}([&](auto) {
        using vector_t = typename vector_type<float, DataPerRead>::MemoryType;

        for(index_t i = 0; i < NRow; ++i)
        {
            for(index_t j = 0; j < NCol; j += DataPerRead)
            {
                const index_t src_index = src_mtx.GetOffsetFromMultiIndex(i, j);
                const index_t dst_index = dst_mtx.GetOffsetFromMultiIndex(i, j);

                *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
                    *reinterpret_cast<const vector_t*>(&p_src[src_index]);
            }
        }

    }).Else([&](auto) {
        // For half/bfloat16, Float type is half4/bfloat2 respectively.
        for(index_t i = 0; i < NRow; ++i)
        {
            for(index_t j = 0; j < NCol; ++j)
            {
                const index_t src_index = src_mtx.GetOffsetFromMultiIndex(i, j);
                const index_t dst_index = dst_mtx.GetOffsetFromMultiIndex(i, j);

                *reinterpret_cast<Float*>(&p_dst[dst_index]) =
                    *reinterpret_cast<const Float*>(&p_src[src_index]);
            }
        }
    });
}

template <class MatrixA,
          class MatrixB,
          class MatrixC,
          bool TransA,
          bool TransB,
          bool TransC,
          class FloatA,
          class FloatB,
          class FloatC>
__device__ void threadwise_gemm(MatrixA,
                                integral_constant<bool, TransA>,
                                const FloatA* __restrict__ p_a_thread,
                                MatrixB,
                                integral_constant<bool, TransB>,
                                const FloatB* __restrict__ p_b_thread,
                                MatrixC,
                                integral_constant<bool, TransC>,
                                FloatC* __restrict__ p_c_thread)
{
    static_if<TransA && (!TransB) && (!TransC)>{}([&](auto) {
        constexpr auto a_mtx = MatrixA{};
        constexpr auto b_mtx = MatrixB{};
        constexpr auto c_mtx = MatrixC{};

        constexpr index_t M = c_mtx.NRow();
        constexpr index_t N = c_mtx.NCol();
        constexpr index_t K = a_mtx.NRow(); // A is transposed

        for(index_t k = 0; k < K; ++k)
        {
            for(index_t i = 0; i < M; ++i)
            {
                for(index_t j = 0; j < N; ++j)
                {
                    const index_t aindex = a_mtx.GetOffsetFromMultiIndex(k, i); // A is transposed
                    const index_t bindex = b_mtx.GetOffsetFromMultiIndex(k, j);
                    const index_t cindex = c_mtx.GetOffsetFromMultiIndex(i, j);

                    //static_if<std::is_same<FloatA, float>::value>{}([&](auto) {
                    //    p_c_thread[cindex] += 
                    //            CVT_FLOAT2ACCUM(p_a_thread[aindex]) * CVT_FLOAT2ACCUM(p_b_thread[bindex]);
                    //}).Else([&](auto) {                        
                        static_if<std::is_same<FloatA, ck::vector_type<half, 4>::MemoryType>::value>{}([&](auto) {
                            const half* s0_half = reinterpret_cast<const half*>(&p_a_thread[aindex]);
                            const half* s1_half = reinterpret_cast<const half*>(&p_b_thread[bindex]);
                            p_c_thread[cindex] += 
                                    CVT_FLOAT2ACCUM(s0_half[0]) * CVT_FLOAT2ACCUM(s1_half[0]) +
                                    CVT_FLOAT2ACCUM(s0_half[1]) * CVT_FLOAT2ACCUM(s1_half[1]) +
                                    CVT_FLOAT2ACCUM(s0_half[2]) * CVT_FLOAT2ACCUM(s1_half[2]) +
                                    CVT_FLOAT2ACCUM(s0_half[3]) * CVT_FLOAT2ACCUM(s1_half[3]);
                        }).Else([&](auto) {     
                            const ushort* s0_ushort = reinterpret_cast<const ushort*>(&p_a_thread[aindex]);
                            const ushort* s1_ushort = reinterpret_cast<const ushort*>(&p_b_thread[bindex]);
                            p_c_thread[cindex] +=  
                                    CVT_FLOAT2ACCUM(s0_ushort[0]) * CVT_FLOAT2ACCUM(s1_ushort[0]) +
                                    CVT_FLOAT2ACCUM(s0_ushort[1]) * CVT_FLOAT2ACCUM(s1_ushort[1]);   
                        });             
                   // });        
                }
            }
        }
    }).Else([&](auto fwd) {
        // not implemented
        static_assert(fwd(false), "wrong! support for this config is not implemented");
    });
}

} // namespace ck
#endif
