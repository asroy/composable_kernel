#ifndef CK_THREADWISE_GEMM_HPP
#define CK_THREADWISE_GEMM_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "math.hpp"

namespace ck {

// C += transpose(A) * B
//   Element of matrix can be vectorized data
template <typename MatrixA, typename MatrixB, typename MatrixC>
struct ThreadwiseGemmTransANormalBNormalC
{
    __device__ constexpr ThreadwiseGemmTransANormalBNormalC()
    {
        static_assert(MatrixA::GetLengths()[0] == MatrixB::GetLengths()[0] &&
                          MatrixA::GetlLengths()[1] == MatrixC::GetLengths()[0] &&
                          MatrixB::GetLengths()[1] == MatrixC::GetLenths()[1],
                      "wrong!");
    }

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run_source(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
        constexpr index_t M = MatrixC::GetLengths()[0];
        constexpr index_t N = MatrixC::GetLengths()[1];
        constexpr index_t K = MatrixA::GetLengths()[0]; // A is transposed

        for(index_t k = 0; k < K; ++k)
        {
            for(index_t m = 0; m < M; ++m)
            {
                for(index_t n = 0; n < N; ++n)
                {
                    const index_t aindex = MatrixA::CalculateOffset({k, m}); // A is transposed
                    const index_t bindex = MatrixB::CalculateOffset({k, n});
                    const index_t cindex = MatrixC::CalculateOffset({m, n});

                    p_c[cindex] +=
                        inner_product_with_conversion<FloatC>{}(p_a[aindex], p_b[bindex]);
                }
            }
        }
    }

#if CK_THREADWISE_GEMM_USE_AMD_INLINE_ASM
    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run_amd_asm(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
        constexpr index_t M = MatrixC::GetLengths()[0];
        constexpr index_t N = MatrixC::GetLengths()[1];
        constexpr index_t K = MatrixA::GetLengths()[0]; // A is transposed

        static_assert(N == 4 || N == 2, "wrong! this config not supported by asm yet");

        for(index_t k = 0; k < K; ++k)
        {
            for(index_t m = 0; m < M; ++m)
            {
                const index_t aindex = MatrixA::CalculateOffset(k, m); // A is transposed

                static_if<N == 2>{}([&](auto) {
                    const index_t bindex_0 = MatrixB::CalculateOffset({k, 0});
                    const index_t bindex_1 = MatrixB::CalculateOffset({k, 1});

                    const index_t cindex_0 = MatrixC::CalculateOffset({m, 0});
                    const index_t cindex_1 = MatrixC::CalculateOffset({m, 1});

                    amd_assembly_outer_product_1x2(
                        p_a[aindex], p_b[bindex_0], p_b[bindex_1], p_c[cindex_0], p_c[cindex_1]);
                });

                static_if<N == 4>{}([&](auto) {
                    const index_t bindex_0 = MatrixB::CalculateOffset({k, 0});
                    const index_t bindex_1 = MatrixB::CalculateOffset({k, 1});
                    const index_t bindex_2 = MatrixB::CalculateOffset({k, 2});
                    const index_t bindex_3 = MatrixB::CalculateOffset({k, 3});

                    const index_t cindex_0 = MatrixC::CalculateOffset({m, 0});
                    const index_t cindex_1 = MatrixC::CalculateOffset({m, 1});
                    const index_t cindex_2 = MatrixC::CalculateOffset({m, 2});
                    const index_t cindex_3 = MatrixC::CalculateOffset({m, 3});

                    amd_assembly_outer_product_1x4(p_a[aindex],
                                                   p_b[bindex_0],
                                                   p_b[bindex_1],
                                                   p_b[bindex_2],
                                                   p_b[bindex_3],
                                                   p_c[cindex_0],
                                                   p_c[cindex_1],
                                                   p_c[cindex_2],
                                                   p_c[cindex_3]);
                });
            }
        }
    }
#endif

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
#if CK_THREADWISE_GEMM_USE_AMD_INLINE_ASM
        constexpr bool has_amd_asm = is_same<FloatC, float>{} &&
                                     ((is_same<FloatA, float>{} && is_same<FloatB, float>{}) ||
                                      (is_same<FloatA, half2_t>{} && is_same<FloatB, half2_t>{}) ||
                                      (is_same<FloatA, half4_t>{} && is_same<FloatB, half4_t>{}));

        static_if<has_amd_asm>{}([&](auto fwd) {
            Run_amd_asm(p_a, p_b, fwd(p_c));
        }).Else([&](auto) { Run_source(p_a, p_b, p_c); });
#else
        Run_source(p_a, p_b, p_c);
#endif
    }
};

} // namespace ck
#endif
