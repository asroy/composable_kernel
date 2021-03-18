#ifndef CK_THREADWISE_GEMM_V3_HPP
#define CK_THREADWISE_GEMM_V3_HPP

#include "common_header.hpp"
#include "math.hpp"

namespace ck {

template <typename Float, typename Desc>
__device__ void threadwise_matrix_set_zero_v3(Desc, Float* __restrict__ p_thread)
{
    static_assert(Desc::IsKnownAtCompileTime(), "wrong! Desc should be known at compile-time");

    constexpr auto I0 = Number<0>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto desc = Desc{};

    constexpr auto K = desc.GetLength(I0);
    constexpr auto H = desc.GetLength(I2);
    constexpr auto W = desc.GetLength(I3);

    static_for<0, K, 1>{}([&](auto i) {
        static_for<0, H, 1>{}([&](auto j) {
            static_for<0, W, 1>{}([&](auto k) {
                constexpr auto offset = desc.CalculateOffset(make_tuple(i, 0, j, k));

                p_thread[offset] = Float(0);
            });
        });
    });
}

// C[M, N] += transpose(A[K, M]) * B[K, N]
//   Element of matrix can be vectorized data
template <typename ADesc,
          typename BDesc,
          typename CDesc,
          typename std::enable_if<ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                                      CDesc::IsKnownAtCompileTime(),
                                  bool>::type = false>
struct ThreadwiseGemm_km_kn_mn_v3
{
    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run_source(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
        static_assert(ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                          CDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto H = BDesc{}.GetLength(I2);
        constexpr auto W = BDesc{}.GetLength(I3);

        constexpr auto CYX = ADesc{}.GetLength(I0);
        constexpr auto K   = ADesc{}.GetLength(I1);

        static_for<0, CYX, 1>{}([&](auto e) {
            static_for<0, K, 1>{}([&](auto k) {
                static_for<0, H, 1>{}([&](auto h) {
                    static_for<0, W, 1>{}([&](auto w) {
                        constexpr auto a_offset = ADesc{}.CalculateOffset(make_tuple(e, k));
                        constexpr auto b_offset = BDesc{}.CalculateOffset(make_tuple(e, 0, h, w));
                        constexpr auto c_offset = CDesc{}.CalculateOffset(make_tuple(k, 0, h, w));

                        p_c[c_offset] +=
                            inner_product_with_conversion<FloatC>{}(p_a[a_offset], p_b[b_offset]);
                    });
                });
            });
        });
    }

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
        Run_source(p_a, p_b, p_c);
    }
};

} // namespace ck
#endif
