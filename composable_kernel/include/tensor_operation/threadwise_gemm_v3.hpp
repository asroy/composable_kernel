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
        constexpr auto I4 = Number<4>{};

        constexpr auto E = ADesc{}.GetLength(I0);
        constexpr auto K = ADesc{}.GetLength(I1);

        constexpr auto N = BDesc{}.GetLength(I1);
        constexpr auto H = BDesc{}.GetLength(I2);
        constexpr auto W = BDesc{}.GetLength(I3);

#if 0
        static_for<0, E, 1>{}([&](auto e) {
            static_for<0, K, 1>{}([&](auto k) {
                if constexpr(H == 8 && W == 1)
                {
                    constexpr auto a_offset = ADesc{}.CalculateOffset(make_tuple(e, k));

                    constexpr auto b_offset_0 = BDesc{}.CalculateOffset(make_tuple(e, 0, 0, 0));
                    constexpr auto b_offset_1 = BDesc{}.CalculateOffset(make_tuple(e, 0, 1, 0));
                    constexpr auto b_offset_2 = BDesc{}.CalculateOffset(make_tuple(e, 0, 2, 0));
                    constexpr auto b_offset_3 = BDesc{}.CalculateOffset(make_tuple(e, 0, 3, 0));

                    constexpr auto b_offset_4 = BDesc{}.CalculateOffset(make_tuple(e, 0, 4, 0));
                    constexpr auto b_offset_5 = BDesc{}.CalculateOffset(make_tuple(e, 0, 5, 0));
                    constexpr auto b_offset_6 = BDesc{}.CalculateOffset(make_tuple(e, 0, 6, 0));
                    constexpr auto b_offset_7 = BDesc{}.CalculateOffset(make_tuple(e, 0, 7, 0));

                    constexpr auto c_offset_0 = CDesc{}.CalculateOffset(make_tuple(k, 0, 0, 0));
                    constexpr auto c_offset_1 = CDesc{}.CalculateOffset(make_tuple(k, 0, 1, 0));
                    constexpr auto c_offset_2 = CDesc{}.CalculateOffset(make_tuple(k, 0, 2, 0));
                    constexpr auto c_offset_3 = CDesc{}.CalculateOffset(make_tuple(k, 0, 3, 0));

                    constexpr auto c_offset_4 = CDesc{}.CalculateOffset(make_tuple(k, 0, 4, 0));
                    constexpr auto c_offset_5 = CDesc{}.CalculateOffset(make_tuple(k, 0, 5, 0));
                    constexpr auto c_offset_6 = CDesc{}.CalculateOffset(make_tuple(k, 0, 6, 0));
                    constexpr auto c_offset_7 = CDesc{}.CalculateOffset(make_tuple(k, 0, 7, 0));

                    amd_assembly_outer_product_1x4(p_a[a_offset],
                                                   p_b[b_offset_0],
                                                   p_b[b_offset_1],
                                                   p_b[b_offset_2],
                                                   p_b[b_offset_3],
                                                   p_c[c_offset_0],
                                                   p_c[c_offset_1],
                                                   p_c[c_offset_2],
                                                   p_c[c_offset_3]);

                    amd_assembly_outer_product_1x4(p_a[a_offset],
                                                   p_b[b_offset_4],
                                                   p_b[b_offset_5],
                                                   p_b[b_offset_6],
                                                   p_b[b_offset_7],
                                                   p_c[c_offset_4],
                                                   p_c[c_offset_5],
                                                   p_c[c_offset_6],
                                                   p_c[c_offset_7]);
                }
                else if constexpr(H == 4 && W == 1)
                {
                    constexpr auto a_offset = ADesc{}.CalculateOffset(make_tuple(e, k));

                    constexpr auto b_offset_0 = BDesc{}.CalculateOffset(make_tuple(e, 0, 0, 0));
                    constexpr auto b_offset_1 = BDesc{}.CalculateOffset(make_tuple(e, 0, 1, 0));
                    constexpr auto b_offset_2 = BDesc{}.CalculateOffset(make_tuple(e, 0, 2, 0));
                    constexpr auto b_offset_3 = BDesc{}.CalculateOffset(make_tuple(e, 0, 3, 0));

                    constexpr auto c_offset_0 = CDesc{}.CalculateOffset(make_tuple(k, 0, 0, 0));
                    constexpr auto c_offset_1 = CDesc{}.CalculateOffset(make_tuple(k, 0, 1, 0));
                    constexpr auto c_offset_2 = CDesc{}.CalculateOffset(make_tuple(k, 0, 2, 0));
                    constexpr auto c_offset_3 = CDesc{}.CalculateOffset(make_tuple(k, 0, 3, 0));

                    amd_assembly_outer_product_1x4(p_a[a_offset],
                                                   p_b[b_offset_0],
                                                   p_b[b_offset_1],
                                                   p_b[b_offset_2],
                                                   p_b[b_offset_3],
                                                   p_c[c_offset_0],
                                                   p_c[c_offset_1],
                                                   p_c[c_offset_2],
                                                   p_c[c_offset_3]);
                }
                else
                {
                    static_for<0, H, 1>{}([&](auto h) {
                        static_for<0, W, 1>{}([&](auto w) {
                            constexpr auto a_offset = ADesc{}.CalculateOffset(make_tuple(e, k));
                            constexpr auto b_offset =
                                BDesc{}.CalculateOffset(make_tuple(e, 0, h, w));
                            constexpr auto c_offset =
                                CDesc{}.CalculateOffset(make_tuple(k, 0, h, w));

                            amd_assembly_outer_product_1x1(
                                p_a[a_offset], p_b[b_offset], p_c[c_offset]);
                        });
                    });
                }
            });
        });
#else
        constexpr auto a_lengths_ = Sequence<K>{};
        constexpr auto b_lengths_ = Sequence<N, H, W>{};

        static_for<0, E, 1>{}([&](auto e) {
            static_ford<decltype(a_lengths_)>{}([&](auto a_idx_) {
                static_ford<decltype(b_lengths_)>{}([&](auto b_idx_) {
                    // lamda
                    // F =
                    //{
                    // auto a_index = to_multi_index(make_tuple(e, k))
                    // a_index[vec_dim] *= vec_size;
                    // return a_index;
                    //}

                    constexpr auto a_idx = generate_tuple(
                        [e, a_idx_](auto i) {
                            if constexpr(i == 0)
                            {
                                return Number<e>{};
                            }
                            else
                            {
                                return a_idx_[i - 1];
                            }
                        },
                        Number<a_lengths_.Size() + 1>{});

                    constexpr auto b_idx = generate_tuple(
                        [e, b_idx_](auto i) {
                            if constexpr(i == 0)
                            {
                                return Number<e>{};
                            }
                            else
                            {
                                return b_idx_[i - 1];
                            }
                        },
                        Number<b_lengths_.Size() + 1>{});

                    constexpr auto c_idx = generate_tuple(
                        [a_idx_, b_idx_](auto i) {
                            if constexpr(i < a_idx_.Size())
                            {
                                return a_idx_[i];
                            }
                            else
                            {
                                return b_idx_[i - a_idx_.Size()];
                            }
                        },
                        Number<a_lengths_.Size() + b_lengths_.Size()>{});

                    constexpr auto a_offset = ADesc{}.CalculateOffset(a_idx);
                    constexpr auto b_offset = BDesc{}.CalculateOffset(b_idx);
                    constexpr auto c_offset = CDesc{}.CalculateOffset(c_idx);

                    amd_assembly_outer_product_1x1(p_a[a_offset], p_b[b_offset], p_c[c_offset]);
                });
            });
        });
#endif
    }

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
        Run_source(p_a, p_b, p_c);
    }
};

} // namespace ck
#endif
