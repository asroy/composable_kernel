#ifndef CK_BLOCKWISE_GEMM_V3_HPP
#define CK_BLOCKWISE_GEMM_V3_HPP

#include "common_header.hpp"
#include "threadwise_gemm_v3.hpp"

namespace ck {

#if 0
// blockwise GEMM: C[M, N] += transpose(A[K, M]) * B[K, N]
// A and B are visable to the whole block, C is distributed among each thread
// If following number are power of 2, index calculation shall be greatly reduced:
//    KPerThread, HPerThread, MLevel0ThreadCluster, NLevel0ThreadCluster,
//    MLevel1ThreadCluster, NLevel1ThreadCluster
template <index_t BlockSize,
          typename BlockMatrixA,
          typename BlockMatrixB,
          typename ThreadMatrixC,
          index_t KPerThread,
          index_t HPerThread,
          index_t WPerThread,
          index_t EPerThreadLoop,
          index_t ThreadGemmADataPerRead_K,
          index_t ThreadGemmBDataPerRead_W>
struct BlockwiseGemm_km_kn_m0m1n0n1_v3
{
    struct MatrixIndex
    {
        index_t k;
        index_t h;
        index_t w;
    };

    index_t mMyThreadOffsetA;

    __device__ BlockwiseGemm_km_kn_m0m1n0n1_v3()
    {
        static_assert(BlockMatrixA::IsKnownAtCompileTime() &&
                          BlockMatrixB::IsKnownAtCompileTime() &&
                          ThreadMatrixC::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        static_assert(BlockMatrixA{}.GetLength(I0) == BlockMatrixB{}.GetLength(I0),
                      "wrong! K dimension not consistent\n");

        constexpr index_t K = BlockMatrixA{}.GetLength(I1); // A is transposed
        constexpr index_t N = BlockMatrixB{}.GetLength(I1);
        constexpr index_t H = BlockMatrixB{}.GetLength(I2);
        constexpr index_t W = BlockMatrixB{}.GetLength(I3);

        static_assert(K % KPerThread == 0 && H % HPerThread == 0 && W % WPerThread == 0,
                      "wrong! Cannot evenly divide work among\n");

        constexpr auto KThreadCluster = K / KPerThread;
        constexpr auto HThreadCluster = H / HPerThread;
        constexpr auto WThreadCluster = W / WPerThread;

        static_assert(BlockSize == KThreadCluster * HThreadCluster * WThreadCluster,
                      "wrong! wrong blocksize\n");

        auto c_thread_mtx_index = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        mMyThreadOffsetA =
            BlockMatrixA{}.CalculateOffset(make_tuple(0, c_thread_mtx_index.k * KPerThread));
    }

    __device__ static constexpr auto GetThreadMatrixCLengths()
    {
        return Sequence<KPerThread, 1, HPerThread, WPerThread>{};
    }

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t thread_id)
    {
        constexpr index_t H = BlockMatrixB{}.GetLength(Number<2>{});
        constexpr index_t W = BlockMatrixB{}.GetLength(Number<3>{});

        constexpr auto num_w_threads  = W / WPerThread;
        constexpr auto num_h_threads  = H / HPerThread;
        constexpr auto num_hw_threads = num_w_threads * num_h_threads;

        index_t k_thread_id  = thread_id / num_hw_threads;
        index_t hw_thread_id = thread_id % num_hw_threads;

        index_t h_thread_id = hw_thread_id / num_w_threads;
        index_t w_thread_id = hw_thread_id % num_w_threads;

        return MatrixIndex{k_thread_id, h_thread_id, w_thread_id};
    }

    template <typename SrcDesc,
              typename DstDesc,
              index_t NSliceRow,
              index_t NSliceCol,
              index_t DataPerAccess>
    struct ThreadwiseSliceCopy_a
    {
        template <typename Data>
        __device__ static void Run(const Data* p_src, Data* p_dst)
        {
            static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                          "wrong! Desc should be known at compile-time");

            using vector_t = typename vector_type_maker<Data, DataPerAccess>::type::type;

            static_for<0, NSliceRow, 1>{}([&](auto i) {
                static_for<0, NSliceCol, DataPerAccess>{}([&](auto j) {
                    constexpr auto src_offset = SrcDesc{}.CalculateOffset(make_tuple(i, j));
                    constexpr auto dst_offset = DstDesc{}.CalculateOffset(make_tuple(i, j));

                    *reinterpret_cast<vector_t*>(&p_dst[dst_offset]) =
                        *reinterpret_cast<const vector_t*>(&p_src[src_offset]);
                });
            });
        }
    };

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ void
    Run_naive(const FloatA* p_a_block, const FloatB* p_b_thread, FloatC* p_c_thread) const
    {

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto a_block_mtx = BlockMatrixA{};

        constexpr auto EPerBlock = a_block_mtx.GetLength(I0);

        constexpr auto KPerThreadSubC = 4;

        constexpr auto HoPerThreadSubC = 2;
        constexpr auto WoPerThreadSubC = 2;

        static_assert(KPerThread % KPerThreadSubC == 0, "");
        static_assert(HPerThread % HoPerThreadSubC == 0, "");
        static_assert(WPerThread % WoPerThreadSubC == 0, "");

        // thread A, B for GEMM
        constexpr auto a_thread_mtx = make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<EPerThreadLoop>{}, Number<KPerThreadSubC>{}));

        constexpr auto b_thread_mtx = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
            Number<EPerThreadLoop>{}, Number<1>{}, Number<HPerThread>{}, Number<WPerThread>{}));

        constexpr auto c_thread_mtx = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
            Number<KPerThreadSubC>{}, Number<1>{}, Number<HPerThread>{}, Number<WPerThread>{}));

        FloatA p_a_thread[a_thread_mtx.GetElementSpaceSize()];

        constexpr auto a_thread_copy = ThreadwiseSliceCopy_a<BlockMatrixA,
                                                             decltype(a_thread_mtx),
                                                             EPerThreadLoop,
                                                             KPerThreadSubC,
                                                             ThreadGemmADataPerRead_K>{};

        constexpr auto threadwise_gemm = ThreadwiseGemm_km_kn_mn_v3<decltype(a_thread_mtx),
                                                                    decltype(b_thread_mtx),
                                                                    decltype(c_thread_mtx),
                                                                    HoPerThreadSubC,
                                                                    WoPerThreadSubC>{};
        // loop over k
#pragma unroll
        for(index_t e_begin = 0; e_begin < EPerBlock; e_begin += EPerThreadLoop)
        {
#pragma unroll
            for(index_t k_begin = 0; k_begin < KPerThread; k_begin += KPerThreadSubC)
            {
                a_thread_copy.Run(p_a_block +
                                      a_block_mtx.CalculateOffset(make_tuple(e_begin, k_begin)) +
                                      mMyThreadOffsetA,
                                  p_a_thread);

#pragma unroll
                for(index_t h_begin = 0; h_begin < HPerThread; h_begin += HoPerThreadSubC)
                {
#pragma unroll
                    for(index_t w_begin = 0; w_begin < WPerThread; w_begin += WoPerThreadSubC)
                    {
                        threadwise_gemm.Run(p_a_thread,
                                            p_b_thread + b_thread_mtx.CalculateOffset(make_tuple(
                                                             e_begin, 0, h_begin, w_begin)),
                                            p_c_thread + c_thread_mtx.CalculateOffset(make_tuple(
                                                             k_begin, 0, h_begin, w_begin)));
                    }
                }
            }
        }
    }

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ void Run(const FloatA* p_a_block, const FloatB* p_b_thread, FloatC* p_c_thread) const
    {
        Run_naive(p_a_block, p_b_thread, p_c_thread);
    }
};
#else
// blockwise GEMM: C[M, N] += transpose(A[K, M]) * B[K, N]
// A and B are visable to the whole block, C is distributed among each thread
// If following number are power of 2, index calculation shall be greatly reduced:
//    KPerThread, HPerThread, MLevel0ThreadCluster, NLevel0ThreadCluster,
//    MLevel1ThreadCluster, NLevel1ThreadCluster
template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename BlockMatrixA,
          typename BlockMatrixB,
          typename ThreadMatrixC,
          index_t KPerThread,
          index_t HPerThread,
          index_t WPerThread,
          index_t EPerThreadLoop,
          index_t ThreadGemmADataPerRead_K,
          index_t ThreadGemmBDataPerRead_W>
struct BlockwiseGemm_km_kn_m0m1n0n1_v3
{
    struct MatrixIndex
    {
        index_t k;
        index_t h;
        index_t w;
    };

    MatrixIndex c_thread_begin_mtx_idx_;

    // HACK: fix this @Jing Zhang
    static constexpr index_t KPerThreadSubC = 4;

    static constexpr auto a_thread_mtx_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<EPerThreadLoop>{}, Number<KPerThreadSubC>{}));

    static constexpr auto b_thread_mtx_ = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
        Number<EPerThreadLoop>{}, Number<1>{}, Number<HPerThread>{}, Number<WPerThread>{}));

    static constexpr auto c_thread_mtx_ = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
        Number<KPerThreadSubC>{}, Number<1>{}, Number<HPerThread>{}, Number<WPerThread>{}));

    using AThreadCopy =
        ThreadwiseDynamicTensorSliceTransfer_v4<FloatA,
                                                FloatA,
                                                BlockMatrixA,
                                                decltype(a_thread_mtx_),
                                                Sequence<EPerThreadLoop, KPerThreadSubC>,
                                                Sequence<0, 1>,
                                                1,
                                                ThreadGemmADataPerRead_K,
                                                AddressSpace::Generic,
                                                AddressSpace::Vgpr,
                                                1>;

    AThreadCopy a_thread_copy_;

    __device__ BlockwiseGemm_km_kn_m0m1n0n1_v3()
        : c_thread_begin_mtx_idx_{GetBeginOfThreadMatrixC(get_thread_local_1d_id())},
          a_thread_copy_{make_tuple(0, c_thread_begin_mtx_idx_.k * KPerThread)}
    {
        static_assert(BlockMatrixA::IsKnownAtCompileTime() &&
                          BlockMatrixB::IsKnownAtCompileTime() &&
                          ThreadMatrixC::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        static_assert(BlockMatrixA{}.GetLength(I0) == BlockMatrixB{}.GetLength(I0),
                      "wrong! K dimension not consistent\n");

        constexpr index_t K = BlockMatrixA{}.GetLength(I1); // A is transposed
        constexpr index_t N = BlockMatrixB{}.GetLength(I1);
        constexpr index_t H = BlockMatrixB{}.GetLength(I2);
        constexpr index_t W = BlockMatrixB{}.GetLength(I3);

        static_assert(K % KPerThread == 0 && H % HPerThread == 0 && W % WPerThread == 0,
                      "wrong! Cannot evenly divide work among\n");

        constexpr auto KThreadCluster = K / KPerThread;
        constexpr auto HThreadCluster = H / HPerThread;
        constexpr auto WThreadCluster = W / WPerThread;

        static_assert(BlockSize == KThreadCluster * HThreadCluster * WThreadCluster,
                      "wrong! wrong blocksize\n");
    }

    __device__ static constexpr auto GetThreadMatrixCLengths()
    {
        return Sequence<KPerThread, 1, HPerThread, WPerThread>{};
    }

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t thread_id)
    {
        constexpr index_t H = BlockMatrixB{}.GetLength(Number<2>{});
        constexpr index_t W = BlockMatrixB{}.GetLength(Number<3>{});

        constexpr auto num_w_threads  = W / WPerThread;
        constexpr auto num_h_threads  = H / HPerThread;
        constexpr auto num_hw_threads = num_w_threads * num_h_threads;

        index_t k_thread_id  = thread_id / num_hw_threads;
        index_t hw_thread_id = thread_id % num_hw_threads;

        index_t h_thread_id = hw_thread_id / num_w_threads;
        index_t w_thread_id = hw_thread_id % num_w_threads;

        return MatrixIndex{k_thread_id, h_thread_id, w_thread_id};
    }

    __device__ void Run(const FloatA* p_a_block, const FloatB* p_b_thread, FloatC* p_c_thread) const
    {
        auto a_block_buf = make_dynamic_buffer(p_a_block);

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto a_block_mtx = BlockMatrixA{};

        constexpr auto EPerBlock = a_block_mtx.GetLength(I0);

        // HACK: fix this @Jing Zhang
        constexpr auto HoPerThreadSubC = 2;
        constexpr auto WoPerThreadSubC = 2;

        static_assert(KPerThread % KPerThreadSubC == 0, "");
        static_assert(HPerThread % HoPerThreadSubC == 0, "");
        static_assert(WPerThread % WoPerThreadSubC == 0, "");

        // thread A, B for GEMM
        FloatA p_a_thread[a_thread_mtx_.GetElementSpaceSize()];

        auto a_thread_buf = make_dynamic_buffer(p_a_thread);
        auto b_thread_buf = make_dynamic_buffer(p_b_thread);
        auto c_thread_buf = make_dynamic_buffer(p_c_thread);

        constexpr auto threadwise_gemm = ThreadwiseGemm_km_kn_mn_v3<FloatA,
                                                                    FloatB,
                                                                    FloatC,
                                                                    decltype(a_thread_mtx_),
                                                                    decltype(b_thread_mtx_),
                                                                    decltype(c_thread_mtx_),
                                                                    HoPerThreadSubC,
                                                                    WoPerThreadSubC>{};

        static_for<0, EPerBlock, EPerThreadLoop>{}([&](auto e_begin) {
            static_for<0, KPerThread, KPerThreadSubC>{}([&](auto k_begin) {

                a_thread_copy_.Run(a_block_mtx,
                                   make_tuple(e_begin, k_begin),
                                   a_block_buf,
                                   a_thread_mtx_,
                                   make_tuple(I0, I0),
                                   a_thread_buf);

                static_for<0, HPerThread, HoPerThreadSubC>{}([&](auto h_begin) {
                    static_for<0, WPerThread, WoPerThreadSubC>{}([&](auto w_begin) {
                        threadwise_gemm.Run(a_thread_buf,
                                            make_tuple(I0, I0),
                                            b_thread_buf,
                                            make_tuple(e_begin, I0, h_begin, w_begin),
                                            c_thread_buf,
                                            make_tuple(k_begin, I0, h_begin, w_begin));
                    });
                });
            });
        });
    }
};
#endif

} // namespace ck
#endif
