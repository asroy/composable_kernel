#ifndef CK_BLOCKWISE_GEMM_V3_HPP
#define CK_BLOCKWISE_GEMM_V3_HPP

#include "common_header.hpp"
#include "threadwise_gemm_v3.hpp"

namespace ck {

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

    using ThreadLengthsA = Sequence<EPerThreadLoop, KPerThread>;
    using ThreadLengthsB = Sequence<EPerThreadLoop, 1, HPerThread, WPerThread>;

    // using BlockLengthsA = Sequence<EPerBlock, KPerBlock>;
    // using BlockLengthsB = Sequence<EPerBlock, 1, HPerBlock, WPerBlock>;

    using BlockLengthsA = Sequence<1, 16>;
    using BlockLengthsB = Sequence<1, 1, 8, 32>;

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

        constexpr auto c_thread_lengths = GetCThreadLengths();
        constexpr auto c_block_lengths  = GetCBlockLengths();
        static_assert(c_thread_lengths.Size() == c_block_lengths.Size(), "");

        constexpr auto num_threads = generate_sequence_v2(
            [&](auto i) {
                static_assert(c_block_lengths[i] % c_thread_lengths[i] == 0, "");
                return Number<c_block_lengths[i] / c_thread_lengths[i]>{};
            },
            Number<c_thread_lengths.Size()>{});

        constexpr index_t total_num_threads =
            reduce_on_sequence(num_threads, math::multiplies<index_t>{}, Number<1>{});

        static_assert(BlockSize == total_num_threads, "wrong! wrong blocksize\n");

        auto c_thread_mtx_index = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        mMyThreadOffsetA =
            BlockMatrixA{}.CalculateOffset(make_tuple(0, c_thread_mtx_index.k * KPerThread));
    }

    __device__ static constexpr auto GetCThreadLengths()
    {
        constexpr auto a_lengths_sub_size = ThreadLengthsA{}.Size() - 1;
        constexpr auto b_lengths_sub_size = ThreadLengthsB{}.Size() - 1;

        constexpr auto c_thread_lengths = generate_sequence_v2(
            [&](auto i) {
                if constexpr(i >= a_lengths_sub_size)
                {
                    return Number<ThreadLengthsB{}[i - a_lengths_sub_size + 1]>{};
                }
                else
                {
                    return Number<ThreadLengthsA{}[i + 1]>{};
                }
            },
            Number<a_lengths_sub_size + b_lengths_sub_size>{});

        return c_thread_lengths;
    }

    __device__ static constexpr auto GetCBlockLengths()
    {
        constexpr auto a_lengths_sub_size = BlockLengthsA{}.Size() - 1;
        constexpr auto b_lengths_sub_size = BlockLengthsB{}.Size() - 1;

        constexpr auto c_block_lengths = generate_sequence_v2(
            [&](auto i) {
                if constexpr(i >= a_lengths_sub_size)
                {
                    return Number<BlockLengthsB{}[i - a_lengths_sub_size + 1]>{};
                }
                else
                {
                    return Number<BlockLengthsA{}[i + 1]>{};
                }
            },
            Number<a_lengths_sub_size + b_lengths_sub_size>{});

        return c_block_lengths;
    }

    __device__ static auto GetBeginOfThreadMatrixC(index_t thread_id)
    {
        constexpr auto c_thread_lengths = GetCThreadLengths();
        constexpr auto c_block_lengths  = GetCBlockLengths();

        constexpr auto num_threads = generate_tuple(
            [&](auto i) { return Number<c_block_lengths[i] / c_thread_lengths[i]>{}; },
            Number<c_thread_lengths.Size()>{});

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

    template <typename seq>
    __device__ static constexpr auto seq_to_tuple()
    {
        return generate_tuple([&](auto i) { return seq{}[i]; }, seq{}.Size());
    }

    template <typename SrcDesc, typename DstDesc, index_t DataPerAccess>
    struct ThreadwiseSliceCopy_a
    {
        template <typename Data>
        __device__ static void Run(const Data* p_src, Data* p_dst)
        {
            static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                          "wrong! Desc should be known at compile-time");

            using vector_t = typename vector_type_maker<Data, DataPerAccess>::type::type;

            constexpr auto a_lengths = ThreadLengthsA{};

            constexpr auto a_lengths_vec = generate_sequence_v2(
                [&](auto i) {
                    if constexpr(i == a_lengths.Size() - 1)
                    {
                        return Number<a_lengths[i] / DataPerAccess>{};
                    }
                    else
                    {
                        return Number<a_lengths[i]>{};
                    }
                },
                Number<a_lengths.Size()>{});

            static_ford<decltype(a_lengths_vec)>{}([&](auto a_idx_vec) {
                constexpr auto a_idx = generate_tuple([&](auto i) { return a_idx_vec[i]; },
                                                      Number<a_idx_vec.Size()>{});

                constexpr auto src_offset = SrcDesc{}.CalculateOffset(a_idx);
                constexpr auto dst_offset = DstDesc{}.CalculateOffset(a_idx);

                *reinterpret_cast<vector_t*>(&p_dst[dst_offset]) =
                    *reinterpret_cast<const vector_t*>(&p_src[src_offset]);
            });
        }
    };

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ void
    Run_naive(const FloatA* p_a_block, const FloatB* p_b_thread, FloatC* p_c_thread) const
    {

        constexpr auto I0 = Number<0>{};

        constexpr auto a_block_mtx = BlockMatrixA{};

        constexpr auto EPerBlock = a_block_mtx.GetLength(I0);

        using ThreadLengthsC = decltype(GetCThreadLengths());

        // thread A, B for GEMM
        constexpr auto a_thread_mtx =
            make_dynamic_naive_tensor_descriptor_packed_v2(seq_to_tuple<ThreadLengthsA>());

        constexpr auto b_thread_mtx =
            make_dynamic_naive_tensor_descriptor_packed_v2(seq_to_tuple<ThreadLengthsB>());

        constexpr auto c_thread_mtx =
            make_dynamic_naive_tensor_descriptor_packed_v2(seq_to_tuple<ThreadLengthsC>());

        FloatA p_a_thread[a_thread_mtx.GetElementSpaceSize()];

        constexpr auto a_thread_copy =
            ThreadwiseSliceCopy_a<BlockMatrixA, decltype(a_thread_mtx), ThreadGemmADataPerRead_K>{};

        // constexpr index_t AVectorDim = 1, AVectorSize = 1, BVectorDim = 2, BVectorSize = 4;
        // constexpr index_t AVectorDim = 1, AVectorSize = 1, BVectorDim = 2, BVectorSize = 2;
        // constexpr index_t AVectorDim = 1, AVectorSize = 2, BVectorDim = 2, BVectorSize = 2;
        constexpr index_t AVectorDim = 1, AVectorSize = 2, BVectorDim = 2, BVectorSize = 2;

        constexpr auto threadwise_gemm = ThreadwiseGemm_km_kn_mn_v3<decltype(a_thread_mtx),
                                                                    decltype(b_thread_mtx),
                                                                    decltype(c_thread_mtx),
                                                                    ThreadLengthsA,
                                                                    ThreadLengthsB,
                                                                    ThreadLengthsC,
                                                                    AVectorDim,
                                                                    AVectorSize,
                                                                    BVectorDim,
                                                                    BVectorSize>{};
        // loop over k
#pragma unroll
        for(index_t e_begin = 0; e_begin < EPerBlock; e_begin += EPerThreadLoop)
        {
            a_thread_copy.Run(p_a_block + a_block_mtx.CalculateOffset(make_tuple(e_begin, 0)) +
                                  mMyThreadOffsetA,
                              p_a_thread);

            threadwise_gemm.Run(p_a_thread,
                                p_b_thread +
                                    b_thread_mtx.CalculateOffset(make_tuple(e_begin, 0, 0, 0)),
                                p_c_thread + c_thread_mtx.CalculateOffset(make_tuple(0, 0, 0, 0)));
        }
    }

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ void Run(const FloatA* p_a_block, const FloatB* p_b_thread, FloatC* p_c_thread) const
    {
        Run_naive(p_a_block, p_b_thread, p_c_thread);
    }
}; // namespace ck

} // namespace ck
#endif
