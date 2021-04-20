#ifndef CK_BLOCKWISE_GEMM_V2_HPP
#define CK_BLOCKWISE_GEMM_V2_HPP

#include "common_header.hpp"
#include "threadwise_gemm_v2.hpp"

namespace ck {

// blockwise GEMM: C[M, N] += transpose(A[K, M]) * B[K, N]
// A and B are visable to the whole block, C is distributed among each thread
// If following number are power of 2, index calculation shall be greatly reduced:
//    MPerThreadSubC, NPerThreadSubC, MLevel0ThreadCluster, NLevel0ThreadCluster,
//    MLevel1ThreadCluster, NLevel1ThreadCluster
template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename BlockMatrixA,
          typename BlockMatrixB,
          typename ThreadMatrixC,
          index_t MPerThreadSubC,
          index_t NPerThreadSubC,
          index_t KPerThreadLoop,
          index_t MLevel0ThreadCluster,
          index_t NLevel0ThreadCluster,
          index_t MLevel1ThreadCluster,
          index_t NLevel1ThreadCluster,
          index_t ThreadGemmADataPerRead_M,
          index_t ThreadGemmBDataPerRead_N>
struct BlockwiseGemm_km_kn_m0m1n0n1_v1
{
    struct MatrixIndex
    {
        index_t row;
        index_t col;
    };

    index_t mMyThreadOffsetA;
    index_t mMyThreadOffsetB;

    __device__ BlockwiseGemm_km_kn_m0m1n0n1_v1()
    {
        static_assert(BlockMatrixA::IsKnownAtCompileTime() &&
                          BlockMatrixB::IsKnownAtCompileTime() &&
                          ThreadMatrixC::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr index_t ThreadPerLevel1Cluster = MLevel0ThreadCluster * NLevel0ThreadCluster *
                                                   MLevel1ThreadCluster * NLevel1ThreadCluster;

        static_assert(BlockSize == ThreadPerLevel1Cluster, "wrong! wrong blocksize\n");

        static_assert(BlockMatrixA{}.GetLength(I0) == BlockMatrixB{}.GetLength(I0),
                      "wrong! K dimension not consistent\n");

        constexpr index_t M = BlockMatrixA{}.GetLength(I1); // A is transposed
        constexpr index_t N = BlockMatrixB{}.GetLength(I1);

        static_assert(M % (MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster) == 0 &&
                          N % (NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster) == 0,
                      "wrong! Cannot evenly divide work among\n");

        static_assert(ThreadMatrixC{}.GetLength(I0) == GetThreadMatrixCLengths()[I0] &&
                          ThreadMatrixC{}.GetLength(I1) == GetThreadMatrixCLengths()[I1],
                      "wrong! ThreadMatrixC lengths is wrong");

        auto c_thread_mtx_index = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        mMyThreadOffsetA = BlockMatrixA{}.CalculateOffset(make_tuple(0, c_thread_mtx_index.row));
        mMyThreadOffsetB = BlockMatrixB{}.CalculateOffset(make_tuple(0, c_thread_mtx_index.col));
    }

    __device__ static constexpr auto GetThreadMatrixCLengths()
    {
        constexpr auto I1 = Number<1>{};

        constexpr index_t M = BlockMatrixA{}.GetLength(I1); // A is transposed
        constexpr index_t N = BlockMatrixB{}.GetLength(I1);

        constexpr index_t MRepeat =
            M / (MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster);
        constexpr index_t NRepeat =
            N / (NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster);

        return Sequence<MRepeat * MPerThreadSubC, NRepeat * NPerThreadSubC>{};
    }

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t thread_id)
    {
        constexpr index_t ThreadPerLevel0Cluster = MLevel0ThreadCluster * NLevel0ThreadCluster;

        index_t level1_id   = thread_id / ThreadPerLevel0Cluster;
        index_t level1_m_id = level1_id / NLevel1ThreadCluster;
        index_t level1_n_id = level1_id % NLevel1ThreadCluster;

        index_t level0_id   = thread_id % ThreadPerLevel0Cluster;
        index_t level0_m_id = level0_id / NLevel0ThreadCluster;
        index_t level0_n_id = level0_id % NLevel0ThreadCluster;

        constexpr index_t MPerLevel0Cluster = MPerThreadSubC * MLevel0ThreadCluster;
        constexpr index_t NPerLevel0Cluster = NPerThreadSubC * NLevel0ThreadCluster;

        return MatrixIndex{level1_m_id * MPerLevel0Cluster + level0_m_id * MPerThreadSubC,
                           level1_n_id * NPerLevel0Cluster + level0_n_id * NPerThreadSubC};
    }

    __device__ void
    Run_naive(const FloatA* p_a_block, const FloatB* p_b_block, FloatC* p_c_thread) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr auto K = a_block_mtx.GetLength(I0);

        constexpr auto MPerThread = c_thread_mtx.GetLength(I0);
        constexpr auto NPerThread = c_thread_mtx.GetLength(I1);

        constexpr index_t MPerLevel1Cluster =
            MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster;
        constexpr index_t NPerLevel1Cluster =
            NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        // thread A, B for GEMM
        constexpr auto a_thread_mtx = make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<KPerThreadLoop>{}, Number<MPerThread>{}));

        constexpr auto b_thread_mtx = make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<KPerThreadLoop>{}, Number<NPerThread>{}));

        FloatA p_a_thread[a_thread_mtx.GetElementSpaceSize()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpaceSize()];

        constexpr auto a_thread_copy = ThreadwiseMatrixSliceCopy_v2<BlockMatrixA,
                                                                    decltype(a_thread_mtx),
                                                                    KPerThreadLoop,
                                                                    MPerThreadSubC,
                                                                    ThreadGemmADataPerRead_M>{};

        constexpr auto b_thread_copy = ThreadwiseMatrixSliceCopy_v2<BlockMatrixB,
                                                                    decltype(b_thread_mtx),
                                                                    KPerThreadLoop,
                                                                    NPerThreadSubC,
                                                                    ThreadGemmBDataPerRead_N>{};

        constexpr auto threadwise_gemm = ThreadwiseGemm_km_kn_mn_v1<FloatA,
                                                                    FloatB,
                                                                    FloatC,
                                                                    decltype(a_thread_mtx),
                                                                    decltype(b_thread_mtx),
                                                                    decltype(c_thread_mtx)>{};
        // loop over k
        static_for<0, K, KPerThreadLoop>{}([&](auto k_begin) {
            // read A
            static_for<0, MRepeat, 1>{}([&](auto m_repeat) {
                a_thread_copy.Run(p_a_block +
                                      a_block_mtx.CalculateOffset(
                                          make_tuple(k_begin, m_repeat * MPerLevel1Cluster)) +
                                      mMyThreadOffsetA,
                                  p_a_thread + a_thread_mtx.CalculateOffset(
                                                   make_tuple(0, m_repeat * MPerThreadSubC)));
            });

            // read B
            static_for<0, NRepeat, 1>{}([&](auto n_repeat) {
                b_thread_copy.Run(p_b_block +
                                      b_block_mtx.CalculateOffset(
                                          make_tuple(k_begin, n_repeat * NPerLevel1Cluster)) +
                                      mMyThreadOffsetB,
                                  p_b_thread + b_thread_mtx.CalculateOffset(
                                                   make_tuple(0, n_repeat * NPerThreadSubC)));
            });

            // C += A * B
            threadwise_gemm.Run(p_a_thread, p_b_thread, p_c_thread);
        });
    }

    __device__ void
    Run_pipelined_2x2(const FloatA* p_a_block, const FloatB* p_b_block, FloatC* p_c_thread) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr auto K = a_block_mtx.GetLength(I0);

        constexpr auto MPerThread = c_thread_mtx.GetLength(I0);
        constexpr auto NPerThread = c_thread_mtx.GetLength(I1);

        constexpr index_t MPerLevel1Cluster =
            MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster;

        constexpr index_t NPerLevel1Cluster =
            NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        static_assert(MRepeat == 2 && NRepeat == 2,
                      "wrong! inline asm cannot deal with this GEMM config yet");

        // thread A, B
        constexpr auto a_thread_mtx = make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<KPerThreadLoop>{}, Number<MPerThread>{}));

        constexpr auto b_thread_mtx = make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<KPerThreadLoop>{}, Number<NPerThread>{}));

        // thread A-sub, B-sub
        constexpr auto a_thread_sub_mtx = make_dynamic_naive_tensor_descriptor_v2(
            make_tuple(Number<KPerThreadLoop>{}, Number<MPerThreadSubC>{}),
            make_tuple(Number<MPerThread>{}, Number<1>{}));

        constexpr auto b_thread_sub_mtx = make_dynamic_naive_tensor_descriptor_v2(
            make_tuple(Number<KPerThreadLoop>{}, Number<NPerThreadSubC>{}),
            make_tuple(Number<NPerThread>{}, Number<1>{}));

        constexpr auto c_thread_sub_mtx = make_dynamic_naive_tensor_descriptor_v2(
            make_tuple(Number<MPerThreadSubC>{}, Number<NPerThreadSubC>{}),
            make_tuple(Number<NPerThread>{}, Number<1>{}));

        FloatA p_a_thread[a_thread_mtx.GetElementSpaceSize()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpaceSize()];

        constexpr auto a_thread_copy = ThreadwiseMatrixSliceCopy_v2<BlockMatrixA,
                                                                    decltype(a_thread_mtx),
                                                                    KPerThreadLoop,
                                                                    MPerThreadSubC,
                                                                    ThreadGemmADataPerRead_M>{};

        constexpr auto b_thread_copy = ThreadwiseMatrixSliceCopy_v2<BlockMatrixB,
                                                                    decltype(b_thread_mtx),
                                                                    KPerThreadLoop,
                                                                    NPerThreadSubC,
                                                                    ThreadGemmBDataPerRead_N>{};

        constexpr auto threadwise_gemm = ThreadwiseGemm_km_kn_mn_v1<FloatA,
                                                                    FloatB,
                                                                    FloatC,
                                                                    decltype(a_thread_sub_mtx),
                                                                    decltype(b_thread_sub_mtx),
                                                                    decltype(c_thread_sub_mtx)>{};

        const FloatA* p_a_block_off = p_a_block + mMyThreadOffsetA;
        const FloatB* p_b_block_off = p_b_block + mMyThreadOffsetB;

        // read A_sub_0
        a_thread_copy.Run(p_a_block_off, p_a_thread);

        // read B_sub_0
        b_thread_copy.Run(p_b_block_off, p_b_thread);

        // read B_sub_1
        b_thread_copy.Run(p_b_block_off +
                              b_block_mtx.CalculateOffset(make_tuple(0, NPerLevel1Cluster)),
                          p_b_thread + b_thread_mtx.CalculateOffset(make_tuple(0, NPerThreadSubC)));

        // read A_sub_1
        a_thread_copy.Run(p_a_block_off +
                              a_block_mtx.CalculateOffset(make_tuple(0, MPerLevel1Cluster)),
                          p_a_thread + a_thread_mtx.CalculateOffset(make_tuple(0, MPerThreadSubC)));

        // C_sub_00 += transpose(A_sub_0) * B_sub_0
        threadwise_gemm.Run(p_a_thread, p_b_thread, p_c_thread);

        // C_sub_01 += transpose(A_sub_0) * B_sub_1
        threadwise_gemm.Run(
            p_a_thread,
            p_b_thread + b_thread_mtx.CalculateOffset(make_tuple(0, NPerThreadSubC)),
            p_c_thread + c_thread_mtx.CalculateOffset(make_tuple(0, NPerThreadSubC)));

        // loop over rest of k
        static_for<KPerThreadLoop, K, KPerThreadLoop>{}([&](auto k) {
            // read A_sub_0
            a_thread_copy.Run(p_a_block_off + a_block_mtx.CalculateOffset(make_tuple(k, 0)),
                              p_a_thread);

            // C_sub_10 += transpose(A_sub_1) * B_sub_0
            threadwise_gemm.Run(
                p_a_thread + a_thread_mtx.CalculateOffset(make_tuple(0, MPerThreadSubC)),
                p_b_thread,
                p_c_thread + c_thread_mtx.CalculateOffset(make_tuple(MPerThreadSubC, 0)));

            // read B_sub_0
            b_thread_copy.Run(p_b_block_off + b_block_mtx.CalculateOffset(make_tuple(k, 0)),
                              p_b_thread);

            // C_sub_11 += transpose(A_sub_1) * B_sub_1
            threadwise_gemm.Run(
                p_a_thread + a_thread_mtx.CalculateOffset(make_tuple(0, MPerThreadSubC)),
                p_b_thread + b_thread_mtx.CalculateOffset(make_tuple(0, NPerThreadSubC)),
                p_c_thread +
                    c_thread_mtx.CalculateOffset(make_tuple(MPerThreadSubC, NPerThreadSubC)));

            // read B_sub_1
            b_thread_copy.Run(
                p_b_block_off + b_block_mtx.CalculateOffset(make_tuple(k, NPerLevel1Cluster)),
                p_b_thread + b_thread_mtx.CalculateOffset(make_tuple(0, NPerThreadSubC)));

            // read A_sub_1
            a_thread_copy.Run(
                p_a_block_off + a_block_mtx.CalculateOffset(make_tuple(k, MPerLevel1Cluster)),
                p_a_thread + a_thread_mtx.CalculateOffset(make_tuple(0, MPerThreadSubC)));

            // C_sub_00 += transpose(A_sub_0) * B_sub_0
            threadwise_gemm.Run(p_a_thread, p_b_thread, p_c_thread);

            // C_sub_01 += transpose(A_sub_0) * B_sub_1
            threadwise_gemm.Run(
                p_a_thread,
                p_b_thread + b_thread_mtx.CalculateOffset(make_tuple(0, NPerThreadSubC)),
                p_c_thread + c_thread_mtx.CalculateOffset(make_tuple(0, NPerThreadSubC)));
        });

        // C_sub_10 += transpose(A_sub_1) * B_sub_0
        threadwise_gemm.Run(
            p_a_thread + a_thread_mtx.CalculateOffset(make_tuple(0, MPerThreadSubC)),
            p_b_thread,
            p_c_thread + c_thread_mtx.CalculateOffset(make_tuple(MPerThreadSubC, 0)));

        // C_sub_11 += transpose(A_sub_1) * B_sub_1
        threadwise_gemm.Run(
            p_a_thread + a_thread_mtx.CalculateOffset(make_tuple(0, MPerThreadSubC)),
            p_b_thread + b_thread_mtx.CalculateOffset(make_tuple(0, NPerThreadSubC)),
            p_c_thread + c_thread_mtx.CalculateOffset(make_tuple(MPerThreadSubC, NPerThreadSubC)));
    }

    __device__ void Run(const FloatA* p_a_block, const FloatB* p_b_block, FloatC* p_c_thread) const
    {
#if CK_EXPERIMENTAL_BLOCKWISE_GEMM_USE_PIPELINE
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr index_t MPerThread = ThreadMatrixC{}.GetLength(I0);
        constexpr index_t NPerThread = ThreadMatrixC{}.GetLength(I1);

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        if constexpr(MRepeat == 2 && NRepeat == 2)
        {
            Run_pipelined_2x2(p_a_block, p_b_block, p_c_thread);
        }
        else
        {
            Run_naive(p_a_block, p_b_block, p_c_thread);
        }
#else
        Run_naive(p_a_block, p_b_block, p_c_thread);
#endif
    }
};

// blockwise GEMM: C[M, N] += transpose(A[K, M]) * B[K, N]
// A and B are visable to the whole block, C is distributed among each thread
// If following number are power of 2, index calculation shall be greatly reduced:
//    MPerThreadSubC, NPerThreadSubC, MLevel0ThreadCluster, NLevel0ThreadCluster,
//    MLevel1ThreadCluster, NLevel1ThreadCluster
template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename BlockMatrixA,
          typename BlockMatrixB,
          typename ThreadMatrixC,
          index_t MPerThreadSubC,
          index_t NPerThreadSubC,
          index_t KPerThreadLoop,
          index_t MLevel0ThreadCluster,
          index_t NLevel0ThreadCluster,
          index_t MLevel1ThreadCluster,
          index_t NLevel1ThreadCluster,
          index_t ThreadGemmADataPerRead_M,
          index_t ThreadGemmBDataPerRead_N,
          typename std::enable_if<BlockMatrixA::IsKnownAtCompileTime() &&
                                      BlockMatrixB::IsKnownAtCompileTime() &&
                                      ThreadMatrixC::IsKnownAtCompileTime(),
                                  bool>::type = false>
struct BlockwiseGemm_km_kn_m0m1n0n1_v1r1
{
    struct MatrixIndex
    {
        index_t row;
        index_t col;
    };

    private:
    static constexpr auto a_thread_mtx_desc_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<KPerThreadLoop>{}, ThreadMatrixC{}.GetLength(Number<0>{})));

    static constexpr auto b_thread_mtx_desc_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<KPerThreadLoop>{}, ThreadMatrixC{}.GetLength(Number<1>{})));

    using AThreadCopy =
        ThreadwiseDynamicTensorSliceTransfer_v4<FloatA,
                                                FloatA,
                                                BlockMatrixA,
                                                decltype(a_thread_mtx_desc_),
                                                Sequence<KPerThreadLoop, MPerThreadSubC>,
                                                Sequence<0, 1>,
                                                1,
                                                ThreadGemmADataPerRead_M,
                                                AddressSpace::Generic,
                                                AddressSpace::Vgpr,
                                                1>;

    using BThreadCopy =
        ThreadwiseDynamicTensorSliceTransfer_v4<FloatB,
                                                FloatB,
                                                BlockMatrixB,
                                                decltype(b_thread_mtx_desc_),
                                                Sequence<KPerThreadLoop, NPerThreadSubC>,
                                                Sequence<0, 1>,
                                                1,
                                                ThreadGemmBDataPerRead_N,
                                                AddressSpace::Generic,
                                                AddressSpace::Vgpr,
                                                1>;

    MatrixIndex c_thread_begin_mtx_idx_;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;

    public:
    __device__ BlockwiseGemm_km_kn_m0m1n0n1_v1r1()
        : c_thread_begin_mtx_idx_{GetBeginOfThreadMatrixC(get_thread_local_1d_id())},
          a_thread_copy_{make_tuple(0, c_thread_begin_mtx_idx_.row)},
          b_thread_copy_{make_tuple(0, c_thread_begin_mtx_idx_.col)}
    {
        static_assert(BlockMatrixA::IsKnownAtCompileTime() &&
                          BlockMatrixB::IsKnownAtCompileTime() &&
                          ThreadMatrixC::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr index_t ThreadPerLevel1Cluster = MLevel0ThreadCluster * NLevel0ThreadCluster *
                                                   MLevel1ThreadCluster * NLevel1ThreadCluster;

        static_assert(BlockSize == ThreadPerLevel1Cluster, "wrong! wrong blocksize\n");

        static_assert(BlockMatrixA{}.GetLength(I0) == BlockMatrixB{}.GetLength(I0),
                      "wrong! K dimension not consistent");

        constexpr index_t M = BlockMatrixA{}.GetLength(I1); // A is transposed
        constexpr index_t N = BlockMatrixB{}.GetLength(I1);

        static_assert(M % (MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster) == 0 &&
                          N % (NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster) == 0,
                      "wrong! Cannot evenly divide work among");

        static_assert(ThreadMatrixC{}.GetLength(I0) == GetThreadMatrixCLengths()[I0] &&
                          ThreadMatrixC{}.GetLength(I1) == GetThreadMatrixCLengths()[I1],
                      "wrong! ThreadMatrixC lengths is wrong");
    }

    __device__ static constexpr auto GetThreadMatrixCLengths()
    {
        constexpr auto I1 = Number<1>{};

        constexpr index_t M = BlockMatrixA{}.GetLength(I1); // A is transposed
        constexpr index_t N = BlockMatrixB{}.GetLength(I1);

        constexpr index_t MRepeat =
            M / (MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster);
        constexpr index_t NRepeat =
            N / (NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster);

        return Sequence<MRepeat * MPerThreadSubC, NRepeat * NPerThreadSubC>{};
    }

    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t thread_id)
    {
        constexpr index_t ThreadPerLevel0Cluster = MLevel0ThreadCluster * NLevel0ThreadCluster;

        index_t level1_id   = thread_id / ThreadPerLevel0Cluster;
        index_t level1_m_id = level1_id / NLevel1ThreadCluster;
        index_t level1_n_id = level1_id % NLevel1ThreadCluster;

        index_t level0_id   = thread_id % ThreadPerLevel0Cluster;
        index_t level0_m_id = level0_id / NLevel0ThreadCluster;
        index_t level0_n_id = level0_id % NLevel0ThreadCluster;

        constexpr index_t MPerLevel0Cluster = MPerThreadSubC * MLevel0ThreadCluster;
        constexpr index_t NPerLevel0Cluster = NPerThreadSubC * NLevel0ThreadCluster;

        return MatrixIndex{level1_m_id * MPerLevel0Cluster + level0_m_id * MPerThreadSubC,
                           level1_n_id * NPerLevel0Cluster + level0_n_id * NPerThreadSubC};
    }

    __device__ void
    Run_pipelined_2x2(const FloatA* p_a_block, const FloatB* p_b_block, FloatC* p_c_thread) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr auto a_block_mtx       = BlockMatrixA{};
        constexpr auto b_block_mtx       = BlockMatrixB{};
        constexpr auto c_thread_mtx_desc = ThreadMatrixC{};

        constexpr auto K = a_block_mtx.GetLength(I0);

        constexpr auto MPerThread = c_thread_mtx_desc.GetLength(I0);
        constexpr auto NPerThread = c_thread_mtx_desc.GetLength(I1);

        constexpr index_t MPerLevel1Cluster =
            MPerThreadSubC * MLevel0ThreadCluster * MLevel1ThreadCluster;

        constexpr index_t NPerLevel1Cluster =
            NPerThreadSubC * NLevel0ThreadCluster * NLevel1ThreadCluster;

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        static_assert(MRepeat == 2 && NRepeat == 2,
                      "wrong! inline asm cannot deal with this GEMM config yet");

        // thread A-sub, B-sub
        constexpr auto a_thread_sub_mtx = make_dynamic_naive_tensor_descriptor_v2(
            make_tuple(Number<KPerThreadLoop>{}, Number<MPerThreadSubC>{}),
            make_tuple(Number<MPerThread>{}, Number<1>{}));

        constexpr auto b_thread_sub_mtx = make_dynamic_naive_tensor_descriptor_v2(
            make_tuple(Number<KPerThreadLoop>{}, Number<NPerThreadSubC>{}),
            make_tuple(Number<NPerThread>{}, Number<1>{}));

        constexpr auto c_thread_sub_mtx = make_dynamic_naive_tensor_descriptor_v2(
            make_tuple(Number<MPerThreadSubC>{}, Number<NPerThreadSubC>{}),
            make_tuple(Number<NPerThread>{}, Number<1>{}));

        FloatA p_a_thread[a_thread_mtx_desc_.GetElementSpaceSize()];
        FloatB p_b_thread[b_thread_mtx_desc_.GetElementSpaceSize()];

        constexpr auto threadwise_gemm = ThreadwiseGemm_km_kn_mn_v1<FloatA,
                                                                    FloatB,
                                                                    FloatC,
                                                                    decltype(a_thread_sub_mtx),
                                                                    decltype(b_thread_sub_mtx),
                                                                    decltype(c_thread_sub_mtx)>{};

        // read A_sub_0
        a_thread_copy_.Run(BlockMatrixA{},
                           make_tuple(Number<0>{}, Number<0>{}),
                           p_a_block,
                           a_thread_mtx_desc_,
                           make_tuple(Number<0>{}, Number<0>{}),
                           p_a_thread);

        // read B_sub_0
        b_thread_copy_.Run(BlockMatrixB{},
                           make_tuple(Number<0>{}, Number<0>{}),
                           p_b_block,
                           b_thread_mtx_desc_,
                           make_tuple(Number<0>{}, Number<0>{}),
                           p_b_thread);

        // read B_sub_1
        b_thread_copy_.Run(BlockMatrixB{},
                           make_tuple(Number<0>{}, Number<NPerLevel1Cluster>{}),
                           p_b_block,
                           b_thread_mtx_desc_,
                           make_tuple(Number<0>{}, Number<NPerThreadSubC>{}),
                           p_b_thread);

        // read A_sub_1
        a_thread_copy_.Run(BlockMatrixA{},
                           make_tuple(Number<0>{}, Number<MPerLevel1Cluster>{}),
                           p_a_block,
                           a_thread_mtx_desc_,
                           make_tuple(Number<0>{}, Number<MPerThreadSubC>{}),
                           p_a_thread);

        // C_sub_00 += transpose(A_sub_0) * B_sub_0
        threadwise_gemm.Run(p_a_thread, p_b_thread, p_c_thread);

        // C_sub_01 += transpose(A_sub_0) * B_sub_1
        threadwise_gemm.Run(
            p_a_thread,
            p_b_thread + b_thread_mtx_desc_.CalculateOffset(make_tuple(0, NPerThreadSubC)),
            p_c_thread + c_thread_mtx_desc.CalculateOffset(make_tuple(0, NPerThreadSubC)));

        // loop over rest of k
        static_for<KPerThreadLoop, K, KPerThreadLoop>{}([&](auto k) {
            // read A_sub_0
            a_thread_copy_.Run(BlockMatrixA{},
                               make_tuple(k, Number<0>{}),
                               p_a_block,
                               a_thread_mtx_desc_,
                               make_tuple(Number<0>{}, Number<0>{}),
                               p_a_thread);

            // C_sub_10 += transpose(A_sub_1) * B_sub_0
            threadwise_gemm.Run(
                p_a_thread + a_thread_mtx_desc_.CalculateOffset(make_tuple(0, MPerThreadSubC)),
                p_b_thread,
                p_c_thread + c_thread_mtx_desc.CalculateOffset(make_tuple(MPerThreadSubC, 0)));

            // read B_sub_0
            b_thread_copy_.Run(BlockMatrixB{},
                               make_tuple(k, Number<0>{}),
                               p_b_block,
                               b_thread_mtx_desc_,
                               make_tuple(Number<0>{}, Number<0>{}),
                               p_b_thread);

            // C_sub_11 += transpose(A_sub_1) * B_sub_1
            threadwise_gemm.Run(
                p_a_thread + a_thread_mtx_desc_.CalculateOffset(make_tuple(0, MPerThreadSubC)),
                p_b_thread + b_thread_mtx_desc_.CalculateOffset(make_tuple(0, NPerThreadSubC)),
                p_c_thread +
                    c_thread_mtx_desc.CalculateOffset(make_tuple(MPerThreadSubC, NPerThreadSubC)));

            // read B_sub_1
            b_thread_copy_.Run(BlockMatrixB{},
                               make_tuple(k, Number<NPerLevel1Cluster>{}),
                               p_b_block,
                               b_thread_mtx_desc_,
                               make_tuple(Number<0>{}, Number<NPerThreadSubC>{}),
                               p_b_thread);

            // read A_sub_1
            a_thread_copy_.Run(BlockMatrixA{},
                               make_tuple(k, Number<MPerLevel1Cluster>{}),
                               p_a_block,
                               a_thread_mtx_desc_,
                               make_tuple(Number<0>{}, Number<MPerThreadSubC>{}),
                               p_a_thread);

            // C_sub_00 += transpose(A_sub_0) * B_sub_0
            threadwise_gemm.Run(p_a_thread, p_b_thread, p_c_thread);

            // C_sub_01 += transpose(A_sub_0) * B_sub_1
            threadwise_gemm.Run(
                p_a_thread,
                p_b_thread + b_thread_mtx_desc_.CalculateOffset(make_tuple(0, NPerThreadSubC)),
                p_c_thread + c_thread_mtx_desc.CalculateOffset(make_tuple(0, NPerThreadSubC)));
        });

        // C_sub_10 += transpose(A_sub_1) * B_sub_0
        threadwise_gemm.Run(
            p_a_thread + a_thread_mtx_desc_.CalculateOffset(make_tuple(0, MPerThreadSubC)),
            p_b_thread,
            p_c_thread + c_thread_mtx_desc.CalculateOffset(make_tuple(MPerThreadSubC, 0)));

        // C_sub_11 += transpose(A_sub_1) * B_sub_1
        threadwise_gemm.Run(
            p_a_thread + a_thread_mtx_desc_.CalculateOffset(make_tuple(0, MPerThreadSubC)),
            p_b_thread + b_thread_mtx_desc_.CalculateOffset(make_tuple(0, NPerThreadSubC)),
            p_c_thread +
                c_thread_mtx_desc.CalculateOffset(make_tuple(MPerThreadSubC, NPerThreadSubC)));
    }

    __device__ void Run(const FloatA* p_a_block, const FloatB* p_b_block, FloatC* p_c_thread) const
    {
#if CK_EXPERIMENTAL_BLOCKWISE_GEMM_USE_PIPELINE
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr index_t MPerThread = ThreadMatrixC{}.GetLength(I0);
        constexpr index_t NPerThread = ThreadMatrixC{}.GetLength(I1);

        constexpr index_t MRepeat = MPerThread / MPerThreadSubC;
        constexpr index_t NRepeat = NPerThread / NPerThreadSubC;

        if constexpr(MRepeat == 2 && NRepeat == 2)
        {
            Run_pipelined_2x2(p_a_block, p_b_block, p_c_thread);
        }
        else
        {
            Run_naive(p_a_block, p_b_block, p_c_thread);
        }
#else
        Run_naive(p_a_block, p_b_block, p_c_thread);
#endif
    }
};
} // namespace ck
#endif
