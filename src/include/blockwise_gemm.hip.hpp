#pragma once
#include "threadwise_gemm.hip.hpp"

template <unsigned BlockSize,
          class BlockMatrixA,
          class BlockMatrixB,
          class ThreadMatrixC,
          unsigned BlockMatrixStrideA,
          unsigned BlockMatrixStrideB,
          unsigned ThreadMatrixStrideC,
          unsigned BatchSize,
          unsigned MPerThreadSubC,
          unsigned NPerThreadSubC,
          unsigned MLevel0Cluster,
          unsigned NLevel0Cluster,
          unsigned MLevel1Cluster,
          unsigned NLevel1Cluster,
          unsigned KPerThreadLoop,
          unsigned BatchPerThread>
struct BlockwiseBatchGemmBlockABlockBThreadCTransANormalBNormalC_V2
{
    unsigned mMyThreadOffsetA = 0;
    unsigned mMyThreadOffsetB = 0;

    struct MatrixIndex
    {
        unsigned batch;
        unsigned row;
        unsigned col;
    };

    __device__ BlockwiseBatchGemmBlockABlockBThreadCTransANormalBNormalC_V2()
    {
        static_assert(BatchSize % BatchPerThread == 0,
                      "wrong! BatchSize is not dividable by BatchPerThread");

        constexpr unsigned BatchThreadWork = BatchSize / BatchPerThread;

        constexpr unsigned ThreadPerLevel1Cluster =
            MLevel0Cluster * NLevel0Cluster * MLevel1Cluster * NLevel1Cluster;

        static_assert(BlockSize == BatchThreadWork * ThreadPerLevel1Cluster,
                      "wrong! wrong blocksize\n");

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        static_assert(a_block_mtx.NRow() == b_block_mtx.NRow(),
                      "wrong! K dimension not consistent\n");

        constexpr unsigned M = a_block_mtx.NCol(); // A is transposed
        constexpr unsigned N = b_block_mtx.NCol();
        constexpr unsigned K = a_block_mtx.NRow();

        constexpr unsigned MPerThread = c_thread_mtx.NRow();
        constexpr unsigned NPerThread = c_thread_mtx.NCol();

        static_assert((MPerThread % MPerThreadSubC == 0) && (NPerThread % NPerThreadSubC == 0),
                      "wrong! Cannot evenly divide thread work among repeat \n");

        constexpr unsigned MRepeat = MPerThread / MPerThreadSubC;
        constexpr unsigned NRepeat = NPerThread / NPerThreadSubC;

        static_assert((M % MRepeat == 0) && (N % NRepeat == 0),
                      "wrong! Cannot evenly divide work among repeat\n");

        constexpr unsigned MPerLevel1Cluster = M / MRepeat;
        constexpr unsigned NPerLevel1Cluster = N / NRepeat;

        static_assert((MPerLevel1Cluster % MLevel1Cluster == 0) &&
                          (NPerLevel1Cluster % NLevel1Cluster == 0),
                      "wrong! Cannot evenly divide work among Level1Cluster\n");

        constexpr unsigned MPerLevel0Cluster = MPerLevel1Cluster / MLevel1Cluster;
        constexpr unsigned NPerLevel0Cluster = NPerLevel1Cluster / NLevel1Cluster;

        static_assert((MPerLevel0Cluster % MLevel0Cluster == 0) &&
                          (NPerLevel0Cluster % NLevel0Cluster == 0),
                      "wrong! Cannot evenly divide work among Level0Cluster\n");

        static_assert((MPerThreadSubC == MPerLevel0Cluster / MLevel0Cluster) &&
                          (NPerThreadSubC == NPerLevel0Cluster / NLevel0Cluster),
                      "wrong! thread work size is wrong\n");

        const auto c_thread_mtx_index = GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        mMyThreadOffsetA = c_thread_mtx_index.batch * BlockMatrixStrideA +
                           a_block_mtx.Get1dIndex(0, c_thread_mtx_index.row);

        mMyThreadOffsetB = c_thread_mtx_index.batch * BlockMatrixStrideB +
                           b_block_mtx.Get1dIndex(0, c_thread_mtx_index.col);

#if 0
        if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
        {
            print_ConstantMatrixDescriptor(BlockMatrixA{}, "a_block_mtx: ");
            print_ConstantMatrixDescriptor(BlockMatrixB{}, "b_block_mtx: ");
            print_ConstantMatrixDescriptor(ThreadMatrixC{}, "c_thread_mtx: ");

            printf("%u %u, %u %u %u, %u %u\n",
                   get_block_1d_id(),
                   get_thread_local_1d_id(),
                   c_thread_mtx_index.batch,
                   c_thread_mtx_index.row,
                   c_thread_mtx_index.col,
                   mMyThreadOffsetA,
                   mMyThreadOffsetB);
        }
#endif
    }

    __device__ MatrixIndex GetBeginOfThreadMatrixC(unsigned thread_id) const
    {
        constexpr unsigned BatchThreadWork = BatchSize / BatchPerThread;

        constexpr unsigned ThreadPerLevel1Cluster =
            MLevel0Cluster * NLevel0Cluster * MLevel1Cluster * NLevel1Cluster;

        constexpr unsigned ThreadPerLevel0Cluster = MLevel0Cluster * NLevel0Cluster;

        unsigned batch_work_id = thread_id / ThreadPerLevel1Cluster;
        unsigned cluster_id    = thread_id - batch_work_id * ThreadPerLevel1Cluster;

        unsigned level1_id   = cluster_id / ThreadPerLevel0Cluster;
        unsigned level1_m_id = level1_id / NLevel1Cluster;
        unsigned level1_n_id = level1_id % NLevel1Cluster;

        unsigned level0_id   = cluster_id % ThreadPerLevel0Cluster;
        unsigned level0_m_id = level0_id / NLevel0Cluster;
        unsigned level0_n_id = level0_id % NLevel0Cluster;

        constexpr unsigned MPerLevel0Cluster = MPerThreadSubC * MLevel0Cluster;
        constexpr unsigned NPerLevel0Cluster = NPerThreadSubC * NLevel0Cluster;

        return MatrixIndex{batch_work_id * BatchPerThread,
                           level1_m_id * MPerLevel0Cluster + level0_m_id * MPerThreadSubC,
                           level1_n_id * NPerLevel0Cluster + level0_n_id * NPerThreadSubC};
    }

    template <class FloatA, class FloatB, class FloatC, class Accumulator>
    __device__ void Run(const FloatA* __restrict__ p_a_block,
                        const FloatB* __restrict__ p_b_block,
                        FloatC* __restrict__ p_c_thread,
                        Accumulator f_accum) const
    {
        constexpr auto True  = integral_constant<bool, true>{};
        constexpr auto False = integral_constant<bool, false>{};

        constexpr auto a_block_mtx  = BlockMatrixA{};
        constexpr auto b_block_mtx  = BlockMatrixB{};
        constexpr auto c_thread_mtx = ThreadMatrixC{};

        constexpr unsigned KPerBlock = a_block_mtx.NRow(); // A is transposed

        constexpr unsigned MPerThread = c_thread_mtx.NRow();
        constexpr unsigned NPerThread = c_thread_mtx.NCol();

        // thread A, B for GEMM
        //   A is transposed, b is not
        constexpr auto a_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<MPerThread>{});

        constexpr auto b_thread_mtx =
            make_ConstantMatrixDescriptor(Number<KPerThreadLoop>{}, Number<NPerThread>{});

        // thread A-sub, B-sub for copy
        constexpr auto a_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<MPerThreadSubC>{}, Number<MPerThread>{});

        constexpr auto b_thread_sub_mtx = make_ConstantMatrixDescriptor(
            Number<KPerThreadLoop>{}, Number<NPerThreadSubC>{}, Number<NPerThread>{});

        FloatA p_a_thread[a_thread_mtx.GetElementSpace()];
        FloatB p_b_thread[b_thread_mtx.GetElementSpace()];

        constexpr unsigned MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
        constexpr unsigned NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;

        constexpr unsigned MRepeat = MPerThread / MPerThreadSubC;
        constexpr unsigned NRepeat = NPerThread / NPerThreadSubC;

        // loop over k
#pragma unroll
        for(unsigned k_begin = 0; k_begin < KPerBlock; k_begin += KPerThreadLoop)
        {
            // read first batch of A, B
            //   copy A-sub to form A
#pragma unroll
            for(unsigned m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
            {
                threadwise_matrix_copy(
                    a_block_mtx,
                    p_a_block + a_block_mtx.Get1dIndex(k_begin, m_repeat * MPerLevel1Cluster) +
                        mMyThreadOffsetA,
                    a_thread_mtx,
                    p_a_thread + a_thread_mtx.Get1dIndex(0, m_repeat * MPerThreadSubC),
                    a_thread_sub_mtx.GetLengths());
            }

            //   copy B-sub to form B
#pragma unroll
            for(unsigned n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
            {
                threadwise_matrix_copy(
                    b_block_mtx,
                    p_b_block + b_block_mtx.Get1dIndex(k_begin, n_repeat * NPerLevel1Cluster) +
                        mMyThreadOffsetB,
                    b_thread_mtx,
                    p_b_thread + b_thread_mtx.Get1dIndex(0, n_repeat * NPerThreadSubC),
                    b_thread_sub_mtx.GetLengths());
            }

            // loop over batch
#pragma unroll
            for(unsigned ib = 0; ib + 1 < BatchPerThread; ++ib)
            {
                // do current batch of gemm
                threadwise_gemm(a_thread_mtx,
                                True,
                                p_a_thread,
                                b_thread_mtx,
                                False,
                                p_b_thread,
                                c_thread_mtx,
                                False,
                                p_c_thread + ib * ThreadMatrixStrideC,
                                f_accum);

                // read next batch of a, b
                if(BlockMatrixStrideA != 0)
                {
#pragma unroll
                    for(unsigned m_repeat = 0; m_repeat < MRepeat; ++m_repeat)
                    {
                        threadwise_matrix_copy(
                            a_block_mtx,
                            p_a_block +
                                a_block_mtx.Get1dIndex(k_begin, m_repeat * MPerLevel1Cluster) +
                                (ib + 1) * BlockMatrixStrideA + mMyThreadOffsetA,
                            a_thread_mtx,
                            p_a_thread + a_thread_mtx.Get1dIndex(0, m_repeat * MPerThreadSubC),
                            a_thread_sub_mtx.GetLengths());
                    }
                }

                if(BlockMatrixStrideB != 0)
                {
#pragma unroll
                    for(unsigned n_repeat = 0; n_repeat < NRepeat; ++n_repeat)
                    {
                        threadwise_matrix_copy(
                            b_block_mtx,
                            p_b_block +
                                b_block_mtx.Get1dIndex(k_begin, n_repeat * NPerLevel1Cluster) +
                                (ib + 1) * BlockMatrixStrideB + mMyThreadOffsetB,
                            b_thread_mtx,
                            p_b_thread + b_thread_mtx.Get1dIndex(0, n_repeat * NPerThreadSubC),
                            b_thread_sub_mtx.GetLengths());
                    }
                }
            }

            // do last batch of gemm
            threadwise_gemm(a_thread_mtx,
                            True,
                            p_a_thread,
                            b_thread_mtx,
                            False,
                            p_b_thread,
                            c_thread_mtx,
                            False,
                            p_c_thread + (BatchPerThread - 1) * ThreadMatrixStrideC,
                            f_accum);
        }
    }
};
