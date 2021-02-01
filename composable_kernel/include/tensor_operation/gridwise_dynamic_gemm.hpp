#ifndef CK_GRIDWISE_DYNAMIC_GEMM_HPP
#define CK_GRIDWISE_DYNAMIC_GEMM_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_dynamic_tensor_slice_transfer.hpp"
#include "threadwise_dynamic_tensor_slice_transfer.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_gemm.hpp"

namespace ck {

template <index_t BlockSize,
          typename Float,
          typename AccFloat,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerThread,
          index_t NPerThread,
          index_t KPerThread,
          index_t MLevel0Cluster,
          index_t NLevel0Cluster,
          index_t MLevel1Cluster,
          index_t NLevel1Cluster,
          typename ABlockTransferThreadSliceLengths_K_M,
          typename ABlockTransferThreadClusterLengths_K_M,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_M,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_K_N,
          typename BBlockTransferThreadClusterLengths_K_N,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_N,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector>
struct GridwiseDynamicGemm_km_kn_mn_v1
{
    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr index_t max_lds_align = math::lcm(ABlockTransferDstScalarPerVector_M,
                                                    BBlockTransferDstScalarPerVector_N,
                                                    MPerThread,
                                                    NPerThread);

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m_block_desc = make_dynamic_naive_tensor_descriptor_aligned<2>(
            make_multi_index(KPerBlock, MPerBlock), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k_n_block_desc = make_dynamic_naive_tensor_descriptor_aligned<2>(
            make_multi_index(KPerBlock, NPerBlock), max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr index_t a_block_space_size =
            math::integer_least_multiple(a_k_m_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr index_t b_block_space_size =
            math::integer_least_multiple(b_k_n_block_desc.GetElementSpaceSize(), max_lds_align);

        return 2 * (a_block_space_size + b_block_space_size) * sizeof(Float);
    }

    template <typename... ADesc,
              typename... BDesc,
              typename... CDesc,
              bool HasMainKBlockLoop,
              bool HasDoubleTailKBlockLoop>
    __device__ void Run(const DynamicTensorDescriptor<ADesc...>& a_k_m_global_desc,
                        const Float* __restrict__ p_a_global,
                        const DynamicTensorDescriptor<BDesc...>& b_k_n_global_desc,
                        const Float* __restrict__ p_b_global,
                        const DynamicTensorDescriptor<CDesc...>& c_m0_m1_n0_n1_global_desc,
                        Float* __restrict__ p_c_global,
                        Float* __restrict__ p_shared_block,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        const index_t K = a_k_m_global_desc.GetLength(I0);
        const index_t M = a_k_m_global_desc.GetLength(I1);
        const index_t N = b_k_n_global_desc.GetLength(I1);

        // divide block work by [M, N]
#if 0
        const index_t m_block_work_num = M / MPerBlock;
        const index_t n_block_work_num = N / NPerBlock;
#else
        // Hack: this force result into SGPR
        const index_t m_block_work_num = __builtin_amdgcn_readfirstlane(M / MPerBlock);
        const index_t n_block_work_num = __builtin_amdgcn_readfirstlane(N / NPerBlock);
#endif

#if 0
        const index_t m_block_work_id = get_block_1d_id() / n_block_work_num;
        const index_t n_block_work_id = get_block_1d_id() - m_block_work_id * n_block_work_num;
#else
        // Hack: this force result into SGPR
        const index_t m_block_work_id =
            __builtin_amdgcn_readfirstlane(get_block_1d_id() / n_block_work_num);
        const index_t n_block_work_id = get_block_1d_id() - m_block_work_id * n_block_work_num;
#endif

        const index_t m_block_data_on_global = m_block_work_id * MPerBlock;
        const index_t n_block_data_on_global = n_block_work_id * NPerBlock;

        // lds max alignment
        constexpr index_t max_lds_align = math::lcm(ABlockTransferDstScalarPerVector_M,
                                                    BBlockTransferDstScalarPerVector_N,
                                                    MPerThread,
                                                    NPerThread);

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m_block_desc = make_dynamic_naive_tensor_descriptor_aligned<2>(
            make_multi_index(KPerBlock, MPerBlock), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k_n_block_desc = make_dynamic_naive_tensor_descriptor_aligned<2>(
            make_multi_index(KPerBlock, NPerBlock), max_lds_align);

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseDynamicTensorSliceTransfer_v4<BlockSize,
                                                   InMemoryDataOperation::Set,
                                                   Sequence<KPerBlock, MPerBlock>,
                                                   ABlockTransferThreadSliceLengths_K_M,
                                                   ABlockTransferThreadClusterLengths_K_M,
                                                   ABlockTransferThreadClusterArrangeOrder,
                                                   Float,
                                                   Float,
                                                   decltype(a_k_m_global_desc),
                                                   decltype(a_k_m_block_desc),
                                                   ABlockTransferSrcAccessOrder,
                                                   Sequence<0, 1>,
                                                   ABlockTransferSrcVectorDim,
                                                   1,
                                                   ABlockTransferSrcScalarPerVector,
                                                   ABlockTransferDstScalarPerVector_M,
                                                   AddressSpace::Global,
                                                   AddressSpace::Lds,
                                                   1,
                                                   1,
                                                   AThreadTransferSrcResetCoordinateAfterRun,
                                                   true>(
                a_k_m_global_desc,
                make_multi_index(0, m_block_data_on_global),
                a_k_m_block_desc,
                make_multi_index(0, 0));

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BlockwiseDynamicTensorSliceTransfer_v4<BlockSize,
                                                   InMemoryDataOperation::Set,
                                                   Sequence<KPerBlock, NPerBlock>,
                                                   BBlockTransferThreadSliceLengths_K_N,
                                                   BBlockTransferThreadClusterLengths_K_N,
                                                   BBlockTransferThreadClusterArrangeOrder,
                                                   Float,
                                                   Float,
                                                   decltype(b_k_n_global_desc),
                                                   decltype(b_k_n_block_desc),
                                                   BBlockTransferSrcAccessOrder,
                                                   Sequence<0, 1>,
                                                   BBlockTransferSrcVectorDim,
                                                   1,
                                                   BBlockTransferSrcScalarPerVector,
                                                   BBlockTransferDstScalarPerVector_N,
                                                   AddressSpace::Global,
                                                   AddressSpace::Lds,
                                                   1,
                                                   1,
                                                   BThreadTransferSrcResetCoordinateAfterRun,
                                                   true>(
                b_k_n_global_desc,
                make_multi_index(0, n_block_data_on_global),
                b_k_n_block_desc,
                make_multi_index(0, 0));

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlocl, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        constexpr index_t a_k_m_block_mtx_stride =
            a_k_m_block_desc.CalculateOffset(make_multi_index(1, 0)) -
            a_k_m_block_desc.CalculateOffset(make_multi_index(0, 0));
        constexpr index_t b_k_n_block_mtx_stride =
            b_k_n_block_desc.CalculateOffset(make_multi_index(1, 0)) -
            b_k_n_block_desc.CalculateOffset(make_multi_index(0, 0));

        constexpr auto a_k_m_block_mtx_desc = make_ConstantMatrixDescriptor(
            Number<KPerBlock>{}, Number<MPerBlock>{}, Number<a_k_m_block_mtx_stride>{});
        constexpr auto b_k_n_block_mtx_desc = make_ConstantMatrixDescriptor(
            Number<KPerBlock>{}, Number<NPerBlock>{}, Number<b_k_n_block_mtx_stride>{});

        // sanity check
        static_assert(MPerBlock % (MPerThread * MLevel0Cluster * MLevel1Cluster) == 0 &&
                          NPerBlock % (NPerThread * NLevel0Cluster * NLevel1Cluster) == 0,
                      "wrong!");

        constexpr index_t MRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
        constexpr index_t NRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_m0m1_n0n1_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
            Number<MRepeat * MPerThread>{}, Number<NRepeat * NPerThread>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
            BlockSize,
            decltype(a_k_m_block_mtx_desc),
            decltype(b_k_n_block_mtx_desc),
            decltype(c_m0m1_n0n1_thread_mtx_desc),
            MPerThread,
            NPerThread,
            KPerThread,
            MLevel0Cluster,
            NLevel0Cluster,
            MLevel1Cluster,
            NLevel1Cluster,
            MPerThread,
            NPerThread>{};

        // LDS allocation for A and B: be careful of alignment
        constexpr index_t a_block_space_size =
            math::integer_least_multiple(a_k_m_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr index_t b_block_space_size =
            math::integer_least_multiple(b_k_n_block_desc.GetElementSpaceSize(), max_lds_align);

        Float* p_a_block_double = p_shared_block;
        Float* p_b_block_double = p_shared_block + 2 * a_block_space_size;

        // register allocation for output
        AccFloat p_c_thread[c_m0m1_n0n1_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_m0m1_n0n1_thread_mtx_desc, p_c_thread);

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock, 0);

        // hack to control index calculation when iterating over a_k_m_global tensor
        constexpr auto a_k_m_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        constexpr auto a_k_m_global_reset_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        // hack to control index calculation when iterating over b_k_n_global tensor
#if 0
        // for padded input
        constexpr auto b_k_n_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2>{}));

        constexpr auto b_k_n_global_move_slice_window_iterator_hack =
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2>{};
#elif 1
        // for non-padded input
        constexpr auto b_k_n_global_iterator_hacks = make_tuple(
            make_tuple(Sequence<0, 0, 0, 0, 0, 1, 0>{}, Sequence<0, 0, 0, 0, 0, 0, 1>{}),
            make_tuple(Sequence<0, 0, 0, 0, 0, 2, 0>{}, Sequence<0, 0, 0, 0, 0, 0, 2>{}));

        constexpr auto b_k_n_global_move_slice_window_iterator_hack =
            Sequence<0, 0, 0, 0, 0, 1, 2>{};
#elif 1
        // for 1x1 case
        constexpr auto b_k_n_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 1, 0>{}, Sequence<0, 0, 1>{}),
                       make_tuple(Sequence<0, 2, 0>{}, Sequence<0, 0, 2>{}));

        constexpr auto b_k_n_global_move_slice_window_iterator_hack = Sequence<0, 1, 2>{};
#endif

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.RunRead(a_k_m_global_desc, p_a_global, a_k_m_global_iterator_hacks);
            b_blockwise_copy.RunRead(b_k_n_global_desc, p_b_global, b_k_n_global_iterator_hacks);

            a_blockwise_copy.RunWrite(a_k_m_block_desc, p_a_block_double);
            b_blockwise_copy.RunWrite(b_k_n_block_desc, p_b_block_double);
        }

        if constexpr(HasMainKBlockLoop)
        {
            Float* p_a_block_even = p_a_block_double;
            Float* p_b_block_even = p_b_block_double;

            Float* p_a_block_odd = p_a_block_double + a_block_space_size;
            Float* p_b_block_odd = p_b_block_double + b_block_space_size;

            index_t k_block_data_begin = 0;

            // LDS double buffer: main body
            // use Do-While loop instead of For loop to simplify control flow
            do
            {
                // even iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_k_m_global_desc, a_block_slice_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_k_n_global_desc,
                                                    b_block_slice_copy_step,
                                                    b_k_n_global_move_slice_window_iterator_hack);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_k_m_global_desc, p_a_global, a_k_m_global_iterator_hacks);
                b_blockwise_copy.RunRead(
                    b_k_n_global_desc, p_b_global, b_k_n_global_iterator_hacks);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(p_a_block_even, p_b_block_even, p_c_thread);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_k_m_block_desc, p_a_block_odd);
                b_blockwise_copy.RunWrite(b_k_n_block_desc, p_b_block_odd);

                // odd iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_k_m_global_desc, a_block_slice_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_k_n_global_desc,
                                                    b_block_slice_copy_step,
                                                    b_k_n_global_move_slice_window_iterator_hack);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_k_m_global_desc, p_a_global, a_k_m_global_iterator_hacks);
                b_blockwise_copy.RunRead(
                    b_k_n_global_desc, p_b_global, b_k_n_global_iterator_hacks);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(p_a_block_odd, p_b_block_odd, p_c_thread);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_k_m_block_desc, p_a_block_even);
                b_blockwise_copy.RunWrite(b_k_n_block_desc, p_b_block_even);

                k_block_data_begin += 2 * KPerBlock;
            } while(k_block_data_begin < K - 2 * KPerBlock);
        }

        // LDS double buffer: tail
        if constexpr(HasDoubleTailKBlockLoop) // if has 2 iteration left
        {
            a_blockwise_copy.MoveSrcSliceWindow(a_k_m_global_desc, a_block_slice_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_k_n_global_desc,
                                                b_block_slice_copy_step,
                                                b_k_n_global_move_slice_window_iterator_hack);

            __syncthreads();

            // LDS double buffer: load last data from device mem
            a_blockwise_copy.RunRead(a_k_m_global_desc, p_a_global, a_k_m_global_iterator_hacks);
            b_blockwise_copy.RunRead(b_k_n_global_desc, p_b_global, b_k_n_global_iterator_hacks);

            // LDS double buffer: GEMM on 2nd-last data
            blockwise_gemm.Run(p_a_block_double, p_b_block_double, p_c_thread);

            // LDS double buffer: store last data to LDS
            a_blockwise_copy.RunWrite(a_k_m_block_desc, p_a_block_double + a_block_space_size);
            b_blockwise_copy.RunWrite(b_k_n_block_desc, p_b_block_double + b_block_space_size);

            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(p_a_block_double + a_block_space_size,
                               p_b_block_double + b_block_space_size,
                               p_c_thread);
        }
        else // if has 1 iteration left
        {
            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(p_a_block_double, p_b_block_double, p_c_thread);
        }

        // output: register to global memory
        {
            constexpr index_t M1 = MPerThread * MLevel0Cluster * MLevel1Cluster;
            constexpr index_t N1 = NPerThread * NLevel0Cluster * NLevel1Cluster;

            // define input tensor descriptor for threadwise copy
            //     thread input tensor, src of threadwise copy
#if 0 // debug
            constexpr auto c_m0_m1_n0_n1_thread_desc =
                make_dynamic_naive_tensor_descriptor_packed<4>(
                    make_multi_index(MRepeat, MPerThread, NRepeat, NPerThread));
#else
            constexpr auto c_m0_m1_n0_n1_thread_desc = make_native_tensor_descriptor_packed(
                Sequence<MRepeat, MPerThread, NRepeat, NPerThread>{});
#endif

            // calculate origin of thread input tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            const index_t m_thread_data_on_global =
                m_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t n_thread_data_on_global =
                n_block_data_on_global + c_thread_mtx_on_block.col;

            // hack to control index calculation when iterating over c_m0_m1_n0_n1_global tensor
            // hack for NKHW format
            constexpr auto c_m0_m1_n0_n1_global_tensor_iterator_hacks =
                make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                      Sequence<0, 0, 0, 0, 0>{},
                                      Sequence<0, 0, 1, 0, 0>{},
                                      Sequence<0, 0, 1, 0, 0>{}),
                           make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                      Sequence<0, 0, 0, 0, 0>{},
                                      Sequence<0, 0, 2, 0, 0>{},
                                      Sequence<0, 0, 2, 0, 0>{}));

            ThreadwiseDynamicTensorSliceTransfer_v1r3<
                AccFloat,
                Float,
                decltype(c_m0_m1_n0_n1_thread_desc),
                decltype(c_m0_m1_n0_n1_global_desc),
                Sequence<MRepeat, MPerThread, NRepeat, NPerThread>,
                CThreadTransferSrcDstAccessOrder,
                CThreadTransferSrcDstVectorDim,
                CThreadTransferDstScalarPerVector,
                AddressSpace::Vgpr,
                AddressSpace::Global,
                CGlobalMemoryDataOperation,
                1,
                true,
                true>(c_m0_m1_n0_n1_global_desc,
                      make_multi_index(m_thread_data_on_global / M1,
                                       m_thread_data_on_global % M1,
                                       n_thread_data_on_global / N1,
                                       n_thread_data_on_global % N1))
                .Run(p_c_thread,
                     c_m0_m1_n0_n1_global_desc,
                     p_c_global,
                     c_m0_m1_n0_n1_global_tensor_iterator_hacks);
        }
    }

    template <typename... ADesc,
              typename... BDesc,
              typename... CDesc,
              bool HasMainKBlockLoop,
              bool HasDoubleTailKBlockLoop>
    __device__ void Run(const DynamicTensorDescriptor<ADesc...>& a_k_m_global_desc,
                        const Float* __restrict__ p_a_global,
                        const DynamicTensorDescriptor<BDesc...>& b_k_n_global_desc,
                        const Float* __restrict__ p_b_global,
                        const DynamicTensorDescriptor<CDesc...>& c_m0_m1_n0_n1_global_desc,
                        Float* __restrict__ p_c_global,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        constexpr index_t shared_block_size = GetSharedMemoryNumberOfByte() / sizeof(Float);

        __shared__ Float p_shared_block[shared_block_size];

        Run(a_k_m_global_desc,
            p_a_global,
            b_k_n_global_desc,
            p_b_global,
            c_m0_m1_n0_n1_global_desc,
            p_c_global,
            p_shared_block,
            integral_constant<bool, HasMainKBlockLoop>{},
            integral_constant<bool, HasDoubleTailKBlockLoop>{});
    }
};

} // namespace ck
#endif
