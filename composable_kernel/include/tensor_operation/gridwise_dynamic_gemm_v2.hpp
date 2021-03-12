#ifndef CK_GRIDWISE_DYNAMIC_GEMM_V2_HPP
#define CK_GRIDWISE_DYNAMIC_GEMM_V2_HPP

#include "common_header.hpp"
#include "dynamic_multi_index_transform_helper.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "blockwise_dynamic_tensor_slice_transfer.hpp"
#include "threadwise_dynamic_tensor_slice_transfer.hpp"
#include "blockwise_gemm_v3.hpp"

namespace ck {

template <index_t BlockSize,
          typename Float,
          typename AccFloat,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          typename AGlobalDesc,
          typename BGlobalDesc,
          typename CGlobalDesc,
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
          index_t CThreadTransferDstScalarPerVector,
          typename AGlobalIteratorHacks,
          typename BGlobalIteratorHacks,
          typename CGlobalIteratorHacks,
          typename AGlobalMoveSliceWindowIteratorHacks,
          typename BGlobalMoveSliceWindowIteratorHacks>
struct GridwiseDynamicGemm_km_kn_mn_v2
{
    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto max_lds_align = math::lcm(Number<ABlockTransferDstScalarPerVector_M>{},
                                                 Number<BBlockTransferDstScalarPerVector_N>{},
                                                 Number<MPerThread>{},
                                                 Number<NPerThread>{});

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<MPerBlock>{}), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_cyx_n_h_w_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<1>{}, Number<8>{}, Number<8>{}), max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_k_m_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size =
            math::integer_least_multiple(b_cyx_n_h_w_block_desc.GetElementSpaceSize(), max_lds_align);

        return 2 * (a_block_space_size + b_block_space_size) * sizeof(Float);
    }

    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ void Run(const AGlobalDesc& a_k_m_global_desc,
                        const Float* __restrict__ p_a_global,
                        const BGlobalDesc& b_cyx_n_h_w_global_desc,
                        const Float* __restrict__ p_b_global,
                        const CGlobalDesc& c_k_n_h_w_global_desc,
                        Float* __restrict__ p_c_global,
                        Float* __restrict__ p_shared_block,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        const auto K = a_k_m_global_desc.GetLength(I0);
        const auto M = a_k_m_global_desc.GetLength(I1);
        const auto N = b_cyx_n_h_w_global_desc.GetLength(I1);

        // divide block work by [M, N]
#if 0
        const auto m_block_work_num = M / Number<MPerBlock>{};
        const auto n_block_work_num = N / Number<NPerBlock>{};

        const index_t m_block_work_id = get_block_1d_id() / n_block_work_num;
        const index_t n_block_work_id = get_block_1d_id() - m_block_work_id * n_block_work_num;

#else
        // Hack: this force result into SGPR
        const index_t m_block_work_num = __builtin_amdgcn_readfirstlane(M / MPerBlock);
        const index_t n_block_work_num = __builtin_amdgcn_readfirstlane(N / NPerBlock);

        const index_t m_block_work_id =
            __builtin_amdgcn_readfirstlane(get_block_1d_id() / n_block_work_num);
        const index_t n_block_work_id = get_block_1d_id() - m_block_work_id * n_block_work_num;
#endif

        const index_t m_block_data_on_global = m_block_work_id * MPerBlock;

        const index_t h_block_data_on_global = n_block_work_id * 8;
        const index_t w_block_data_on_global = n_block_work_id * 8;

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(Number<ABlockTransferDstScalarPerVector_M>{},
                                                 Number<BBlockTransferDstScalarPerVector_N>{},
                                                 Number<MPerThread>{},
                                                 Number<NPerThread>{});

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<MPerBlock>{}), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_cyx_n_h_w_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, Number<1>{}, Number<8>{}, Number<8>{}), max_lds_align);

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
        auto b_blockwise_copy = BlockwiseDynamicTensorSliceTransfer_v4<
            BlockSize,
            InMemoryDataOperation::Set,
            Sequence<KPerBlock, 1, 8, 8>, // BlockSliceLengths
            Sequence<KPerBlock, 1, 1, 1>, // ThreadSliceLengths_K_N
            Sequence<1, 1, 8, 8>,         // ThreadClusterLengths_K_N
            Sequence<3, 2, 0, 1>,         // ThreadClusterArrangeOrder
            Float,
            Float,
            decltype(b_cyx_n_h_w_global_desc), // SrcDesc
            decltype(b_cyx_n_h_w_block_desc),  // DstDesc
            Sequence<3, 2, 0, 1>,        // SrcDimAccessOrder
            Sequence<3, 2, 0, 1>,        // DstDimAccessOrder
            3,                           // SrcVectorDim
            3,                           // DstVectorDim
            1,                           // SrcScalarPerVector
            1,                           // DstScalarPerVector
            AddressSpace::Global,
            AddressSpace::Lds,
            1,
            1,
            BThreadTransferSrcResetCoordinateAfterRun,
            true>(b_cyx_n_h_w_global_desc,
                  make_multi_index(0, 0, h_block_data_on_global, w_block_data_on_global),
                  b_cyx_n_h_w_block_desc,
                  make_multi_index(0, 0, 0, 0));

#if 0
        constexpr auto b_cyx_n_h_w_thread_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<KPerThread>{}, Number<NPerThread>{}));

        using BThreadwiseTransfer =
            ThreadwiseDynamicTensorSliceTransfer_v2<Float,
                                                      Float,
                                                      decltype(b_cyx_n_h_w_global_desc),
                                                      decltype(b_cyx_n_h_w_thread_desc),
                                                      Sequence<KPerThread, NPerThread>,
                                                      BBlockTransferSrcAccessOrder,
                                                      BBlockTransferSrcVectorDim,
                                                      BBlockTransferSrcScalarPerVector,
                                                      AddressSpace::Global,
                                                      AddressSpace::Vgpr,
                                                      InMemoryDataOperation::Set,
                                                      1,
                                                      true>;
#endif

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlocl, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check
        //static_assert(MPerBlock % (MPerThread * MLevel0Cluster * MLevel1Cluster) == 0 &&
                          //NPerBlock % (NPerThread * NLevel0Cluster * NLevel1Cluster) == 0,
                      //"wrong!");

        // constexpr index_t MRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
        // constexpr index_t NRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_k_n_h_w_thread_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
            make_tuple(Number<MPerThread>{}, Number<1>{}, Number<1>{}, Number<1>{}));

#if 0
        const auto blockwise_gemm =
            BlockwiseGemm_km_kn_m0m1n0n1_v3<BlockSize,
            decltype(a_k_m_block_desc),
            decltype(b_cyx_n_h_w_block_desc),
            decltype(c_k_n_h_w_thread_desc),
            MPerThread,
            NPerThread,
            KPerThread,
            MLevel0Cluster,
            NLevel0Cluster,
            MLevel1Cluster,
            NLevel1Cluster,
            1,
            1>{};
#endif

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_k_m_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size =
            math::integer_least_multiple(b_cyx_n_h_w_block_desc.GetElementSpaceSize(), max_lds_align);

        Float* p_a_block_double = p_shared_block;
        Float* p_b_block_double = p_shared_block + 2 * a_block_space_size;

        // register allocation for output
        AccFloat p_c_thread[c_k_n_h_w_thread_desc.GetElementSpaceSize()];

        for(index_t i = 0; i < c_k_n_h_w_thread_desc.GetElementSpaceSize(); i++)
        {
            p_c_thread[i] = 0;
        }

        // zero out threadwise output
        // threadwise_matrix_set_zero_v2(c_k_n_h_w_thread_desc, p_c_thread);

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock, 0, 0, 0);

        // hack to control index calculation when iterating over A and B matrix for threadwise copy
        constexpr auto a_k_m_global_iterator_hacks = AGlobalIteratorHacks{};
        constexpr auto b_cyx_n_h_w_global_iterator_hacks = BGlobalIteratorHacks{};

        // hack to control index calculation when move slice window for A and B matrix for
        // threadwise copy
        constexpr auto a_k_m_global_move_slice_window_iterator_hack =
            AGlobalMoveSliceWindowIteratorHacks{};
        constexpr auto b_cyx_n_h_w_global_move_slice_window_iterator_hack =
            BGlobalMoveSliceWindowIteratorHacks{};

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.RunRead(a_k_m_global_desc, p_a_global, a_k_m_global_iterator_hacks);
            b_blockwise_copy.RunRead(b_cyx_n_h_w_global_desc, p_b_global, b_cyx_n_h_w_global_iterator_hacks);

            a_blockwise_copy.RunWrite(a_k_m_block_desc, p_a_block_double);
            b_blockwise_copy.RunWrite(b_cyx_n_h_w_block_desc, p_b_block_double);
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
                a_blockwise_copy.MoveSrcSliceWindow(a_k_m_global_desc,
                                                    a_block_slice_copy_step,
                                                    a_k_m_global_move_slice_window_iterator_hack);

                // b_blockwise_copy.MoveSrcSliceWindow(b_cyx_n_h_w_global_desc,
                // b_block_slice_copy_step,
                // b_cyx_n_h_w_global_move_slice_window_iterator_hack);

                b_blockwise_copy.MoveSrcSliceWindow(b_cyx_n_h_w_global_desc,
                                                    b_block_slice_copy_step,
                                                    b_cyx_n_h_w_global_move_slice_window_iterator_hack);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_k_m_global_desc, p_a_global, a_k_m_global_iterator_hacks);
                b_blockwise_copy.RunRead(
                    b_cyx_n_h_w_global_desc, p_b_global, b_cyx_n_h_w_global_iterator_hacks);

                // LDS double buffer: GEMM on current data
                // blockwise_gemm.Run(p_a_block_even, p_b_block_even, p_c_thread);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_k_m_block_desc, p_a_block_odd);
                b_blockwise_copy.RunWrite(b_cyx_n_h_w_block_desc, p_b_block_odd);

                // odd iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_k_m_global_desc,
                                                    a_block_slice_copy_step,
                                                    a_k_m_global_move_slice_window_iterator_hack);
                b_blockwise_copy.MoveSrcSliceWindow(b_cyx_n_h_w_global_desc,
                                                    b_block_slice_copy_step,
                                                    b_cyx_n_h_w_global_move_slice_window_iterator_hack);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_k_m_global_desc, p_a_global, a_k_m_global_iterator_hacks);
                b_blockwise_copy.RunRead(
                    b_cyx_n_h_w_global_desc, p_b_global, b_cyx_n_h_w_global_iterator_hacks);

                // LDS double buffer: GEMM on current data
                // blockwise_gemm.Run(p_a_block_odd, p_b_block_odd, p_c_thread);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_k_m_block_desc, p_a_block_even);
                b_blockwise_copy.RunWrite(b_cyx_n_h_w_block_desc, p_b_block_even);

                k_block_data_begin += 2 * KPerBlock;
            } while(k_block_data_begin < K - 2 * KPerBlock);
        }

        // LDS double buffer: tail
        if constexpr(HasDoubleTailKBlockLoop) // if has 2 iteration left
        {
            a_blockwise_copy.MoveSrcSliceWindow(a_k_m_global_desc,
                                                a_block_slice_copy_step,
                                                a_k_m_global_move_slice_window_iterator_hack);
            b_blockwise_copy.MoveSrcSliceWindow(b_cyx_n_h_w_global_desc,
                                                b_block_slice_copy_step,
                                                b_cyx_n_h_w_global_move_slice_window_iterator_hack);

            __syncthreads();

            // LDS double buffer: load last data from device mem
            a_blockwise_copy.RunRead(a_k_m_global_desc, p_a_global, a_k_m_global_iterator_hacks);
            b_blockwise_copy.RunRead(b_cyx_n_h_w_global_desc, p_b_global, b_cyx_n_h_w_global_iterator_hacks);

            // LDS double buffer: GEMM on 2nd-last data
            // blockwise_gemm.Run(p_a_block_double, p_b_block_double, p_c_thread);

            // LDS double buffer: store last data to LDS
            a_blockwise_copy.RunWrite(a_k_m_block_desc, p_a_block_double + a_block_space_size);
            b_blockwise_copy.RunWrite(b_cyx_n_h_w_block_desc, p_b_block_double + b_block_space_size);

            __syncthreads();

            // LDS double buffer: GEMM on last data
            // blockwise_gemm.Run(p_a_block_double + a_block_space_size,
            // p_b_block_double + b_block_space_size,
            // p_c_thread);
        }
        else // if has 1 iteration left
        {
            __syncthreads();

            // LDS double buffer: GEMM on last data
            // blockwise_gemm.Run(p_a_block_double, p_b_block_double, p_c_thread);
        }

#if 1
        // output: register to global memory
        {
            // define input tensor descriptor for threadwise copy
            //     thread input tensor, src of threadwise copy
            constexpr auto c_k_n_h_w_thread_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
                make_tuple(Number<MPerThread>{}, Number<1>{}, Number<1>{}, Number<1>{}));

            // calculate origin of thread input tensor on global memory
            //     blockwise GEMM c matrix starting index
#if 0
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            const index_t m_thread_data_on_global =
                m_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t n_thread_data_on_global =
                n_block_data_on_global + c_thread_mtx_on_block.col;
#endif
            const index_t h_thread_id = get_thread_local_1d_id() / 8;
            const index_t w_thread_id = get_thread_local_1d_id() % 8;

            const index_t m_thread_data_on_global = m_block_data_on_global;
            const index_t h_thread_data_on_global = h_block_data_on_global + h_thread_id;
            const index_t w_thread_data_on_global = w_block_data_on_global + w_thread_id;

            // hack to control index calculation when iterating over c_k_n_h_w_global tensor
            constexpr auto c_k_n_h_w_global_tensor_iterator_hacks = CGlobalIteratorHacks{};

            // constexpr auto tmp = make_unmerge_transform(make_tuple(
            // Number<MRepeat>{}, Number<MPerThread>{}, Number<NRepeat>{}, Number<NPerThread>{}));

            ThreadwiseDynamicTensorSliceTransfer_v1r3<
                AccFloat,
                Float,
                decltype(c_k_n_h_w_thread_desc),
                decltype(c_k_n_h_w_global_desc),
                Sequence<MPerThread, 1, 1, 1>,
                Sequence<3, 2, 0, 1>, // CThreadTransferSrcDstAccessOrder
                3,                    // CThreadTransferSrcDstVectorDim
                1,                    // CThreadTransferDstScalarPerVector,
                AddressSpace::Vgpr,
                AddressSpace::Global,
                CGlobalMemoryDataOperation,
                1,
                true>(
                c_k_n_h_w_global_desc,
                make_multi_index(
                    m_thread_data_on_global, 0, h_thread_data_on_global, w_thread_data_on_global))
                .Run(c_k_n_h_w_thread_desc,
                     make_tuple(I0, I0, I0, I0),
                     p_c_thread,
                     c_k_n_h_w_global_desc,
                     p_c_global,
                     c_k_n_h_w_global_tensor_iterator_hacks);
        }
#endif
    }

    // pass tensor descriptor by reference
    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ void Run(const AGlobalDesc& a_k_m_global_desc,
                        const Float* __restrict__ p_a_global,
                        const BGlobalDesc& b_cyx_n_h_w_global_desc,
                        const Float* __restrict__ p_b_global,
                        const CGlobalDesc& c_k_n_h_w_global_desc,
                        Float* __restrict__ p_c_global,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        constexpr index_t shared_block_size = GetSharedMemoryNumberOfByte() / sizeof(Float);

        __shared__ Float p_shared_block[shared_block_size];

        Run(a_k_m_global_desc,
            p_a_global,
            b_cyx_n_h_w_global_desc,
            p_b_global,
            c_k_n_h_w_global_desc,
            p_c_global,
            p_shared_block,
            integral_constant<bool, HasMainKBlockLoop>{},
            integral_constant<bool, HasDoubleTailKBlockLoop>{});
    }

    // pass tensor descriptors by their pointers
    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ void Run(const AGlobalDesc* p_a_k_m_global_desc,
                        const Float* __restrict__ p_a_global,
                        const BGlobalDesc* p_b_cyx_n_h_w_global_desc,
                        const Float* __restrict__ p_b_global,
                        const CGlobalDesc* p_c_k_n_h_w_global_desc,
                        Float* __restrict__ p_c_global,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        const auto a_k_m_global_desc         = *p_a_k_m_global_desc;
        const auto b_cyx_n_h_w_global_desc         = *p_b_cyx_n_h_w_global_desc;
        const auto c_k_n_h_w_global_desc = *p_c_k_n_h_w_global_desc;

        Run(a_k_m_global_desc,
            p_a_global,
            b_cyx_n_h_w_global_desc,
            p_b_global,
            c_k_n_h_w_global_desc,
            p_c_global,
            integral_constant<bool, HasMainKBlockLoop>{},
            integral_constant<bool, HasDoubleTailKBlockLoop>{});
    }

    // pass tensor descriptors by void*
    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ void Run(const void* p_a_k_m_global_desc,
                        const Float* __restrict__ p_a_global,
                        const void* p_b_cyx_n_h_w_global_desc,
                        const Float* __restrict__ p_b_global,
                        const void* p_c_k_n_h_w_global_desc,
                        Float* __restrict__ p_c_global,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        const auto a_k_m_global_desc = *reinterpret_cast<const AGlobalDesc*>(p_a_k_m_global_desc);
        const auto b_cyx_n_h_w_global_desc = *reinterpret_cast<const BGlobalDesc*>(p_b_cyx_n_h_w_global_desc);
        const auto c_k_n_h_w_global_desc =
            *reinterpret_cast<const CGlobalDesc*>(p_c_k_n_h_w_global_desc);

        Run(a_k_m_global_desc,
            p_a_global,
            b_cyx_n_h_w_global_desc,
            p_b_global,
            c_k_n_h_w_global_desc,
            p_c_global,
            integral_constant<bool, HasMainKBlockLoop>{},
            integral_constant<bool, HasDoubleTailKBlockLoop>{});
    }
};

} // namespace ck
#endif
