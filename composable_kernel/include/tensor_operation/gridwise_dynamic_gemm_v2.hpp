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
          index_t KPerBlock,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t CYXPerBlock,
          index_t KPerThread,
          index_t HPerThread,
          index_t WPerThread,
          index_t CYXPerThread,
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
        constexpr auto max_lds_align =
            math::lcm(Number<ABlockTransferDstScalarPerVector_M>{}, Number<KPerThread>{});

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_cyx_k_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<CYXPerBlock>{}, Number<KPerBlock>{}), max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_cyx_k_block_desc.GetElementSpaceSize(), max_lds_align);

        return 2 * (a_block_space_size) * sizeof(Float);
    }

    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ void Run(const AGlobalDesc& a_cyx_k_global_desc,
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
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        const auto CYX = a_cyx_k_global_desc.GetLength(I0);
        const auto K   = a_cyx_k_global_desc.GetLength(I1);

        const auto N = b_cyx_n_h_w_global_desc.GetLength(I1);
        const auto H = b_cyx_n_h_w_global_desc.GetLength(I2);
        const auto W = b_cyx_n_h_w_global_desc.GetLength(I3);

        // divide block work by [M, N]
#if 1
        const auto m_block_work_num  = K / Number<KPerBlock>{};
        const auto h_block_work_num  = H / Number<HPerBlock>{};
        const auto w_block_work_num  = W / Number<WPerBlock>{};
        const auto hw_block_work_num = h_block_work_num * w_block_work_num;

        const index_t k_block_work_id  = get_block_1d_id() / hw_block_work_num;
        const index_t hw_block_work_id = get_block_1d_id() - k_block_work_id * hw_block_work_num;

#else
        // Hack: this force result into SGPR
        const index_t m_block_work_num  = __builtin_amdgcn_readfirstlane(K / KPerBlock);
        const index_t h_block_work_num  = __builtin_amdgcn_readfirstlane(H / HPerBlock);
        const index_t w_block_work_num  = __builtin_amdgcn_readfirstlane(W / WPerBlock);
        const index_t hw_block_work_num = h_block_work_num * w_block_work_num;

        const index_t k_block_work_id =
            __builtin_amdgcn_readfirstlane(get_block_1d_id() / hw_block_work_num);
        const index_t hw_block_work_id = get_block_1d_id() - k_block_work_id * hw_block_work_num;
#endif

        const index_t h_block_work_id = hw_block_work_id / w_block_work_num;
        const index_t w_block_work_id = hw_block_work_id - h_block_work_id * w_block_work_num;

        constexpr auto h_num_threads = HPerBlock / HPerThread;
        constexpr auto w_num_threads = WPerBlock / WPerThread;

        static_assert(KPerBlock == KPerThread, "");

        const auto k_thread_id = 0;
        const auto h_thread_id = get_thread_local_1d_id() / w_num_threads;
        const auto w_thread_id = get_thread_local_1d_id() % w_num_threads;

        const index_t k_block_data_on_global = k_block_work_id * KPerBlock;
        const index_t h_block_data_on_global = h_block_work_id * HPerBlock;
        const index_t w_block_data_on_global = w_block_work_id * WPerBlock;

        const index_t k_thread_data_on_global = k_block_data_on_global + k_thread_id * KPerThread;
        const index_t h_thread_data_on_global = h_block_data_on_global + h_thread_id * HPerThread;
        const index_t w_thread_data_on_global = w_block_data_on_global + w_thread_id * WPerThread;

        // lds max alignment
        constexpr auto max_lds_align =
            math::lcm(Number<ABlockTransferDstScalarPerVector_M>{}, Number<KPerThread>{});

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_cyx_k_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<CYXPerBlock>{}, Number<KPerBlock>{}), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_cyx_n_h_w_block_desc =
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
                Number<CYXPerBlock>{}, Number<1>{}, Number<HPerBlock>{}, Number<WPerBlock>{}));

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseDynamicTensorSliceTransfer_v4<BlockSize,
                                                   InMemoryDataOperation::Set,
                                                   Sequence<CYXPerBlock, KPerBlock>,
                                                   ABlockTransferThreadSliceLengths_K_M,
                                                   ABlockTransferThreadClusterLengths_K_M,
                                                   ABlockTransferThreadClusterArrangeOrder,
                                                   Float,
                                                   Float,
                                                   decltype(a_cyx_k_global_desc),
                                                   decltype(a_cyx_k_block_desc),
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
                a_cyx_k_global_desc,
                make_multi_index(0, k_block_data_on_global),
                a_cyx_k_block_desc,
                make_multi_index(0, 0));

        constexpr auto b_cyx_n_h_w_thread_desc =
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
                Number<CYXPerBlock>{}, Number<1>{}, Number<HPerThread>{}, Number<WPerThread>{}));

        using ThreadwiseTensorSliceTransferB = ThreadwiseDynamicTensorSliceTransfer_v2<
            Float,
            Float,
            decltype(b_cyx_n_h_w_global_desc),
            decltype(b_cyx_n_h_w_thread_desc),
            Sequence<CYXPerBlock, 1, HPerThread, WPerThread>,
            Sequence<3, 2, 0, 1>, // BBlockTransferSrcAccessOrder,
            3,                    // BBlockTransferSrcVectorDim,
            1,                    // BBlockTransferSrcScalarPerVector,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            InMemoryDataOperation::Set,
            1,
            true>;

        ThreadwiseTensorSliceTransferB b_threadwise_transfer(
            b_cyx_n_h_w_global_desc,
            make_multi_index(0, 0, h_thread_data_on_global, w_thread_data_on_global));

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_k_n_h_w_thread_desc =
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
                Number<KPerThread>{}, Number<1>{}, Number<HPerThread>{}, Number<WPerThread>{}));

#if 1
        const auto blockwise_gemm =
            BlockwiseGemm_km_kn_m0m1n0n1_v3<BlockSize,
                                            decltype(a_cyx_k_block_desc),
                                            decltype(b_cyx_n_h_w_block_desc),
                                            decltype(c_k_n_h_w_thread_desc),
                                            KPerThread,    // KPerThreadSubC
                                            HPerThread,    // HPerThreadSubC
                                            WPerThread,    // WPerThreadSubC
                                            CYXPerThread,  // CYXPerThreadLoop
                                            h_num_threads, // HThreadCluster
                                            w_num_threads, // WThreadCluster
                                            1,             // ThreadGemmADataPerRead_K
                                            1              // ThreadGemmBDataPerRead_W
                                            >{};
#endif

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_cyx_k_block_desc.GetElementSpaceSize(), max_lds_align);

        Float* p_a_block_double = p_shared_block;

        // register allocation for output
        AccFloat p_c_thread[c_k_n_h_w_thread_desc.GetElementSpaceSize()];

        // zero out threadwise output
        threadwise_matrix_set_zero_v3(c_k_n_h_w_thread_desc, p_c_thread);

        constexpr auto a_block_slice_copy_step  = make_multi_index(CYXPerBlock, 0);
        constexpr auto b_thread_slice_copy_step = make_multi_index(CYXPerBlock, 0, 0, 0);

        // hack to control index calculation when iterating over A and B matrix for threadwise copy
        constexpr auto a_k_m_global_iterator_hacks       = AGlobalIteratorHacks{};
        constexpr auto b_cyx_n_h_w_global_iterator_hacks = BGlobalIteratorHacks{};

        // hack to control index calculation when move slice window for A and B matrix for
        // threadwise copy
        constexpr auto a_k_m_global_move_slice_window_iterator_hack =
            AGlobalMoveSliceWindowIteratorHacks{};
        constexpr auto b_cyx_n_h_w_global_move_slice_window_iterator_hack =
            BGlobalMoveSliceWindowIteratorHacks{};

        constexpr auto b_thread_space_size = b_cyx_n_h_w_thread_desc.GetElementSpaceSize();
        Float p_b_thread[b_thread_space_size * 2];

        Float* p_b_thread_double = p_b_thread;

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.RunRead(a_cyx_k_global_desc, p_a_global, a_k_m_global_iterator_hacks);

            b_threadwise_transfer.Run(b_cyx_n_h_w_global_desc,
                                      p_b_global,
                                      b_cyx_n_h_w_thread_desc,
                                      make_tuple(I0, I0, I0, I0),
                                      p_b_thread_double,
                                      b_cyx_n_h_w_global_iterator_hacks);

            a_blockwise_copy.RunWrite(a_cyx_k_block_desc, p_a_block_double);

#if 0
            __syncthreads();

            p_c_thread[0] += p_b_thread_double[0] + p_b_thread_double[1] + p_b_thread_double[2];
            p_c_thread[0] += p_b_thread_double[3] + p_b_thread_double[4] + p_b_thread_double[5];
            p_c_thread[0] += p_b_thread_double[6] + p_b_thread_double[7] + p_b_thread_double[8];
#endif
        }

#if 1
        if constexpr(HasMainKBlockLoop)
        {
            Float* p_a_block_even = p_a_block_double;
            Float* p_a_block_odd  = p_a_block_double + a_block_space_size;

            Float* p_b_thread_even = p_b_thread_double;
            Float* p_b_thread_odd  = p_b_thread_double + b_thread_space_size;

            index_t b_block_data_begin = 0;

            // LDS double buffer: main body
            // use Do-While loop instead of For loop to simplify control flow
            do
            {
                // even iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_cyx_k_global_desc,
                                                    a_block_slice_copy_step,
                                                    a_k_m_global_move_slice_window_iterator_hack);

                b_threadwise_transfer.MoveSrcSliceWindow(b_cyx_n_h_w_global_desc,
                                                         b_thread_slice_copy_step);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_cyx_k_global_desc, p_a_global, a_k_m_global_iterator_hacks);

                b_threadwise_transfer.Run(b_cyx_n_h_w_global_desc,
                                          p_b_global,
                                          b_cyx_n_h_w_thread_desc,
                                          make_tuple(I0, I0, I0, I0),
                                          p_b_thread_odd,
                                          b_cyx_n_h_w_global_iterator_hacks);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(p_a_block_even, p_b_thread_even, p_c_thread);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_cyx_k_block_desc, p_a_block_odd);

                // odd iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_cyx_k_global_desc,
                                                    a_block_slice_copy_step,
                                                    a_k_m_global_move_slice_window_iterator_hack);

                b_threadwise_transfer.MoveSrcSliceWindow(b_cyx_n_h_w_global_desc,
                                                         b_thread_slice_copy_step);
                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_cyx_k_global_desc, p_a_global, a_k_m_global_iterator_hacks);

                b_threadwise_transfer.Run(b_cyx_n_h_w_global_desc,
                                          p_b_global,
                                          b_cyx_n_h_w_thread_desc,
                                          make_tuple(I0, I0, I0, I0),
                                          p_b_thread_even,
                                          b_cyx_n_h_w_global_iterator_hacks);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(p_a_block_odd, p_b_thread_odd, p_c_thread);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_cyx_k_block_desc, p_a_block_even);

                b_block_data_begin += 2 * CYXPerBlock;
            } while(b_block_data_begin < CYX - 2 * CYXPerBlock);
        }

        // LDS double buffer: tail
        if constexpr(HasDoubleTailKBlockLoop) // if has 2 iteration left
        {
            a_blockwise_copy.MoveSrcSliceWindow(a_cyx_k_global_desc,
                                                a_block_slice_copy_step,
                                                a_k_m_global_move_slice_window_iterator_hack);

            b_threadwise_transfer.MoveSrcSliceWindow(b_cyx_n_h_w_global_desc,
                                                     b_thread_slice_copy_step);

            __syncthreads();

            // LDS double buffer: load last data from device mem
            a_blockwise_copy.RunRead(a_cyx_k_global_desc, p_a_global, a_k_m_global_iterator_hacks);

            b_threadwise_transfer.Run(b_cyx_n_h_w_global_desc,
                                      p_b_global,
                                      b_cyx_n_h_w_thread_desc,
                                      make_tuple(I0, I0, I0, I0),
                                      p_b_thread_double + b_thread_space_size,
                                      b_cyx_n_h_w_global_iterator_hacks);

            // LDS double buffer: GEMM on 2nd-last data
            blockwise_gemm.Run(p_a_block_double, p_b_thread_double, p_c_thread);

            // LDS double buffer: store last data to LDS
            a_blockwise_copy.RunWrite(a_cyx_k_block_desc, p_a_block_double + a_block_space_size);

            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(p_a_block_double + a_block_space_size,
                               p_b_thread_double + b_thread_space_size,
                               p_c_thread);
        }
        else // if has 1 iteration left
        {
            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(p_a_block_double, p_b_thread_double, p_c_thread);
        }
#endif

#if 1
        // output: register to global memory
        {
            // define input tensor descriptor for threadwise copy
            //     thread input tensor, src of threadwise copy
            constexpr auto c_k_n_h_w_thread_desc =
                make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
                    Number<KPerThread>{}, Number<1>{}, Number<HPerThread>{}, Number<WPerThread>{}));

            // hack to control index calculation when iterating over c_k_n_h_w_global tensor
            constexpr auto c_k_n_h_w_global_tensor_iterator_hacks = CGlobalIteratorHacks{};

            ThreadwiseDynamicTensorSliceTransfer_v1r3<
                AccFloat,
                Float,
                decltype(c_k_n_h_w_thread_desc),
                decltype(c_k_n_h_w_global_desc),
                Sequence<KPerThread, 1, HPerThread, WPerThread>,
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
                    k_thread_data_on_global, 0, h_thread_data_on_global, w_thread_data_on_global))
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
    __device__ void Run(const AGlobalDesc& a_cyx_k_global_desc,
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

        Run(a_cyx_k_global_desc,
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
    __device__ void Run(const AGlobalDesc* p_a_cyx_k_global_desc,
                        const Float* __restrict__ p_a_global,
                        const BGlobalDesc* p_b_cyx_n_h_w_global_desc,
                        const Float* __restrict__ p_b_global,
                        const CGlobalDesc* p_c_k_n_h_w_global_desc,
                        Float* __restrict__ p_c_global,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        const auto a_cyx_k_global_desc     = *p_a_cyx_k_global_desc;
        const auto b_cyx_n_h_w_global_desc = *p_b_cyx_n_h_w_global_desc;
        const auto c_k_n_h_w_global_desc   = *p_c_k_n_h_w_global_desc;

        Run(a_cyx_k_global_desc,
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
    __device__ void Run(const void* p_a_cyx_k_global_desc,
                        const Float* __restrict__ p_a_global,
                        const void* p_b_cyx_n_h_w_global_desc,
                        const Float* __restrict__ p_b_global,
                        const void* p_c_k_n_h_w_global_desc,
                        Float* __restrict__ p_c_global,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        const auto a_cyx_k_global_desc =
            *reinterpret_cast<const AGlobalDesc*>(p_a_cyx_k_global_desc);
        const auto b_cyx_n_h_w_global_desc =
            *reinterpret_cast<const BGlobalDesc*>(p_b_cyx_n_h_w_global_desc);
        const auto c_k_n_h_w_global_desc =
            *reinterpret_cast<const CGlobalDesc*>(p_c_k_n_h_w_global_desc);

        Run(a_cyx_k_global_desc,
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
