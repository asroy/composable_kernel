#ifndef CK_GRIDWISE_GEMM_XDLOPS_FP16_BFP16_HPP
#define CK_GRIDWISE_GEMM_XDLOPS_FP16_BFP16_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "blockwise_generic_tensor_slice_copy_v2.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "blockwise_gemm_xdlops.hpp"

namespace ck {

enum WorkgroupScheduleOrder
{
    MBlock1NBlock0,
    NBlock1MBlock0
};

template <index_t Gi,
          index_t MBlockWork,
          index_t NBlockWork,
          WorkgroupScheduleOrder WorkgroupSchdOrder>
struct make_batch_block_work_sequence;

template <index_t Gi, index_t MBlockWork, index_t NBlockWork>
struct make_batch_block_work_sequence<Gi, MBlockWork, NBlockWork, MBlock1NBlock0>
{
    __device__ constexpr auto get() { return Sequence<Gi, MBlockWork, NBlockWork>{}; }
};

template <index_t Gi, index_t MBlockWork, index_t NBlockWork>
struct make_batch_block_work_sequence<Gi, MBlockWork, NBlockWork, NBlock1MBlock0>
{
    __device__ constexpr auto get() { return Sequence<Gi, NBlockWork, MBlockWork>{}; }
};

template <index_t MBlockWork, index_t NBlockWork, WorkgroupScheduleOrder WorkgroupSchdOrder>
struct make_block_work_sequence;

template <index_t MBlockWork, index_t NBlockWork>
struct make_block_work_sequence<MBlockWork, NBlockWork, MBlock1NBlock0>
{
    __device__ constexpr auto get() { return Sequence<MBlockWork, NBlockWork>{}; }
};

template <index_t MBlockWork, index_t NBlockWork>
struct make_block_work_sequence<MBlockWork, NBlockWork, NBlock1MBlock0>
{
    __device__ constexpr auto get() { return Sequence<NBlockWork, MBlockWork>{}; }
};

template <index_t GridSize,
          index_t BlockSize,
          class ABFloat,
          class AccFloat,
          class CFloat,
          class AGlobalDesc,
          class BGlobalDesc,
          class CGlobalDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWave,
          index_t NPerWave,
          class ABlockCopyThreadSliceLengths_G_K_M_KPACK,
          class ABlockCopyThreadClusterLengths_G_K_M_KPACK,
          class ABlockCopyThreadClusterArrangeOrder,
          class ABlockCopySrcAccessOrder,
          class ABlockCopyDstAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          index_t ABlockCopySrcDataPerRead,
          index_t ABlockCopyDstDataPerWrite_KPACK,
          class BBlockCopyThreadSliceLengths_G_K_N_KPACK,
          class BBlockCopyThreadClusterLengths_G_K_N_KPACK,
          class BBlockCopyThreadClusterArrangeOrder,
          class BBlockCopySrcAccessOrder,
          class BBlockCopyDstAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          index_t BBlockCopySrcDataPerRead,
          index_t BBlockCopyDstDataPerWrite_KPACK,
          InMemoryDataOperation CGlobalMemoryOp,
          WorkgroupScheduleOrder WorkgroupSchdOrder>
struct GridwiseBatchGemmXdlops_gkmkpack_gknkpack_gmn_v2_org
{
    __device__ void Run(const ABFloat* const __restrict__ p_a_global,
                        const ABFloat* const __restrict__ p_b_global,
                        CFloat* const __restrict__ p_c_global) const
    {
        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto a_g_k_m_kpack_global_desc = AGlobalDesc{};
        constexpr auto b_g_k_n_kpack_global_desc = BGlobalDesc{};
        constexpr auto c_g_m_n_global_desc       = CGlobalDesc{};

        constexpr auto G     = c_g_m_n_global_desc.GetLengths()[0];
        constexpr auto M     = c_g_m_n_global_desc.GetLengths()[1];
        constexpr auto N     = c_g_m_n_global_desc.GetLengths()[2];
        constexpr auto K     = b_g_k_n_kpack_global_desc.GetLengths()[1];
        constexpr auto KPack = b_g_k_n_kpack_global_desc.GetLengths()[3];

        // divide block work by [M, N]
        static_assert(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t MBlockWork = M / MPerBlock;
        constexpr index_t NBlockWork = N / NPerBlock;

        constexpr index_t MWavePerBlock = MPerBlock / MPerWave;
        constexpr index_t NWavePerBlock = NPerBlock / NPerWave;

        constexpr auto block_work_sequence =
            make_batch_block_work_sequence<G, MBlockWork, NBlockWork, WorkgroupSchdOrder>{}.get();
        constexpr auto block_work_desc = make_cluster_descriptor(block_work_sequence);

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t g_block_data_on_global = block_work_id[Number<0>{}];
        const index_t m_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[Number<1>{}] * MPerBlock)
                                                   : (block_work_id[Number<2>{}] * MPerBlock);
        const index_t n_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[Number<2>{}] * NPerBlock)
                                                   : (block_work_id[Number<1>{}] * NPerBlock);

        constexpr index_t max_align = KPack;

        //   LDS be careful of LDS alignment
        constexpr auto a_g_k_m_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, MPerBlock, KPack>{}, Number<max_align>{});

        auto a_blockwise_copy = BlockwiseGenericTensorSliceCopy_v4<
            BlockSize,
            decltype(a_g_k_m_kpack_global_desc),
            decltype(a_g_k_m_kpack_block_desc),
            decltype(a_g_k_m_kpack_block_desc.GetLengths()),
            ABlockCopyThreadSliceLengths_G_K_M_KPACK,
            ABlockCopyThreadClusterLengths_G_K_M_KPACK,
            ABlockCopyThreadClusterArrangeOrder,
            ABlockCopySrcAccessOrder,
            ABlockCopyDstAccessOrder,
            ABlockCopySrcVectorReadDim, // Src dim to be read in vector form
            3,                          // Dst dim to be written in vector form (KPack dimension)
            ABlockCopySrcDataPerRead,
            ABlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set>(
            make_multi_index(g_block_data_on_global, 0, m_block_data_on_global, 0),
            make_multi_index(0, 0, 0, 0));

        constexpr auto b_g_k_n_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, NPerBlock, KPack>{}, Number<max_align>{});

        // input blockwise copy
        auto b_blockwise_copy = BlockwiseGenericTensorSliceCopy_v4<
            BlockSize,
            decltype(b_g_k_n_kpack_global_desc),
            decltype(b_g_k_n_kpack_block_desc),
            decltype(b_g_k_n_kpack_block_desc.GetLengths()),
            BBlockCopyThreadSliceLengths_G_K_N_KPACK,
            BBlockCopyThreadClusterLengths_G_K_N_KPACK,
            BBlockCopyThreadClusterArrangeOrder,
            BBlockCopySrcAccessOrder,
            BBlockCopyDstAccessOrder,
            BBlockCopySrcVectorReadDim, // Src dim to be read in vector form
            3,                          // Dst dim to be written in vector form (KPack dimension)
            BBlockCopySrcDataPerRead,
            BBlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set>(
            make_multi_index(g_block_data_on_global, 0, n_block_data_on_global, 0),
            make_multi_index(0, 0, 0, 0));

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlocl, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //     register
        constexpr auto a_k_m_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<MPerBlock>{});
        constexpr auto b_k_n_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<NPerBlock>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops<
            BlockSize,
            decltype(a_k_m_block_mtx_desc),
            decltype(b_k_n_block_mtx_desc),
            ABFloat,
            MPerWave,
            NPerWave,
            MWavePerBlock,
            NWavePerBlock,
            1,
            1>{};

        constexpr index_t a_block_space =
            math::integer_least_multiple(a_g_k_m_kpack_block_desc.GetElementSpace(), max_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_g_k_n_kpack_block_desc.GetElementSpace(), max_align);

        __shared__ ABFloat p_a_block[a_block_space];
        __shared__ ABFloat p_b_block[b_block_space];

        // get zero-initialized output register of vector type
        constexpr index_t c_thread_size = MPerBlock * NPerBlock / BlockSize;
        auto c_thread_vec               = GetRegBuffer<AccFloat, c_thread_size>();

        // preload data into LDS
        {
            a_blockwise_copy.Run(p_a_global, p_a_block);
            b_blockwise_copy.Run(p_b_global, p_b_block);
        }

        constexpr auto blockwise_a_copy_src_step = Sequence<0, KPerBlock, 0, 0>{};
        constexpr auto blockwise_b_copy_src_step = Sequence<0, KPerBlock, 0, 0>{};

        // main body
        for(index_t k_block_data_begin = 0; k_block_data_begin < K - KPerBlock;
            k_block_data_begin += KPerBlock)
        {
            ABFloat p_a_thread_buffer[a_blockwise_copy.GetThreadBufferSize()];
            ABFloat p_b_thread_buffer[b_blockwise_copy.GetThreadBufferSize()];

            // load next data from device mem
            a_blockwise_copy.MoveSrcSliceWindow(blockwise_a_copy_src_step, True);
            b_blockwise_copy.MoveSrcSliceWindow(blockwise_b_copy_src_step, True);

            a_blockwise_copy.RunLoadThreadBuffer(p_a_global, p_a_thread_buffer);
            b_blockwise_copy.RunLoadThreadBuffer(p_b_global, p_b_thread_buffer);

            block_sync_lds();

            // GEMM on current data
            const typename vector_type<ABFloat, KPack>::MemoryType* p_a_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_a_block);
            const typename vector_type<ABFloat, KPack>::MemoryType* p_b_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_b_block);

            c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);

            block_sync_lds();

            // store next data to LDS
            a_blockwise_copy.RunStoreThreadBuffer(p_a_thread_buffer, p_a_block);
            b_blockwise_copy.RunStoreThreadBuffer(p_b_thread_buffer, p_b_block);
        }

        // tail
        {
            block_sync_lds();

            // GEMM on last data
            const typename vector_type<ABFloat, KPack>::MemoryType* p_a_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_a_block);
            const typename vector_type<ABFloat, KPack>::MemoryType* p_b_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_b_block);

            c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);
        }

        // copy output: register to global memory
        {
            ///\todo inconsistent layout of xdlops and tensor
            // xdlops layout
            // M1 = num_groups;
            // M0 = group_size;
            // N1 = num_blks_per_wave;
            // N0 = num_threads_per_blks;
            constexpr auto CLayout = blockwise_gemm.GetOutputLayout();
            constexpr index_t M0   = CLayout.M1();
            constexpr index_t M1   = CLayout.N1();
            constexpr index_t M2   = CLayout.M0();

            constexpr auto c_g_m0_m1_m2_n_global_desc = transform_tensor_descriptor(
                c_g_m_n_global_desc,
                make_tuple(
                    PassThrough<G>{}, UnMerge<Sequence<M / (M1 * M2), M1, M2>>{}, PassThrough<N>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}));

            //     src descriptor
            constexpr auto c_g_m0_m1_m2_n_thread_desc =
                make_native_tensor_descriptor_packed(Sequence<1, M0, 1, M2, 1>{});

            using CThreadCopySliceLengths = Sequence<1, M0, 1, M2, 1>;

            constexpr index_t BlkSize = blockwise_gemm.GetBlkSize();
            constexpr index_t NumBlks = blockwise_gemm.GetNumBlks();

// force unrolling the output loop to get ride of scratches
#pragma unroll
            for(index_t i = 0; i < NumBlks; ++i)
            {
                // calculate origin of thread output tensor on global memory
                //     blockwise GEMM c matrix starting index
                const auto c_thread_mtx_on_block = blockwise_gemm.GetBeginOfThreadMatrixC(i);

                const index_t m_thread_data_on_global =
                    m_block_data_on_global + c_thread_mtx_on_block.row;

                const index_t n_thread_data_on_global =
                    n_block_data_on_global + c_thread_mtx_on_block.col;

                ThreadwiseGenericTensorSliceCopy_v4r2<decltype(c_g_m0_m1_m2_n_thread_desc),
                                                      decltype(c_g_m0_m1_m2_n_global_desc),
                                                      CThreadCopySliceLengths,
                                                      arithmetic_sequence_gen<0, 5, 1>::type,
                                                      4,
                                                      1,
                                                      1,
                                                      AddressSpace::Vgpr,
                                                      AddressSpace::Global,
                                                      CGlobalMemoryOp>(
                    make_multi_index(0, 0, 0, 0, 0),
                    make_multi_index(g_block_data_on_global,
                                     m_thread_data_on_global / (M2 * M1),
                                     m_thread_data_on_global % (M2 * M1) / M2,
                                     m_thread_data_on_global % M2,
                                     n_thread_data_on_global))
                    .Run(c_thread_vec.n + i * BlkSize, p_c_global);
            }
        }
    }
};

template <index_t GridSize,
          index_t BlockSize,
          class ABFloat,
          class AccFloat,
          class CFloat,
          class AGlobalDesc,
          class BGlobalDesc,
          class CGlobalDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWave,
          index_t NPerWave,
          class ABlockCopyThreadSliceLengths_G_K_M_KPACK,
          class ABlockCopyThreadClusterLengths_G_K_M_KPACK,
          class ABlockCopyThreadClusterArrangeOrder,
          class ABlockCopySrcAccessOrder,
          class ABlockCopyDstAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          index_t ABlockCopySrcDataPerRead,
          index_t ABlockCopyDstDataPerWrite_KPACK,
          class BBlockCopyThreadSliceLengths_G_K_N_KPACK,
          class BBlockCopyThreadClusterLengths_G_K_N_KPACK,
          class BBlockCopyThreadClusterArrangeOrder,
          class BBlockCopySrcAccessOrder,
          class BBlockCopyDstAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          index_t BBlockCopySrcDataPerRead,
          index_t BBlockCopyDstDataPerWrite_KPACK,
          InMemoryDataOperation CGlobalMemoryOp,
          WorkgroupScheduleOrder WorkgroupSchdOrder>
struct GridwiseBatchGemmXdlops_gkmkpack_gknkpack_gmn_v2
{
    __device__ void Run(const ABFloat* const __restrict__ p_a_global,
                        const ABFloat* const __restrict__ p_b_global,
                        CFloat* const __restrict__ p_c_global) const
    {
        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto a_g_k_m_kpack_global_desc = AGlobalDesc{};
        constexpr auto b_g_k_n_kpack_global_desc = BGlobalDesc{};
        constexpr auto c_g_m_n_global_desc       = CGlobalDesc{};

        constexpr auto G     = c_g_m_n_global_desc.GetLengths()[0];
        constexpr auto M     = c_g_m_n_global_desc.GetLengths()[1];
        constexpr auto N     = c_g_m_n_global_desc.GetLengths()[2];
        constexpr auto K     = b_g_k_n_kpack_global_desc.GetLengths()[1];
        constexpr auto KPack = b_g_k_n_kpack_global_desc.GetLengths()[3];

        // divide block work by [M, N]
        static_assert(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t MBlockWork = M / MPerBlock;
        constexpr index_t NBlockWork = N / NPerBlock;

        constexpr index_t MWavePerBlock = MPerBlock / MPerWave;
        constexpr index_t NWavePerBlock = NPerBlock / NPerWave;

        constexpr auto block_work_sequence =
            make_batch_block_work_sequence<G, MBlockWork, NBlockWork, WorkgroupSchdOrder>{}.get();
        constexpr auto block_work_desc = make_cluster_descriptor(block_work_sequence);

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t g_block_data_on_global = block_work_id[Number<0>{}];
        const index_t m_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[Number<1>{}] * MPerBlock)
                                                   : (block_work_id[Number<2>{}] * MPerBlock);
        const index_t n_block_data_on_global = (WorkgroupSchdOrder == MBlock1NBlock0)
                                                   ? (block_work_id[Number<2>{}] * NPerBlock)
                                                   : (block_work_id[Number<1>{}] * NPerBlock);

        constexpr index_t max_align = KPack;

        //   LDS be careful of LDS alignment
        constexpr auto a_g_k_m_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, MPerBlock, KPack>{}, Number<max_align>{});

        auto a_blockwise_copy = BlockwiseGenericTensorSliceCopy_v5<
            BlockSize,
            decltype(a_g_k_m_kpack_global_desc),
            decltype(a_g_k_m_kpack_block_desc),
            decltype(a_g_k_m_kpack_block_desc.GetLengths()),
            ABlockCopyThreadSliceLengths_G_K_M_KPACK,
            ABlockCopyThreadClusterLengths_G_K_M_KPACK,
            ABlockCopyThreadClusterArrangeOrder,
            ABlockCopySrcAccessOrder,
            ABlockCopyDstAccessOrder,
            ABlockCopySrcVectorReadDim, // Src dim to be read in vector form
            3,                          // Dst dim to be written in vector form (KPack dimension)
            ABlockCopySrcDataPerRead,
            ABlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set>(
            make_multi_index(g_block_data_on_global, 0, m_block_data_on_global, 0),
            make_multi_index(0, 0, 0, 0));

        constexpr auto b_g_k_n_kpack_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<1, KPerBlock, NPerBlock, KPack>{}, Number<max_align>{});

        // input blockwise copy
        auto b_blockwise_copy = BlockwiseGenericTensorSliceCopy_v5<
            BlockSize,
            decltype(b_g_k_n_kpack_global_desc),
            decltype(b_g_k_n_kpack_block_desc),
            decltype(b_g_k_n_kpack_block_desc.GetLengths()),
            BBlockCopyThreadSliceLengths_G_K_N_KPACK,
            BBlockCopyThreadClusterLengths_G_K_N_KPACK,
            BBlockCopyThreadClusterArrangeOrder,
            BBlockCopySrcAccessOrder,
            BBlockCopyDstAccessOrder,
            BBlockCopySrcVectorReadDim, // Src dim to be read in vector form
            3,                          // Dst dim to be written in vector form (KPack dimension)
            BBlockCopySrcDataPerRead,
            BBlockCopyDstDataPerWrite_KPACK,
            AddressSpace::Global,
            AddressSpace::Vgpr,
            AddressSpace::Lds,
            InMemoryDataOperation::Set>(
            make_multi_index(g_block_data_on_global, 0, n_block_data_on_global, 0),
            make_multi_index(0, 0, 0, 0));

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlock] is in LDS
        //     b_mtx[KPerBlocl, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //     register
        constexpr auto a_k_m_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<MPerBlock>{});
        constexpr auto b_k_n_block_mtx_desc =
            make_ConstantMatrixDescriptor_packed(Number<KPerBlock>{}, Number<NPerBlock>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_xdlops<
            BlockSize,
            decltype(a_k_m_block_mtx_desc),
            decltype(b_k_n_block_mtx_desc),
            ABFloat,
            MPerWave,
            NPerWave,
            MWavePerBlock,
            NWavePerBlock,
            1,
            1>{};

        constexpr index_t a_block_space =
            math::integer_least_multiple(a_g_k_m_kpack_block_desc.GetElementSpace(), max_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_g_k_n_kpack_block_desc.GetElementSpace(), max_align);

        __shared__ ABFloat p_a_block[a_block_space];
        __shared__ ABFloat p_b_block[b_block_space];

        // get zero-initialized output register of vector type
        // auto c_thread_vec = blockwise_gemm.CreateOutputVecZero();
        constexpr index_t c_thread_size = MPerBlock * NPerBlock / BlockSize;
        auto c_thread_vec               = GetRegBuffer<AccFloat, c_thread_size>();

        // preload data into LDS
        {
            a_blockwise_copy.Run(p_a_global, p_a_block);
            b_blockwise_copy.Run(p_b_global, p_b_block);
        }

        constexpr auto blockwise_a_copy_src_step = Sequence<0, KPerBlock, 0, 0>{};
        constexpr auto blockwise_b_copy_src_step = Sequence<0, KPerBlock, 0, 0>{};

        // main body
        for(index_t k_block_data_begin = 0; k_block_data_begin < K - KPerBlock;
            k_block_data_begin += KPerBlock)
        {
            // load next data from device mem
            a_blockwise_copy.MoveSrcSliceWindow(blockwise_a_copy_src_step, True);
            b_blockwise_copy.MoveSrcSliceWindow(blockwise_b_copy_src_step, True);

            a_blockwise_copy.RunLoadThreadBuffer(p_a_global);
            b_blockwise_copy.RunLoadThreadBuffer(p_b_global);

            block_sync_lds();

            // GEMM on current data
            const typename vector_type<ABFloat, KPack>::MemoryType* p_a_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_a_block);
            const typename vector_type<ABFloat, KPack>::MemoryType* p_b_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_b_block);

            c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);

            block_sync_lds();

            // store next data to LDS
            a_blockwise_copy.RunStoreThreadBuffer(p_a_block);
            b_blockwise_copy.RunStoreThreadBuffer(p_b_block);
        }

        // tail
        {
            block_sync_lds();

            // GEMM on last data
            const typename vector_type<ABFloat, KPack>::MemoryType* p_a_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_a_block);
            const typename vector_type<ABFloat, KPack>::MemoryType* p_b_block_vec =
                reinterpret_cast<const typename vector_type<ABFloat, KPack>::MemoryType*>(
                    p_b_block);

            c_thread_vec = blockwise_gemm.Run(p_a_block_vec, p_b_block_vec, c_thread_vec);
        }

        // copy output: register to global memory
        {
            ///\todo inconsistent layout of xdlops and tensor
            // xdlops layout
            // M1 = num_groups;
            // M0 = group_size;
            // N1 = num_blks_per_wave;
            // N0 = num_threads_per_blks;
            constexpr auto CLayout = blockwise_gemm.GetOutputLayout();
            constexpr index_t M0   = CLayout.M1();
            constexpr index_t M1   = CLayout.N1();
            constexpr index_t M2   = CLayout.M0();

            constexpr auto c_g_m0_m1_m2_n_global_desc = transform_tensor_descriptor(
                c_g_m_n_global_desc,
                make_tuple(
                    PassThrough<G>{}, UnMerge<Sequence<M / (M1 * M2), M1, M2>>{}, PassThrough<N>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}));

            //     src descriptor
            constexpr auto c_g_m0_m1_m2_n_thread_desc =
                make_native_tensor_descriptor_packed(Sequence<1, M0, 1, M2, 1>{});

            using CThreadCopySliceLengths = Sequence<1, M0, 1, M2, 1>;

            constexpr index_t BlkSize = blockwise_gemm.GetBlkSize();
            constexpr index_t NumBlks = blockwise_gemm.GetNumBlks();

            // force unrolling the output loop to get ride of scratches
            static_for<0, NumBlks, 1>{}([&](auto blk_id) {
                // calculate origin of thread output tensor on global memory
                //     blockwise GEMM c matrix starting index
                const auto c_thread_mtx_on_block = blockwise_gemm.GetBeginOfThreadMatrixC(blk_id);

                const index_t m_thread_data_on_global =
                    m_block_data_on_global + c_thread_mtx_on_block.row;

                const index_t n_thread_data_on_global =
                    n_block_data_on_global + c_thread_mtx_on_block.col;

                ThreadwiseGenericTensorSliceCopy_v5<decltype(c_g_m0_m1_m2_n_thread_desc),
                                                    decltype(c_g_m0_m1_m2_n_global_desc),
                                                    CThreadCopySliceLengths,
                                                    arithmetic_sequence_gen<0, 5, 1>::type,
                                                    arithmetic_sequence_gen<0, 5, 1>::type,
                                                    4,
                                                    4,
                                                    1,
                                                    1,
                                                    AddressSpace::Vgpr,
                                                    AddressSpace::Global,
                                                    CGlobalMemoryOp>(
                    make_multi_index(0, 0, 0, 0, 0),
                    make_multi_index(g_block_data_on_global,
                                     m_thread_data_on_global / (M2 * M1),
                                     m_thread_data_on_global % (M2 * M1) / M2,
                                     m_thread_data_on_global % M2,
                                     n_thread_data_on_global))
                    .GlobalStore(c_thread_vec, p_c_global);
                //.GlobalStore(c_thread_vec.GetVector(Number<BlkSize>{})[Number<blk_id>{}],
                // p_c_global);
            });
        }
    }
};

} // namespace ck
#endif
