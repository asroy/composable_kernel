#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V3_NCHW_CYXK_NKHW_LDS_DOUBLE_BUFFER
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V3_NCHW_CYXK_NKHW_LDS_DOUBLE_BUFFER

#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "ConstantMergedTensorDescriptor_deprecated.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "blockwise_gemm.hpp"

namespace ck {

// define B = merge(N0, Ho, Wo)
template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          index_t BPerBlock,
          index_t KPerBlock,
          index_t CPerBlock,
          index_t N1,
          index_t N2,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class InBlockCopySubLengths_C_N1_B_N2,
          class InBlockCopyClusterLengths_C_N1_B_N2,
          index_t InBlockCopySrcDataPerRead_B,
          index_t InBlockCopyDstDataPerWrite_N2,
          class WeiBlockCopySubLengths_C_K,
          class WeiBlockCopyClusterLengths_C_K,
          index_t WeiBlockCopyDataPerAccess_K>
struct GridwiseConvolutionImplicitGemm_v3_nchw_cyxk_nkhw_lds_double_buffer
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        // this is a mess
        // TODO: find more elegent way of specifying (or calculating) performance parameters
        static_assert(N2 == GemmNPerThreadSubC, "wrong!");
        static_assert((N1 * N2 * BPerBlock) %
                              (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};
        constexpr auto I5 = Number<5>{};
        constexpr auto I6 = Number<6>{};
        constexpr auto I7 = Number<7>{};

        constexpr auto in_n_c_h_w_global_desc  = InGlobalDesc{};
        constexpr auto wei_c_y_x_k_global_desc = WeiGlobalDesc{};
        constexpr auto out_n_k_h_w_global_desc = OutGlobalDesc{};

        constexpr index_t N  = in_n_c_h_w_global_desc.GetLength(I0);
        constexpr index_t C  = in_n_c_h_w_global_desc.GetLength(I1);
        constexpr index_t Hi = in_n_c_h_w_global_desc.GetLength(I2);
        constexpr index_t Wi = in_n_c_h_w_global_desc.GetLength(I3);

        constexpr index_t K  = out_n_k_h_w_global_desc.GetLength(I1);
        constexpr index_t Ho = out_n_k_h_w_global_desc.GetLength(I2);
        constexpr index_t Wo = out_n_k_h_w_global_desc.GetLength(I3);

        constexpr index_t Y = wei_c_y_x_k_global_desc.GetLength(I1);
        constexpr index_t X = wei_c_y_x_k_global_desc.GetLength(I2);

        static_assert(N % (N1 * N2) == 0, "wrong! cannot divice N evenly among thread");

        constexpr index_t N0 = N / (N1 * N2);

        constexpr index_t B = N0 * Ho * Wo;

        // divide block work by [K, B]
        static_assert(K % KPerBlock == 0 && B % BPerBlock == 0 && C % (2 * CPerBlock) == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t KBlockWork = K / KPerBlock;
        constexpr index_t BBlockWork = B / BPerBlock;

        constexpr auto block_work_desc =
            make_ConstantTensorDescriptor_packed(Sequence<KBlockWork, BBlockWork>{});

        const auto block_work_multi_id =
            block_work_desc.GetMultiIndexFrom1dIndex(get_block_1d_id());

        const index_t k_block_data_on_global = block_work_multi_id[0] * KPerBlock;
        const index_t b_block_data_on_global = block_work_multi_id[1] * BPerBlock;

        // input tensor
        //     memory layout descriptor in device memory [N0, N1, N2, C, H, W]
        constexpr auto in_n0_n1_n2_c_h_w_global_mem_desc =
            in_n_c_h_w_global_desc.Fold(I0, Number<N1>{}, Number<N2>{});

        //     merged tensor descriptor in device memory [C, N1, B, N2], src of blockwise copy
        constexpr auto in_c_n1_b_n2_global_merged_desc = make_ConstantMergedTensorDescriptor(
            in_n0_n1_n2_c_h_w_global_mem_desc.Slice(I4, Number<Ho>{}).Slice(I5, Number<Wo>{}),
            Sequence<3>{},
            Sequence<1>{},
            Sequence<0, 4, 5>{},
            Sequence<2>{});

        //     memory layout descriptor in LDS [C, N1, B, N2], dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto in_c_n1_b_n2_block_mem_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, N1, BPerBlock, N2>{}, Number<InBlockCopyDstDataPerWrite_N2>{});

        //     this check is ad-hoc
        //     TODO: need to properly implement tensor descriptor with alignment
        static_assert(in_c_n1_b_n2_block_mem_desc.GetStride(I1) % GemmDataPerReadB == 0,
                      "GemmDataPerReadB alignment requirement is not satisfied");

        // input blockwise copy
        //     slice a merged tensor, reorder and copy to a normal tensor
        //     this copy operator already has blockwise offset built-in
        const auto blockwise_in_copy = BlockwiseGenericTensorSliceCopy_v1_deprecated<
            BlockSize,
            Float,
            decltype(in_c_n1_b_n2_global_merged_desc),
            decltype(in_c_n1_b_n2_block_mem_desc),
            decltype(in_c_n1_b_n2_block_mem_desc.GetLengths()),
            InBlockCopySubLengths_C_N1_B_N2,
            InBlockCopyClusterLengths_C_N1_B_N2,
            Sequence<0, 1, 3, 2>, // thread_arrange_order [C, N1, N2, B]
            Sequence<1, 3, 0, 2>, // src_access_order [N1, N2, C, B]
            Sequence<0, 1, 2, 3>, // dst_access_order [C, N1, B, N2]
            InBlockCopySrcDataPerRead_B,
            InBlockCopyDstDataPerWrite_N2>({0, 0, b_block_data_on_global, 0}, {0, 0, 0, 0});

        // weight tensor
        //     tensor descriptor in device memory, src of blockwise copy
        constexpr auto wei_c_k_global_desc = wei_c_y_x_k_global_desc.Extract(I0, I3);

        //     tensor descriptor in LDS, dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto wei_c_k_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, KPerBlock>{},
            Number<math::lcm(WeiBlockCopyDataPerAccess_K, GemmDataPerReadA)>{});

        // operator for blockwise copy of weight into LDS
        //     slice a tensor, and copy it into another tensor
        //     this copy operator already have blockwise offset built-in
        const auto blockwise_wei_copy = BlockwiseGenericTensorSliceCopy_v1_deprecated<
            BlockSize,
            Float,
            decltype(wei_c_k_global_desc),
            decltype(wei_c_k_block_desc),
            decltype(wei_c_k_block_desc.GetLengths()),
            WeiBlockCopySubLengths_C_K,
            WeiBlockCopyClusterLengths_C_K,
            Sequence<0, 1>, // thread_arrange_order [C, K]
            Sequence<0, 1>, // src_access_order [C, K]
            Sequence<0, 1>, // dst_access_order [C, K]
            WeiBlockCopyDataPerAccess_K,
            WeiBlockCopyDataPerAccess_K>({0, k_block_data_on_global}, {0, 0});

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[CPerBlock, KPerBlock] is in LDS
        //     b_mtx[CPerBlocl, N1 * BPerBlock * N2] is in LDS
        //     c_mtx[KPerBlock, N1 * BPerBlock * N2] is distributed among threads, and saved in
        //     register
        constexpr auto a_c_k_block_mtx_desc = make_ConstantMatrixDescriptor(
            Number<CPerBlock>{}, Number<KPerBlock>{}, Number<wei_c_k_block_desc.GetStride(I0)>{});

        constexpr auto b_c_n1bn2_block_mtx_desc =
            make_ConstantMatrixDescriptor(Number<CPerBlock>{},
                                          Number<N1 * BPerBlock * N2>{},
                                          Number<in_c_n1_b_n2_block_mem_desc.GetStride(I0)>{});

        // sanity check
        static_assert(KPerBlock % (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr index_t GemmMRepeat =
            KPerBlock / (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster);

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_k0k2_n1n2_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
            Number<GemmMRepeat * GemmMPerThreadSubC>{}, Number<N1 * N2>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
            BlockSize,
            decltype(a_c_k_block_mtx_desc),
            decltype(b_c_n1bn2_block_mtx_desc),
            decltype(c_k0k2_n1n2_thread_mtx_desc),
            GemmMPerThreadSubC,
            GemmNPerThreadSubC,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmKPerThreadLoop,
            GemmDataPerReadA,
            GemmDataPerReadB>{};

        // LDS allocation for input and weight: be careful of alignment
        constexpr index_t max_align = math::lcm(InBlockCopyDstDataPerWrite_N2,
                                                WeiBlockCopyDataPerAccess_K,
                                                GemmDataPerReadA,
                                                GemmDataPerReadB);

        constexpr index_t in_block_space =
            math::integer_least_multiple(in_c_n1_b_n2_block_mem_desc.GetElementSpace(), max_align);

        constexpr index_t wei_block_space =
            math::integer_least_multiple(wei_c_k_block_desc.GetElementSpace(), max_align);

        __shared__ Float p_in_block_double[2 * in_block_space];
        __shared__ Float p_wei_block_double[2 * wei_block_space];

        // register allocation for output
        Float p_out_thread[c_k0k2_n1n2_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_k0k2_n1n2_thread_mtx_desc, p_out_thread);

        // do work
        for(index_t y = 0; y < Y; ++y)
        {
            for(index_t x = 0; x < X; ++x)
            {
                // calculate origin of block input and weight tensor on global memory
                const Float* p_in_block_on_global =
                    p_in_global + in_n_c_h_w_global_desc.GetOffsetFromMultiIndex(0, 0, y, x);

                const Float* p_wei_block_on_global =
                    p_wei_global + wei_c_y_x_k_global_desc.GetOffsetFromMultiIndex(0, y, x, 0);

                // LDS double buffer: preload data into LDS
                {
                    blockwise_in_copy.Run(p_in_block_on_global, p_in_block_double);
                    blockwise_wei_copy.Run(p_wei_block_on_global, p_wei_block_double);
                }

                // LDS double buffer: main body
                for(index_t c_block_data_begin = 0; c_block_data_begin + 2 * CPerBlock < C;
                    c_block_data_begin += 2 * CPerBlock)
                {
#pragma unroll
                    for(index_t iloop = 0; iloop < 2; ++iloop)
                    {
                        const bool even_loop = (iloop % 2 == 0);

                        Float* p_in_block_now =
                            even_loop ? p_in_block_double : p_in_block_double + in_block_space;
                        Float* p_wei_block_now =
                            even_loop ? p_wei_block_double : p_wei_block_double + wei_block_space;

                        Float* p_in_block_next =
                            even_loop ? p_in_block_double + in_block_space : p_in_block_double;
                        Float* p_wei_block_next =
                            even_loop ? p_wei_block_double + wei_block_space : p_wei_block_double;

                        Float p_in_register_buffer[blockwise_in_copy.GetRegisterBufferSize()];
                        Float p_wei_register_buffer[blockwise_wei_copy.GetRegisterBufferSize()];

                        p_in_block_on_global += CPerBlock * in_n_c_h_w_global_desc.GetStride(I1);
                        p_wei_block_on_global += CPerBlock * wei_c_y_x_k_global_desc.GetStride(I0);

                        __syncthreads();

                        // LDS doubel buffer: load next data from device mem
                        blockwise_in_copy.RunLoadRegisterBuffer(p_in_block_on_global,
                                                                p_in_register_buffer);
                        blockwise_wei_copy.RunLoadRegisterBuffer(p_wei_block_on_global,
                                                                 p_wei_register_buffer);

                        // LDS double buffer: GEMM on current data
                        blockwise_gemm.Run(p_wei_block_now, p_in_block_now, p_out_thread);

                        // LDS double buffer: store next data to LDS
                        blockwise_in_copy.RunStoreRegisterBuffer(p_in_register_buffer,
                                                                 p_in_block_next);
                        blockwise_wei_copy.RunStoreRegisterBuffer(p_wei_register_buffer,
                                                                  p_wei_block_next);
                    }
                }

                // LDS double buffer: tail
                {
                    Float p_in_register_buffer[blockwise_in_copy.GetRegisterBufferSize()];
                    Float p_wei_register_buffer[blockwise_wei_copy.GetRegisterBufferSize()];

                    // even iteration
                    p_in_block_on_global += CPerBlock * in_n_c_h_w_global_desc.GetStride(I1);
                    p_wei_block_on_global += CPerBlock * wei_c_y_x_k_global_desc.GetStride(I0);

                    __syncthreads();

                    // LDS doubel buffer: load next data from device mem
                    blockwise_in_copy.RunLoadRegisterBuffer(p_in_block_on_global,
                                                            p_in_register_buffer);
                    blockwise_wei_copy.RunLoadRegisterBuffer(p_wei_block_on_global,
                                                             p_wei_register_buffer);

                    // LDS double buffer: GEMM on current data
                    blockwise_gemm.Run(p_wei_block_double, p_in_block_double, p_out_thread);

                    // LDS double buffer: store next data to LDS
                    blockwise_in_copy.RunStoreRegisterBuffer(p_in_register_buffer,
                                                             p_in_block_double + in_block_space);
                    blockwise_wei_copy.RunStoreRegisterBuffer(p_wei_register_buffer,
                                                              p_wei_block_double + wei_block_space);

                    // odd iteration
                    __syncthreads();

                    // LDS double buffer: GEMM on current data
                    blockwise_gemm.Run(p_wei_block_double + wei_block_space,
                                       p_in_block_double + in_block_space,
                                       p_out_thread);
                }
            }
        }

        // copy output: register to global memory
        {
            constexpr index_t K2 = GemmMPerThreadSubC;
            constexpr index_t K1 = GemmMLevel0Cluster * GemmMLevel1Cluster;
            constexpr index_t K0 = K / (K1 * K2);

            // define tensor descriptor for threadwise copy
            //     output memory layout descriptor in register
            constexpr auto out_k0_k1_k2_n1_n0_h_w_n2_thread_mem_desc =
                make_ConstantTensorDescriptor_packed(
                    Sequence<KPerBlock / (K1 * K2), 1, K2, N1, 1, 1, 1, N2>{});

            //     output tensor descriptor in register, src of threadwise copy
            constexpr auto out_n0_n1_n2_k0_k1_k2_h_w_thread_desc =
                out_k0_k1_k2_n1_n0_h_w_n2_thread_mem_desc.ReorderGivenNew2Old(
                    Sequence<4, 3, 7, 0, 1, 2, 5, 6>{});

            //     output memory layout descriptor in device memory, dst of threadwise copy
            constexpr auto out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc =
                out_n_k_h_w_global_desc.Fold(I1, Number<K1>{}, Number<K2>{})
                    .Fold(I0, Number<N1>{}, Number<N2>{});

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            const index_t k_thread_data_on_global =
                k_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t b_thread_data_on_global =
                b_block_data_on_global + c_thread_mtx_on_block.col / N2;

            //     output merged global tensor descriptor, for calculating origin of thread tensor
            //     in global memory
            constexpr auto out_k_n1_b_n2_global_merged_desc = make_ConstantMergedTensorDescriptor(
                out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc.Unfold(I3, I5),
                Sequence<3>{},
                Sequence<1>{},
                Sequence<0, 4, 5>{},
                Sequence<2>{});

            //     origin of dst in device memory
            Float* p_out_thread_on_global =
                p_out_global +
                out_k_n1_b_n2_global_merged_desc.GetOffsetFromMultiIndex(
                    k_thread_data_on_global, 0, b_thread_data_on_global, 0);

            threadwise_generic_tensor_slice_copy_v1(
                out_n0_n1_n2_k0_k1_k2_h_w_thread_desc,
                p_out_thread,
                {0, 0, 0, 0, 0, 0, 0, 0},
                out_n0_n1_n2_k0_k1_k2_h_w_global_mem_desc,
                p_out_thread_on_global,
                {0, 0, 0, 0, 0, 0, 0, 0},
                out_n0_n1_n2_k0_k1_k2_h_w_thread_desc.GetLengths(),
                arithmetic_sequence_gen<0, 8, 1>::type{},
                Number<1>{});
        }
    }
};

} // namesspace ck
#endif
