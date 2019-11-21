#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V1R1_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V1R1_NCHW_KCYX_NKHW_LDS_DOUBLE_BUFFER_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "blockwise_gemm.hpp"

namespace ck {

template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccDataType,
          typename InGlobalDesc,
          typename WeiGlobalDesc,
          typename OutGlobalDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads,
          index_t BPerBlock,
          index_t EPerBlock,
          index_t KPerBlock,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          typename OutBlockCopySubLengths_K_B,
          typename OutBlockCopyClusterLengths_K_B,
          index_t OutBlockCopyDataPerAccess_B,
          typename WeiBlockCopySubLengths_K_E,
          typename WeiBlockCopyClusterLengths_K_E,
          index_t WeiBlockCopyDataPerAccess_E,
          index_t InThreadCopyDataPerAccess_B>
struct GridwiseConvolutionBackwardDataImplicitGemm_v1r1_nchw_kcyx_nkhw_lds_double_buffer
{
    __device__ void Run(Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        const Float* const __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto in_n_c_hi_wi_global_desc  = InGlobalDesc{};
        constexpr auto wei_k_c_y_x_global_desc   = WeiGlobalDesc{};
        constexpr auto out_n_k_ho_wo_global_desc = OutGlobalDesc{};

        constexpr index_t N  = in_n_c_hi_wi_global_desc.GetLengths()[0];
        constexpr index_t C  = in_n_c_hi_wi_global_desc.GetLengths()[1];
        constexpr index_t Hi = in_n_c_hi_wi_global_desc.GetLengths()[2];
        constexpr index_t Wi = in_n_c_hi_wi_global_desc.GetLengths()[3];

        constexpr index_t K  = out_n_k_ho_wo_global_desc.GetLengths()[1];
        constexpr index_t Ho = out_n_k_ho_wo_global_desc.GetLengths()[2];
        constexpr index_t Wo = out_n_k_ho_wo_global_desc.GetLengths()[3];

        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLengths()[2];
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLengths()[3];

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t E = C * Y * X;
        constexpr index_t B = N * Ho * Wo;

        // sanity-check for vectorized memory load
        static_assert((Wo == 1 || (ConvStrideW == 1 || InThreadCopyDataPerAccess_B == 1)) &&
                          (X == 1 || ConvDilationW % InThreadCopyDataPerAccess_B == 0),
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // lds max alignment
        constexpr index_t max_lds_align = math::lcm(WeiBlockCopyDataPerAccess_E,
                                                    OutBlockCopyDataPerAccess_B,
                                                    GemmDataPerReadA,
                                                    GemmDataPerReadB);

        // divide block work by [K, B]
        static_assert(E % EPerBlock == 0 && B % BPerBlock == 0 && K % KPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t EBlockWork = E / EPerBlock;
        constexpr index_t BBlockWork = B / BPerBlock;

        constexpr auto block_work_desc =
            make_cluster_descriptor(Sequence<EBlockWork, BBlockWork>{});

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t e_block_data_on_global = block_work_id[0] * EPerBlock;
        const index_t b_block_data_on_global = block_work_id[1] * BPerBlock;

        // output tensor
        //     global tensor in global memory
        constexpr auto out_n_k_howo_global_desc =
            unfold_tensor_descriptor(out_n_k_ho_wo_global_desc, I2, I3);

        //     global tensor in global memory, src of blockwise copy
        constexpr auto out_k_b_global_desc =
            transform_tensor_descriptor(out_n_k_howo_global_desc,
                                        make_tuple(PassThrough<K>{}, Merge<Sequence<N, Ho * Wo>>{}),
                                        make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        //     block tensor in LDS memory, dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto out_k_b_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, BPerBlock>{}, Number<max_lds_align>{});

        // input tensor blockwise copy
        auto blockwise_out_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(out_k_b_global_desc),
                                               decltype(out_k_b_block_desc),
                                               decltype(out_k_b_block_desc.GetLengths()),
                                               OutBlockCopySubLengths_K_B,
                                               OutBlockCopyClusterLengths_K_B,
                                               Sequence<0, 1>,
                                               Sequence<0, 1>,
                                               Sequence<0, 1>,
                                               1,
                                               1,
                                               OutBlockCopyDataPerAccess_B,
                                               OutBlockCopyDataPerAccess_B,
                                               AddressSpace::global,
                                               AddressSpace::vgpr,
                                               AddressSpace::lds,
                                               InMemoryDataOperation::none>(
                {0, b_block_data_on_global}, {0, 0});

        // weight tensor
        //     global tensor in global memory, src of blockwise copy
        //     It is constructed differently, depending on whether forward or backward weight
        //       convolution
        constexpr auto wei_k_e_global_desc =
            unfold_tensor_descriptor(wei_k_c_y_x_global_desc, I1, I3);

        //     block tensor in LDS memory, dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto wei_k_e_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, EPerBlock>{}, Number<max_lds_align>{});

        // weight tensor blockwise copy
        auto blockwise_wei_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(wei_k_e_global_desc),
                                               decltype(wei_k_e_block_desc),
                                               decltype(wei_k_e_block_desc.GetLengths()),
                                               WeiBlockCopySubLengths_K_E,
                                               WeiBlockCopyClusterLengths_K_E,
                                               Sequence<0, 1>,
                                               Sequence<0, 1>,
                                               Sequence<0, 1>,
                                               1,
                                               1,
                                               WeiBlockCopyDataPerAccess_E,
                                               WeiBlockCopyDataPerAccess_E,
                                               AddressSpace::global,
                                               AddressSpace::vgpr,
                                               AddressSpace::lds,
                                               InMemoryDataOperation::none>(
                {0, e_block_data_on_global}, {0, 0});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, EPerBlock] is in LDS
        //     b_mtx[KPerBlocl, BPerBlock] is in LDS
        //     c_mtx[EPerBlock, BPerBlock] is distributed among threads, and saved in
        //       register
        constexpr auto a_k_e_block_mtx_desc = make_ConstantMatrixDescriptor(wei_k_e_block_desc);
        constexpr auto b_k_b_block_mtx_desc = make_ConstantMatrixDescriptor(out_k_b_block_desc);

        // sanity check
        static_assert(
            EPerBlock % (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster) == 0 &&
                BPerBlock % (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) == 0,
            "wrong!");

        constexpr index_t GemmMRepeat =
            EPerBlock / (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster);

        constexpr index_t GemmNRepeat =
            BPerBlock / (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster);

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_e0e1_b0b1_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
            Number<GemmMRepeat * GemmMPerThreadSubC>{}, Number<GemmNRepeat * GemmNPerThreadSubC>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
            BlockSize,
            decltype(a_k_e_block_mtx_desc),
            decltype(b_k_b_block_mtx_desc),
            decltype(c_e0e1_b0b1_thread_mtx_desc),
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
        constexpr index_t out_block_space =
            math::integer_least_multiple(out_k_b_block_desc.GetElementSpace(), max_lds_align);

        constexpr index_t wei_block_space =
            math::integer_least_multiple(wei_k_e_block_desc.GetElementSpace(), max_lds_align);

        __shared__ Float p_out_block_double[2 * out_block_space];
        __shared__ Float p_wei_block_double[2 * wei_block_space];

        // register allocation for output
        AccDataType p_in_thread[c_e0e1_b0b1_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_e0e1_b0b1_thread_mtx_desc, p_in_thread);

        // LDS double buffer: preload data into LDS
        {
            blockwise_out_copy.Run(p_out_global, p_out_block_double);
            blockwise_wei_copy.Run(p_wei_global, p_wei_block_double);
        }

        // LDS double buffer: main body
        for(index_t k_block_data_begin = 0; k_block_data_begin + 2 * KPerBlock < K;
            k_block_data_begin += 2 * KPerBlock)
        {
#pragma unroll
            for(index_t iloop = 0; iloop < 2; ++iloop)
            {
                const bool even_loop = (iloop % 2 == 0);

                Float* p_out_block_now =
                    even_loop ? p_out_block_double : p_out_block_double + out_block_space;
                Float* p_wei_block_now =
                    even_loop ? p_wei_block_double : p_wei_block_double + wei_block_space;

                Float* p_out_block_next =
                    even_loop ? p_out_block_double + out_block_space : p_out_block_double;
                Float* p_wei_block_next =
                    even_loop ? p_wei_block_double + wei_block_space : p_wei_block_double;

                Float p_out_thread_buffer[blockwise_out_copy.GetThreadBufferSize()];
                Float p_wei_thread_buffer[blockwise_wei_copy.GetThreadBufferSize()];

                blockwise_out_copy.MoveSrcSliceWindow(Sequence<KPerBlock, 0>{}, True);
                blockwise_wei_copy.MoveSrcSliceWindow(Sequence<KPerBlock, 0>{}, True);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                blockwise_out_copy.RunLoadThreadBuffer(p_out_global, p_out_thread_buffer);
                blockwise_wei_copy.RunLoadThreadBuffer(p_wei_global, p_wei_thread_buffer);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(p_wei_block_now, p_out_block_now, p_in_thread);

                // LDS double buffer: store next data to LDS
                blockwise_out_copy.RunStoreThreadBuffer(p_out_thread_buffer, p_out_block_next);
                blockwise_wei_copy.RunStoreThreadBuffer(p_wei_thread_buffer, p_wei_block_next);
            }
        }

        // LDS double buffer: tail
        {
            constexpr bool has_two_iteration_left = (K % (2 * KPerBlock) == 0);

            if(has_two_iteration_left) // if has 2 iteration left
            {
                Float p_out_thread_buffer[blockwise_out_copy.GetThreadBufferSize()];
                Float p_wei_thread_buffer[blockwise_wei_copy.GetThreadBufferSize()];

                blockwise_out_copy.MoveSrcSliceWindow(Sequence<KPerBlock, 0>{}, True);
                blockwise_wei_copy.MoveSrcSliceWindow(Sequence<KPerBlock, 0>{}, True);

                __syncthreads();

                // LDS double buffer: load last data from device mem
                blockwise_out_copy.RunLoadThreadBuffer(p_out_global, p_out_thread_buffer);
                blockwise_wei_copy.RunLoadThreadBuffer(p_wei_global, p_wei_thread_buffer);

                // LDS double buffer: GEMM on 2nd-last data
                blockwise_gemm.Run(p_wei_block_double, p_out_block_double, p_in_thread);

                // LDS double buffer: store last data to LDS
                blockwise_out_copy.RunStoreThreadBuffer(p_out_thread_buffer,
                                                        p_out_block_double + out_block_space);
                blockwise_wei_copy.RunStoreThreadBuffer(p_wei_thread_buffer,
                                                        p_wei_block_double + wei_block_space);

                __syncthreads();

                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(p_wei_block_double + wei_block_space,
                                   p_out_block_double + out_block_space,
                                   p_in_thread);
            }
            else // if has 1 iteration left
            {
                __syncthreads();

                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(p_wei_block_double, p_out_block_double, p_in_thread);
            }
        }

        // input: register to global memory, atomic add
        {
            constexpr index_t E1 = GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster;
            constexpr index_t E0 = E / E1;

            constexpr index_t B1 = GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster;
            constexpr index_t B0 = B / B1;

            // define input tensor descriptor for threadwise copy
            //     thread input tensor, src of threadwise copy
            constexpr auto in_e0_e1_b0_b1_thread_desc = make_native_tensor_descriptor_packed(
                Sequence<GemmMRepeat, GemmMPerThreadSubC, GemmNRepeat, GemmNPerThreadSubC>{});

            //     global input tensor, dst of threadwise copy
            constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
                in_n_c_hi_wi_global_desc,
                make_tuple(PassThrough<N>{},
                           PassThrough<C>{},
                           Pad<Sequence<Hi, Wi>, LeftPads, RightPads>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

            constexpr auto in_n_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
                in_n_c_hip_wip_global_desc,
                make_tuple(PassThrough<N>{},
                           PassThrough<C>{},
                           Embed<Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                           Embed<Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

            constexpr auto in_e_b_global_desc = transform_tensor_descriptor(
                in_n_c_y_ho_x_wo_global_desc,
                make_tuple(Merge<Sequence<C, Y, X>>{}, Merge<Sequence<N, Ho, Wo>>{}),
                make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            constexpr auto in_e0_e1_b0_b1_global_desc = transform_tensor_descriptor(
                in_e_b_global_desc,
                make_tuple(UnMerge<Sequence<E0, E1>>{}, UnMerge<Sequence<B0, B1>>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

            // calculate origin of thread input tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            const index_t e_thread_data_on_global =
                e_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t b_thread_data_on_global =
                b_block_data_on_global + c_thread_mtx_on_block.col;

            ThreadwiseGenericTensorSliceCopy_v4r2<decltype(in_e0_e1_b0_b1_thread_desc),
                                                  decltype(in_e0_e1_b0_b1_global_desc),
                                                  decltype(in_e0_e1_b0_b1_thread_desc.GetLengths()),
                                                  Sequence<0, 1, 2, 3>,
                                                  3,
                                                  InThreadCopyDataPerAccess_B,
                                                  InThreadCopyDataPerAccess_B,
                                                  AddressSpace::vgpr,
                                                  AddressSpace::global,
                                                  InMemoryDataOperation::atomic_add>(
                {0, 0, 0, 0},
                {e_thread_data_on_global / E1,
                 e_thread_data_on_global % E1,
                 b_thread_data_on_global / B1,
                 b_thread_data_on_global % B1})
                .Run(p_in_thread, p_in_global);
        }
    }
};

} // namespace ck
#endif
