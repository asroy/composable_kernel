#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V4R2_NCHW_KCYX_NKHW_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V4R2_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

// Number of GEMMs: YTilda * XTilda
// GemmM = C
// GemmB = N0 * HTildaSlice * WTildaSlice
// GemmK = K * YDotSlice * XDotSlice
template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccFloat,
          typename InGlobalDesc,
          typename WeiGlobalDesc,
          typename OutGlobalDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerThread,
          index_t GemmNPerThread,
          index_t GemmKPerThread,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t ThreadGemmDataPerRead_GemmM,
          index_t ThreadGemmDataPerRead_GemmN,
          typename OutBlockCopySliceLengths_K_N1_B_N2,
          typename OutBlockCopyClusterLengths_K_N1_B_N2,
          typename OutBlockCopyThreadClusterArrangeOrder,
          typename OutBlockCopySrcAccessOrder,
          typename OutBlockCopyDstAccessOrder,
          index_t OutBlockCopySrcDataPerRead_B,
          index_t OutBlockCopySrcDataPerWrite_N2,
          typename WeiBlockCopySliceLengths_K_M,
          typename WeiBlockCopyClusterLengths_K_M,
          typename WeiBlockCopyThreadClusterArrangeOrder,
          typename WeiBlockCopySrcAccessOrder,
          typename WeiBlockCopyDstAccessOrder,
          index_t WeiBlockCopySrcDataPerRead_M,
          index_t WeiBlockCopySrcDataPerWrite_M>
struct GridwiseConvolutionBackwardDataImplicitGemm_v4r2_nchw_kcyx_nkhw
{
    __host__ __device__ static constexpr index_t GetNumberOfGemm()
    {
        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        return YTilda * XTilda;
    }

    __host__ __device__ static constexpr auto GetGemmSizeImpl(index_t iYTilda, index_t iXTilda)
    {
        constexpr index_t N  = InGlobalDesc::GetLengths()[0];
        constexpr index_t C  = InGlobalDesc::GetLengths()[1];
        constexpr index_t Hi = InGlobalDesc::GetLengths()[2];
        constexpr index_t Wi = InGlobalDesc::GetLengths()[3];

        constexpr index_t K  = OutGlobalDesc::GetLengths()[1];
        constexpr index_t Ho = OutGlobalDesc::GetLengths()[2];
        constexpr index_t Wo = OutGlobalDesc::GetLengths()[3];

        constexpr index_t Y = WeiGlobalDesc::GetLengths()[2];
        constexpr index_t X = WeiGlobalDesc::GetLengths()[3];

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        constexpr index_t YDot = math::integer_divide_ceil(Y, YTilda);
        constexpr index_t XDot = math::integer_divide_ceil(X, XTilda);

        constexpr index_t HTilda =
            Ho + math::integer_divide_ceil(ConvDilationH * (Y - 1), ConvStrideH);
        constexpr index_t WTilda =
            Wo + math::integer_divide_ceil(ConvDilationW * (X - 1), ConvStrideW);

        // only work on HTilda and WTilda that contribute to non-padding area of input tensor
        constexpr index_t iHTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[0] - ConvDilationH * (YTilda - 1)), ConvStrides{}[0]);
        constexpr index_t iWTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[1] - ConvDilationW * (XTilda - 1)), ConvStrides{}[1]);

        constexpr index_t iHTildaRight = math::min(
            HTilda, math::integer_divide_ceil(InLeftPads{}[0] + Hi - 1, ConvStrides{}[0]) + 1);
        constexpr index_t iWTildaRight = math::min(
            WTilda, math::integer_divide_ceil(InLeftPads{}[1] + Wi - 1, ConvStrides{}[1]) + 1);

        constexpr index_t HTildaSlice = iHTildaRight - iHTildaLeft;
        constexpr index_t WTildaSlice = iWTildaRight - iWTildaLeft;

        // GemmM and GemmN
        constexpr index_t GemmM = C;
        constexpr index_t GemmN = N * HTildaSlice * WTildaSlice;

        // GemmK is different for each GEMM
        index_t YDotSlice = (iYTilda + 1) * YDot <= Y ? YDot : Y % YDot;
        index_t XDotSlice = (iXTilda + 1) * XDot <= X ? XDot : X % XDot;

        index_t GemmK = K * YDotSlice * XDotSlice;

        return Array<index_t, 3>{GemmM, GemmN, GemmK};
    }

    __host__ __device__ static constexpr auto GetGemmSize(index_t gemm_id)
    {
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        index_t iYTilda = gemm_id / XTilda;
        index_t iXTilda = gemm_id % XTilda;

        return GetGemmSizeImpl(iYTilda, iXTilda);
    }

    template <index_t iYTilda, index_t iXTilda>
    __device__ static void RunImpl(Float* __restrict__ p_in_global,
                                   const Float* __restrict__ p_wei_global,
                                   const Float* __restrict__ p_out_global)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

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

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        constexpr index_t YDot = math::integer_divide_ceil(Y, YTilda);
        constexpr index_t XDot = math::integer_divide_ceil(X, XTilda);

        constexpr index_t HTilda =
            Ho + math::integer_divide_ceil(ConvDilationH * (Y - 1), ConvStrideH);
        constexpr index_t WTilda =
            Wo + math::integer_divide_ceil(ConvDilationW * (X - 1), ConvStrideW);

        // only work on HTilda and WTilda that contribute to non-padding area of input tensor
        constexpr index_t iHTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[0] - ConvDilationH * (YTilda - 1)), ConvStrides{}[0]);
        constexpr index_t iWTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[1] - ConvDilationW * (XTilda - 1)), ConvStrides{}[1]);

        constexpr index_t iHTildaRight = math::min(
            HTilda, math::integer_divide_ceil(InLeftPads{}[0] + Hi - 1, ConvStrides{}[0]) + 1);
        constexpr index_t iWTildaRight = math::min(
            WTilda, math::integer_divide_ceil(InLeftPads{}[1] + Wi - 1, ConvStrides{}[1]) + 1);

        constexpr index_t HTildaSlice = iHTildaRight - iHTildaLeft;
        constexpr index_t WTildaSlice = iWTildaRight - iWTildaLeft;

        // GEMM
        constexpr index_t YDotSlice = (iYTilda + 1) * YDot <= Y ? YDot : Y % YDot;
        constexpr index_t XDotSlice = (iXTilda + 1) * XDot <= X ? XDot : X % XDot;

        constexpr index_t max_lds_align = math::lcm(OutBlockCopySrcDataPerRead_B,
                                            WeiBlockCopySrcDataPerRead_M,
                                            ThreadGemmDataPerRead_GemmM,
                                            ThreadGemmDataPerRead_GemmN);

        static_assert(GemmMPerBlock % (GemmMPerThread * GemmMLevel0Cluster * GemmMLevel1Cluster) ==
                          0, "wrong!");

        constexpr index_t GemmMRepeat =
            GemmMPerBlock / (GemmMPerThread * GemmMLevel0Cluster * GemmMLevel1Cluster);

        static_assert(GemmNPerBlock % (GemmNPerThread * GemmNLevel0Cluster * GemmNLevel1Cluster) ==
                          0, "wrong!");

        constexpr index_t GemmNRepeat =
            GemmNPerBlock / (GemmNPerThread * GemmNLevel0Cluster * GemmNLevel1Cluster);

        constexpr index_t N1 = GemmNRepeat;
        constexpr index_t N2 = GemmNPerThread;

        static_assert(N % (N1 * N2) == 0, "wrong! cannot divide N evenly among thread");
        constexpr index_t N0 = N / (N1 * N2);

        constexpr index_t GemmM = C;
        constexpr index_t GemmN = N * HTildaSlice * WTildaSlice;
        constexpr index_t GemmK = K * YDotSlice * XDotSlice;

        static_assert(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 && GemmK % GemmKPerBlock == 0,
                "wrong! cannot divide work evenly among block");
        static_assert(GemmNPerBlock % (N1 * N2) == 0, "wrong! cannot divide N1 from N");

        // B dimension is divided from N
        constexpr index_t GemmBPerBlock = GemmNPerBlock / (N1 * N2);
        constexpr index_t GemmB = GemmN / (N1 * N2);

        constexpr index_t MBlockWork = GemmM / GemmMPerBlock;
        constexpr index_t BBlockWork = GemmB / GemmBPerBlock;

        constexpr auto block_work_desc =
            make_cluster_descriptor(Sequence<MBlockWork, BBlockWork>{});

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t m_block_data_on_global = block_work_id[0] * GemmMPerBlock;
        const index_t b_block_data_on_global = block_work_id[1] * GemmBPerBlock;

        // weight out-of-bound check can be skipped
        constexpr bool wei_skip_out_of_bound_check = true;

        // weight transform
        constexpr auto wei_k_c_ydot_ytilda_xdot_xtilda_global_desc = transform_tensor_descriptor(
            wei_k_c_y_x_global_desc,
            make_tuple(PassThrough<K>{},
                       PassThrough<C>{},
                       Embed<Y,
                             Sequence<YDot, YTilda>,
                             Sequence<ConvStrideH / GcdStrideDilationH, 1, 0>,
                             wei_skip_out_of_bound_check>{},
                       Embed<X,
                             Sequence<XDot, XTilda>,
                             Sequence<ConvStrideW / GcdStrideDilationW, 1, 0>,
                             wei_skip_out_of_bound_check>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto wei_k_c_ydotslice_ytidaslice_xdotslice_xtildaslice_global_desc =
            transform_tensor_descriptor(
                wei_k_c_ydot_ytilda_xdot_xtilda_global_desc,
                make_tuple(
                    PassThrough<K>{},
                    PassThrough<C>{},
                    Slice<Sequence<YDot, XDot>, Sequence<0, 0>, Sequence<YDotSlice, XDotSlice>>{},
                    Slice<Sequence<YTilda, XTilda>,
                          Sequence<iYTilda, iXTilda>,
                          Sequence<iYTilda + 1, iXTilda + 1>>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 4>{}, Sequence<3, 5>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 4>{}, Sequence<3, 5>{}));

        constexpr auto wei_k_m_global_desc = transform_tensor_descriptor(
            wei_k_c_ydotslice_ytidaslice_xdotslice_xtildaslice_global_desc,
            make_tuple(Merge<Sequence<K, YDotSlice, XDotSlice>>{}, Merge<Sequence<C, 1, 1>>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        constexpr auto wei_k_m_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<GemmKPerBlock, GemmMPerBlock>{}, Number<max_lds_align>{});

        // weight tensor blockwise copy
        auto wei_blockwise_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(wei_k_m_global_desc),
                                               decltype(wei_k_m_block_desc),
                                               decltype(wei_k_m_block_desc.GetLengths()),
                                               WeiBlockCopySliceLengths_K_M,
                                               WeiBlockCopyClusterLengths_K_M,
                                               WeiBlockCopyThreadClusterArrangeOrder,   // ThreadClusterArrangeOrder
                                               WeiBlockCopySrcAccessOrder,              // SrcDimAccessOrder
                                               WeiBlockCopyDstAccessOrder,              // DstDimAccessOrder
                                               1,                                       // SrcVectoReadDim
                                               1,                                       // DstVectorWriteDim
                                               WeiBlockCopySrcDataPerRead_M,
                                               WeiBlockCopySrcDataPerWrite_M,
                                               AddressSpace::Global,
                                               AddressSpace::Vgpr,
                                               AddressSpace::Lds,
                                               InMemoryDataOperation::Set>(
                {0, m_block_data_on_global}, {0, 0});

#if !CK_EXPERIMENTAL_IMPLICIT_GEMM_BACKWARD_DATA_V4R2_OUTPUT_SKIP_OUT_OF_BOUND_CHECK
        constexpr bool out_skip_out_of_bound_check = false;
#else
        //\todo sometimes output tensor out-of-bound check can be skipped, find out all such
        // situations
        constexpr bool out_skip_out_of_bound_check = true;
#endif
        // output transform
        constexpr auto out_n0_n1_n2_k_ydot_htilda_xdot_wtilda_global_desc = transform_tensor_descriptor(
            out_n_k_ho_wo_global_desc,
            make_tuple(UnMerge<Sequence<N0, N1, N2>>{},
                       PassThrough<K>{},
                       Embed<Ho,
                             Sequence<YDot, HTilda>,
                             Sequence<-ConvDilationH / GcdStrideDilationH, 1, 0>,
                             out_skip_out_of_bound_check>{},
                       Embed<Wo,
                             Sequence<XDot, WTilda>,
                             Sequence<-ConvDilationW / GcdStrideDilationW, 1, 0>,
                             out_skip_out_of_bound_check>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}, Sequence<6, 7>{}));

        constexpr auto out_n0_n1_n2_k_ydot_htildaslice_xdot_wtildaslice_global_desc =
            transform_tensor_descriptor(
                out_n0_n1_n2_k_ydot_htilda_xdot_wtilda_global_desc,
                make_tuple(PassThrough<N0>{},
                           PassThrough<N1>{},
                           PassThrough<N2>{},
                           PassThrough<K>{},
                           PassThrough<YTilda>{},
                           PassThrough<XTilda>{},
                           Slice<Sequence<HTilda, WTilda>,
                                 Sequence<iHTildaLeft, iWTildaLeft>,
                                 Sequence<iHTildaRight, iWTildaRight>>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<6>{}, Sequence<5, 7>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<6>{}, Sequence<5, 7>{}));

        constexpr auto out_n0_n1_n2_k_ydotslice_htildaslice_xdotslice_wtildaslice_global_desc =
            transform_tensor_descriptor(
                out_n0_n1_n2_k_ydot_htildaslice_xdot_wtildaslice_global_desc,
                make_tuple(
                    PassThrough<N0>{},
                    PassThrough<N1>{},
                    PassThrough<N2>{},
                    PassThrough<K>{},
                    PassThrough<HTildaSlice>{},
                    PassThrough<WTildaSlice>{},
                    Slice<Sequence<YDot, XDot>, Sequence<0, 0>, Sequence<YDotSlice, XDotSlice>>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<5>{}, Sequence<7>{}, Sequence<4, 6>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<5>{}, Sequence<7>{}, Sequence<4, 6>{}));

        constexpr auto out_k_n1_b_n2_global_desc = transform_tensor_descriptor(
            out_n0_n1_n2_k_ydotslice_htildaslice_xdotslice_wtildaslice_global_desc,
            make_tuple(Merge<Sequence<K, YDotSlice, XDotSlice>>{},
                       PassThrough<N1>{},
                       Merge<Sequence<N0, HTildaSlice, WTildaSlice>>{},
                       PassThrough<N2>{}),
            make_tuple(Sequence<3, 4, 6>{}, Sequence<1>{}, Sequence<0, 5, 7>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));


        constexpr auto out_k_n1_b_n2_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<GemmKPerBlock, N1, GemmBPerBlock, N2>{}, Number<max_lds_align>{});

        // out tensor blockwise copy
        auto out_blockwise_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(out_k_n1_b_n2_global_desc),
                                               decltype(out_k_n1_b_n2_block_desc),
                                               decltype(out_k_n1_b_n2_block_desc.GetLengths()),
                                               OutBlockCopySliceLengths_K_N1_B_N2,
                                               OutBlockCopyClusterLengths_K_N1_B_N2,
                                               OutBlockCopyThreadClusterArrangeOrder,   // ThreadClusterArrangeOrder
                                               OutBlockCopySrcAccessOrder,              // SrcDimAccessOrder
                                               OutBlockCopyDstAccessOrder,              // DstDimAccessOrder
                                               2,                                       // SrcVectoReadDim
                                               3,                                       // DstVectorWriteDim
                                               OutBlockCopySrcDataPerRead_B,
                                               OutBlockCopySrcDataPerWrite_N2,
                                               AddressSpace::Global,
                                               AddressSpace::Vgpr,
                                               AddressSpace::Lds,
                                               InMemoryDataOperation::Set>(
                {0, 0, b_block_data_on_global, 0}, {0, 0, 0, 0});

        // GEMM definition
        constexpr auto a_k_m_block_mtx_desc = make_ConstantMatrixDescriptor(wei_k_m_block_desc);
        constexpr auto b_k_n_block_mtx_desc = make_ConstantMatrixDescriptor_packed(
            out_k_n1_b_n2_block_desc.GetLength(I0),
            out_k_n1_b_n2_block_desc.GetLength(I1) * out_k_n1_b_n2_block_desc.GetLength(I2) * out_k_n1_b_n2_block_desc.GetLength(I3));

        constexpr auto c_m_n_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
            Number<GemmMRepeat * GemmMPerThread>{}, Number<GemmNRepeat * GemmNPerThread>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
                                        BlockSize,
                                        decltype(a_k_m_block_mtx_desc),
                                        decltype(b_k_n_block_mtx_desc),
                                        decltype(c_m_n_thread_mtx_desc),
                                        GemmMPerThread,
                                        GemmNPerThread,
                                        GemmMLevel0Cluster,
                                        GemmNLevel0Cluster,
                                        GemmMLevel1Cluster,
                                        GemmNLevel1Cluster,
                                        GemmKPerThread,
                                        ThreadGemmDataPerRead_GemmM,
                                        ThreadGemmDataPerRead_GemmN>{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr index_t out_block_space =
            math::integer_least_multiple(out_k_n1_b_n2_block_desc.GetElementSpace(), max_lds_align);

        constexpr index_t wei_block_space =
            math::integer_least_multiple(wei_k_m_block_desc.GetElementSpace(), max_lds_align);

        __shared__ Float p_out_block_double[2 * out_block_space];
        __shared__ Float p_wei_block_double[2 * wei_block_space];

        AccFloat p_in_thread[c_m_n_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_m_n_thread_mtx_desc, p_in_thread);

        // LDS double buffer: preload data into LDS
        {
            out_blockwise_copy.Run(p_out_global, p_out_block_double);
            wei_blockwise_copy.Run(p_wei_global, p_wei_block_double);
        }

        constexpr auto out_block_slice_copy_steps = Sequence<GemmKPerBlock, 0, 0, 0>{};
        constexpr auto wei_block_slice_copy_steps = Sequence<GemmKPerBlock, 0>{};

        // LDS double buffer: main body
        for(index_t k_block_data_begin = 0; k_block_data_begin + 2 * GemmKPerBlock < GemmK;
            k_block_data_begin += 2 * GemmKPerBlock)
        {
#pragma unroll
            for(index_t iloop = 0; iloop < 2; ++iloop)
            {
                const bool even_loop = (iloop % 2 == 0);

                Float* p_wei_block_now =
                    even_loop ? p_wei_block_double : p_wei_block_double + wei_block_space;
                Float* p_out_block_now =
                    even_loop ? p_out_block_double : p_out_block_double + out_block_space;

                Float* p_wei_block_next =
                    even_loop ? p_wei_block_double + wei_block_space : p_wei_block_double;
                Float* p_out_block_next =
                    even_loop ? p_out_block_double + out_block_space : p_out_block_double;

                Float p_wei_thread_buffer[wei_blockwise_copy.GetThreadBufferSize()];
                Float p_out_thread_buffer[out_blockwise_copy.GetThreadBufferSize()];

                wei_blockwise_copy.MoveSrcSliceWindow(wei_block_slice_copy_steps, True);
                out_blockwise_copy.MoveSrcSliceWindow(out_block_slice_copy_steps, True);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                wei_blockwise_copy.RunLoadThreadBuffer(p_wei_global, p_wei_thread_buffer);
                out_blockwise_copy.RunLoadThreadBuffer(p_out_global, p_out_thread_buffer);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(p_wei_block_now, p_out_block_now, p_in_thread);

                // LDS double buffer: store next data to LDS
                wei_blockwise_copy.RunStoreThreadBuffer(p_wei_thread_buffer, p_wei_block_next);
                out_blockwise_copy.RunStoreThreadBuffer(p_out_thread_buffer, p_out_block_next);
            }
        }

        // LDS double buffer: tail
        {
            constexpr bool has_two_iteration_left = (GemmK % (2 * GemmKPerBlock) == 0);

            if(has_two_iteration_left) // if has 2 iteration left
            {
                Float p_wei_thread_buffer[wei_blockwise_copy.GetThreadBufferSize()];
                Float p_out_thread_buffer[out_blockwise_copy.GetThreadBufferSize()];

                wei_blockwise_copy.MoveSrcSliceWindow(wei_block_slice_copy_steps, True);
                out_blockwise_copy.MoveSrcSliceWindow(out_block_slice_copy_steps, True);

                __syncthreads();

                // LDS double buffer: load last data from device mem
                wei_blockwise_copy.RunLoadThreadBuffer(p_wei_global, p_wei_thread_buffer);
                out_blockwise_copy.RunLoadThreadBuffer(p_out_global, p_out_thread_buffer);

                // LDS double buffer: GEMM on 2nd-last data
                blockwise_gemm.Run(p_wei_block_double, p_out_block_double, p_in_thread);

                // LDS double buffer: store last data to LDS
                wei_blockwise_copy.RunStoreThreadBuffer(p_wei_thread_buffer,
                                                      p_wei_block_double + wei_block_space);
                out_blockwise_copy.RunStoreThreadBuffer(p_out_thread_buffer,
                                                      p_out_block_double + out_block_space);

                __syncthreads();

                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(
                    p_wei_block_double + wei_block_space, p_out_block_double + out_block_space, p_in_thread);
            }
            else // if has 1 iteration left
            {
                __syncthreads();

                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(p_wei_block_double, p_out_block_double, p_in_thread);
            }
        }

        //for(int i=0; i < c_m_n_thread_mtx_desc.GetElementSpace(); i++)
        //    p_in_thread[i] = (AccFloat)2;

        // store to global
        {
            constexpr index_t M1 = GemmMPerThread * GemmMLevel0Cluster * GemmMLevel1Cluster;
            constexpr index_t M0 = GemmM / M1;

#if !CK_EXPERIMENTAL_IMPLICIT_GEMM_BACKWARD_DATA_V4R2_INPUT_SKIP_OUT_OF_BOUND_CHECK
            constexpr bool in_skip_out_of_bound_check = false;
#else
            //\todo sometimes input out-of-bound check can be skipped, find out all such situations
            constexpr bool in_skip_out_of_bound_check = true;
#endif
            // input tensor
            constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
                in_n_c_hi_wi_global_desc,
                make_tuple(
                    PassThrough<N>{},
                    PassThrough<C>{},
                    Pad<Sequence<Hi, Wi>, InLeftPads, InRightPads, in_skip_out_of_bound_check>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

            constexpr index_t Hip = in_n_c_hip_wip_global_desc.GetLengths()[2];
            constexpr index_t Wip = in_n_c_hip_wip_global_desc.GetLengths()[3];

            constexpr auto in_n0_n1_n2_c_ytilda_htilda_xtilda_wtilda_global_desc = transform_tensor_descriptor(
                in_n_c_hip_wip_global_desc,
                make_tuple( UnMerge<Sequence<N0, N1, N2>>{},
                            PassThrough<C>{},
                            Embed<Hip,
                                    Sequence<YTilda, HTilda>,
                                    Sequence<ConvDilationH, ConvStrideH, 0>,
                                    in_skip_out_of_bound_check>{},
                            Embed<Wip,
                                    Sequence<XTilda, WTilda>,
                                    Sequence<ConvDilationW, ConvStrideW, 0>,
                                    in_skip_out_of_bound_check>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}, Sequence<6, 7>{}));

            constexpr auto in_n0_n1_n2_c_ytilda_htildaslice_xtilda_wtildaslice_global_desc =
                transform_tensor_descriptor(
                    in_n0_n1_n2_c_ytilda_htilda_xtilda_wtilda_global_desc,
                    make_tuple( PassThrough<N0>{},
                                PassThrough<N1>{},
                                PassThrough<N2>{},
                                PassThrough<C>{},
                                PassThrough<YTilda>{},
                                PassThrough<XTilda>{},
                                Slice<Sequence<HTilda, WTilda>,
                                        Sequence<iHTildaLeft, iWTildaLeft>,
                                        Sequence<iHTildaRight, iWTildaRight>>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<6>{}, Sequence<5, 7>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<6>{}, Sequence<5, 7>{}));

            constexpr auto in_n0_n1_n2_c_ytildaslice_htildaslice_xtildaslice_wtildaslice_global_desc =
                transform_tensor_descriptor(
                    in_n0_n1_n2_c_ytilda_htildaslice_xtilda_wtildaslice_global_desc,
                    make_tuple(
                            PassThrough<N0>{},
                            PassThrough<N1>{},
                            PassThrough<N2>{},
                            PassThrough<C>{},
                            PassThrough<HTildaSlice>{},
                            PassThrough<WTildaSlice>{},
                            Slice<Sequence<YTilda, XTilda>,
                                    Sequence<iYTilda, iXTilda>,
                                    Sequence<iYTilda + 1, iXTilda + 1>>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<5>{}, Sequence<7>{}, Sequence<4, 6>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<5>{}, Sequence<7>{}, Sequence<4, 6>{}));

            constexpr auto in_m_n1_b_n2_global_desc = transform_tensor_descriptor(
                in_n0_n1_n2_c_ytildaslice_htildaslice_xtildaslice_wtildaslice_global_desc,
                make_tuple(Merge<Sequence<C, 1, 1>>{},
                        PassThrough<N1>{},
                        Merge<Sequence<N0, HTildaSlice, WTildaSlice>>{},
                        PassThrough<N2>{}),
                make_tuple(Sequence<3, 4, 6>{}, Sequence<1>{}, Sequence<0, 5, 7>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            constexpr auto in_m0_m1_n1_b_n2_global_desc = transform_tensor_descriptor(
                in_m_n1_b_n2_global_desc,
                make_tuple(UnMerge<Sequence<M0, M1>>{},
                        PassThrough<N1>{},
                        PassThrough<N0 * HTildaSlice * WTildaSlice>{},
                        PassThrough<N2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            constexpr auto in_m0_m1_n1_b_n2_thread_desc = make_native_tensor_descriptor_packed(
                Sequence<GemmMRepeat, GemmMPerThread, GemmNRepeat, 1, GemmNPerThread>{});

            // calculate origin of thread input tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            const index_t m_thread_data_on_global =
                m_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t b_thread_data_on_global =
                b_block_data_on_global + c_thread_mtx_on_block.col / N2;

            ThreadwiseGenericTensorSliceCopy_v4r2<decltype(in_m0_m1_n1_b_n2_thread_desc),
                                                  decltype(in_m0_m1_n1_b_n2_global_desc),
                                                  decltype(in_m0_m1_n1_b_n2_thread_desc.GetLengths()),
                                                  Sequence<0, 1, 2, 3, 4>,
                                                  4,
                                                  1,
                                                  1,
                                                  AddressSpace::Vgpr,
                                                  AddressSpace::Global,
                                                  InMemoryDataOperation::Set>(
                {0, 0, 0, 0, 0},
                {m_thread_data_on_global / M1,
                 m_thread_data_on_global % M1,
                 0,
                 b_thread_data_on_global,
                 0})
                .Run(p_in_thread, p_in_global);
        }
    }

    template <index_t GemmId>
    __device__ static void Run(Float* __restrict__ p_in_global,
                               const Float* __restrict__ p_wei_global,
                               const Float* __restrict__ p_out_global)
    {
        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        constexpr index_t iYTilda = GemmId / XTilda;
        constexpr index_t iXTilda = GemmId % XTilda;

        static_assert(iYTilda < YTilda && iXTilda < XTilda, "wrong! iYtilda, iXtilda");

        RunImpl<iYTilda, iXTilda>(p_in_global, p_wei_global, p_out_global);
    }
};

} // namespace ck
#endif
