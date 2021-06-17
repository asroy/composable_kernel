#include "common_header.hpp"
#include "type_helper.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_contraction_v1r1_olc.hpp"
#include "transform_forward_convolution_into_gemm_v4r5_nchw_kcyx_nkhw.hpp"

using namespace ck;

constexpr index_t InWeiVectorSize = CK_PARAM_InWeiVectorSize; 
using TInWei = typename get_type_from_type_id<static_cast<char>(CK_PARAM_IN_WEI_DATATYPE)>::type;
using FloatAB = typename vector_type<TInWei, InWeiVectorSize>::type;

using FloatC = typename get_type_from_type_id<static_cast<char>(CK_PARAM_OUT_DATATYPE)>::type;
using FloatAcc = typename get_type_from_type_id<static_cast<char>(CK_PARAM_CONV_COMPTYPE)>::type;

constexpr index_t BlockSize = CK_PARAM_BlockSize; 
constexpr index_t N0 = CK_PARAM_N0; 

constexpr index_t GM1PerBlockGM11 = CK_PARAM_GM1PerBlockGM11; 
constexpr index_t GN1PerBlockGN11 = CK_PARAM_GN1PerBlockGN11; 
constexpr index_t KPerBlock = CK_PARAM_KPerBlock; 
constexpr index_t M1PerThread = CK_PARAM_M1PerThread; 
constexpr index_t N1PerThread = CK_PARAM_N1PerThread; 
constexpr index_t KPerThread = CK_PARAM_KPerThread; 
constexpr index_t M1N1ThreadClusterM10 = CK_PARAM_M1N1ThreadClusterM10;
constexpr index_t M1N1ThreadClusterN10 = CK_PARAM_M1N1ThreadClusterN10;
constexpr index_t M1N1ThreadClusterM11 = CK_PARAM_M1N1ThreadClusterM11;
constexpr index_t M1N1ThreadClusterN11 = CK_PARAM_M1N1ThreadClusterN11; 

using ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11 = Sequence<CK_PARAM_ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11>;
using ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11 = Sequence<CK_PARAM_ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11>; 
using ABlockTransferThreadClusterArrangeOrder = Sequence<CK_PARAM_ABlockTransferThreadClusterArrangeOrder>;
using ABlockTransferSrcAccessOrder = Sequence<CK_PARAM_ABlockTransferSrcAccessOrder>;

constexpr index_t ABlockTransferSrcVectorDim = CK_PARAM_ABlockTransferSrcVectorDim; 
constexpr index_t ABlockTransferSrcScalarPerVector = CK_PARAM_ABlockTransferSrcScalarPerVector; 
constexpr index_t ABlockTransferDstScalarPerVector_GM11 = CK_PARAM_ABlockTransferDstScalarPerVector_GM11;
constexpr bool AThreadTransferSrcResetCoordinateAfterRun = static_cast<bool>(CK_PARAM_AThreadTransferSrcResetCoordinateAfterRun);

using BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11 = Sequence<CK_PARAM_BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11>;
using BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11 = Sequence<CK_PARAM_BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11>;
using BBlockTransferThreadClusterArrangeOrder = Sequence<CK_PARAM_BBlockTransferThreadClusterArrangeOrder>; 
using BBlockTransferSrcAccessOrder = Sequence<CK_PARAM_BBlockTransferSrcAccessOrder>;

constexpr index_t BBlockTransferSrcVectorDim = CK_PARAM_BBlockTransferSrcVectorDim; 
constexpr index_t BBlockTransferSrcScalarPerVector = CK_PARAM_BBlockTransferSrcScalarPerVector; 
constexpr index_t BBlockTransferDstScalarPerVector_GN11 = CK_PARAM_BBlockTransferDstScalarPerVector_GN11;
constexpr bool BThreadTransferSrcResetCoordinateAfterRun = static_cast<bool>(CK_PARAM_BThreadTransferSrcResetCoordinateAfterRun);

using CThreadTransferSrcDstAccessOrder = Sequence<CK_PARAM_CThreadTransferSrcDstAccessOrder>;
constexpr index_t CThreadTransferSrcDstVectorDim = CK_PARAM_CThreadTransferSrcDstVectorDim;
constexpr index_t CThreadTransferDstScalarPerVector = CK_PARAM_CThreadTransferDstScalarPerVector; 

extern "C" __global__ void dynamic_convolution_forward_implicit_gemm_v4r5_nchw_kcyx_nkhw_prepare(
        int n, int c, int hi, int wi, int k, int y, int x,
        int convStrideH, int convStrideW, int convDilationY, int convDilationX,
        int leftPadH, int leftPadW, int rightPadH, int rightPadW,
        void* p_a_gk_gm0_gm10_gm11_grid_desc, void* p_b_gk_gn0_gn10_gn11_grid_desc,
       	void* p_c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc, void* p_c_blockid_to_gm10_gn10_block_cluster_adaptor)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    const index_t ho = (hi + leftPadH + rightPadH - convDilationY * (y - 1) - 1) / convStrideH + 1;
    const index_t wo = (wi + leftPadW + rightPadW - convDilationX * (x - 1) - 1) / convStrideW + 1;

    const auto in_n_c_hi_wi_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(n, c, hi, wi));
    const auto wei_k_c_y_x_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(k, c, y, x));
    const auto out_n_k_ho_wo_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(n, k, ho, wo));

    const auto descs = transform_forward_convolution_into_contraction_v4r5_nchw_kcyx_nkhw_pad<N0>(wei_k_c_y_x_desc,
                                                                                       in_n_c_hi_wi_desc,
                                                                                       out_n_k_ho_wo_desc,
                                                                                       make_tuple(convStrideH, convStrideW),
                                                                                       make_tuple(convDilationY, convDilationX),
                                                                                       make_tuple(leftPadH, leftPadW),
                                                                                       make_tuple(rightPadH, rightPadW));

    const auto a_gk_gm0_gm1_grid_desc = descs[I0];
    const auto b_gk_gn0_gn1_grid_desc = descs[I1];
    const auto c_gm0_gm1_gn0_gn1_grid_desc = descs[I2];

    using AGKGM0GM1GridDesc = decltype(a_gk_gm0_gm1_grid_desc);
    using BGKGN0GN1GridDesc = decltype(b_gk_gn0_gn1_grid_desc);
    using CGM0GM1GN0GN1GridDesc = decltype(c_gm0_gm1_gn0_gn1_grid_desc);

    using AGridIteratorHacks = decltype(
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{})) );

    using BGridIteratorHacks = decltype(
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{})) );    

    using CGridIteratorHacks = decltype(make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{}),
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{})) );

    using AGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0>;

    using BGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0>; 

    using GridwiseContraction = GridwiseDynamicContraction_km0m1_kn0n1_m0m1n0n1_v1r1<BlockSize,
                                                   FloatAB,
                                                   FloatAcc,
                                                   FloatC,
                                                   InMemoryDataOperation::Set,  /* ToDo tunable */	
                                                   AGKGM0GM1GridDesc,
                                                   BGKGN0GN1GridDesc,
                                                   CGM0GM1GN0GN1GridDesc,
                                                   GM1PerBlockGM11,
                                                   GN1PerBlockGN11,
                                                   KPerBlock,
                                                   M1PerThread,
                                                   N1PerThread,
                                                   KPerThread,
                                                   M1N1ThreadClusterM10,
                                                   M1N1ThreadClusterN10,
                                                   M1N1ThreadClusterM11,
                                                   M1N1ThreadClusterN11,
                                                   ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11,
                                                   ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11,
                                                   ABlockTransferThreadClusterArrangeOrder,
                                                   ABlockTransferSrcAccessOrder,
                                                   ABlockTransferSrcVectorDim,
                                                   ABlockTransferSrcScalarPerVector,
                                                   ABlockTransferDstScalarPerVector_GM11,
                                                   AThreadTransferSrcResetCoordinateAfterRun,
                                                   BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11,
                                                   BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11,
                                                   BBlockTransferThreadClusterArrangeOrder,
                                                   BBlockTransferSrcAccessOrder,
                                                   BBlockTransferSrcVectorDim,
                                                   BBlockTransferSrcScalarPerVector,
                                                   BBlockTransferDstScalarPerVector_GN11,
                                                   BThreadTransferSrcResetCoordinateAfterRun,
                                                   CThreadTransferSrcDstAccessOrder,
                                                   CThreadTransferSrcDstVectorDim,
                                                   CThreadTransferDstScalarPerVector,
                                                   AGridIteratorHacks,
                                                   BGridIteratorHacks,
                                                   CGridIteratorHacks,
                                                   AGridMoveSliceWindowIteratorHacks,
                                                   BGridMoveSliceWindowIteratorHacks>;

    auto a_gk_gm0_gm10_gm11_grid_desc = GridwiseContraction::MakeAGKGM0GM10GM11GridDescriptor(a_gk_gm0_gm1_grid_desc);
    auto b_gk_gn0_gn10_gn11_grid_desc = GridwiseContraction::MakeBGKGN0GN10GN11GridDescriptor(b_gk_gn0_gn1_grid_desc);
    auto c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc = GridwiseContraction::MakeCGM10BM0BM1GN10BN0BN1GridDescriptor(c_gm0_gm1_gn0_gn1_grid_desc);
    auto c_blockid_to_gm10_gn10_block_cluster_adaptor = GridwiseContraction::MakeCBlockIdToGM10GN10BlockClusterAdaptor(c_gm0_gm1_gn0_gn1_grid_desc);

    *static_cast<decltype(a_gk_gm0_gm10_gm11_grid_desc)*>(p_a_gk_gm0_gm10_gm11_grid_desc) = a_gk_gm0_gm10_gm11_grid_desc; 
    *static_cast<decltype(b_gk_gn0_gn10_gn11_grid_desc)*>(p_b_gk_gn0_gn10_gn11_grid_desc) = b_gk_gn0_gn10_gn11_grid_desc;
    *static_cast<decltype(c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc)*>(p_c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc) = c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc;
    *static_cast<decltype(c_blockid_to_gm10_gn10_block_cluster_adaptor)*>(p_c_blockid_to_gm10_gn10_block_cluster_adaptor) = c_blockid_to_gm10_gn10_block_cluster_adaptor;
}; 


extern "C" __global__ void dynamic_convolution_forward_implicit_gemm_v4r5_nchw_kcyx_nkhw(
        int n, int c, int hi, int wi, int k, int y, int x, 
        int convStrideH, int convStrideW, int convDilationY, int convDilationX, 
        int leftPadH, int leftPadW, int rightPadH, int rightPadW,	
        const void* p_a_grid, const void* p_b_grid, void* p_c_grid,
        const void* p_a_gk_gm0_gm10_gm11_grid_desc, const void* p_b_gk_gn0_gn10_gn11_grid_desc, 
	const void* p_c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc, const void* p_c_blockid_to_gm10_gn10_block_cluster_adaptor)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    const index_t ho = (hi + leftPadH + rightPadH - convDilationY * (y - 1) - 1) / convStrideH + 1;
    const index_t wo = (wi + leftPadW + rightPadW - convDilationX * (x - 1) - 1) / convStrideW + 1;

    const auto in_n_c_hi_wi_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(n, c, hi, wi));
    const auto wei_k_c_y_x_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(k, c, y, x));
    const auto out_n_k_ho_wo_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(n, k, ho, wo));

    const auto descs = transform_forward_convolution_into_contraction_v4r5_nchw_kcyx_nkhw_pad<N0>(wei_k_c_y_x_desc,
                                                                                       in_n_c_hi_wi_desc,
                                                                                       out_n_k_ho_wo_desc,
                                                                                       make_tuple(convStrideH, convStrideW),
                                                                                       make_tuple(convDilationY, convDilationX),
                                                                                       make_tuple(leftPadH, leftPadW),
                                                                                       make_tuple(rightPadH, rightPadW));

    const auto a_gk_gm0_gm1_grid_desc = descs[I0];
    const auto b_gk_gn0_gn1_grid_desc = descs[I1];
    const auto c_gm0_gm1_gn0_gn1_grid_desc = descs[I2];

    using AGKGM0GM1GridDesc = decltype(a_gk_gm0_gm1_grid_desc);
    using BGKGN0GN1GridDesc = decltype(b_gk_gn0_gn1_grid_desc);
    using CGM0GM1GN0GN1GridDesc = decltype(c_gm0_gm1_gn0_gn1_grid_desc);

    using AGridIteratorHacks = decltype(
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0>{})) );

    using BGridIteratorHacks = decltype(
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{})) );

    using CGridIteratorHacks = decltype(make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>{}),
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{},
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{})) );

    using AGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0>;
    using BGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0>;

    using GridwiseContraction = GridwiseDynamicContraction_km0m1_kn0n1_m0m1n0n1_v1r1<
        BlockSize,
        FloatAB,
        FloatAcc,
        FloatC,
        InMemoryDataOperation::Set,  /* ToDo tunable */	
        AGKGM0GM1GridDesc,
        BGKGN0GN1GridDesc,
        CGM0GM1GN0GN1GridDesc,
        GM1PerBlockGM11,
        GN1PerBlockGN11,
        KPerBlock,
        M1PerThread,
        N1PerThread,
        KPerThread,
        M1N1ThreadClusterM10,
        M1N1ThreadClusterN10,
        M1N1ThreadClusterM11,
        M1N1ThreadClusterN11,
        ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11,
        ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_GM11,
        AThreadTransferSrcResetCoordinateAfterRun,
        BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11,
        BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_GN11,
        BThreadTransferSrcResetCoordinateAfterRun,
        CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector,
        AGridIteratorHacks,
        BGridIteratorHacks,
        CGridIteratorHacks,
        AGridMoveSliceWindowIteratorHacks,
        BGridMoveSliceWindowIteratorHacks>;    

    const auto K = a_gk_gm0_gm1_grid_desc.GetLength(I0);

    using AGKGM0GM10GM11GridDesc = decltype(GridwiseContraction::MakeAGKGM0GM10GM11GridDescriptor(a_gk_gm0_gm1_grid_desc));
    using BGKGN0GN10GN11GridDesc = decltype(GridwiseContraction::MakeBGKGN0GN10GN11GridDescriptor(b_gk_gn0_gn1_grid_desc));
    using CGM10BM0BM1GN10BN0BN1GridDesc = decltype(GridwiseContraction::MakeCGM10BM0BM1GN10BN0BN1GridDescriptor(c_gm0_gm1_gn0_gn1_grid_desc));
    using CBlockIdToGM10GN10BlockClusterAdaptor = decltype(GridwiseContraction::MakeCBlockIdToGM10GN10BlockClusterAdaptor(c_gm0_gm1_gn0_gn1_grid_desc));

    const auto a_gk_gm0_gm10_gm11_grid_desc = *static_cast<const AGKGM0GM10GM11GridDesc *>(p_a_gk_gm0_gm10_gm11_grid_desc); 
    const auto b_gk_gn0_gn10_gn11_grid_desc = *static_cast<const BGKGN0GN10GN11GridDesc *>(p_b_gk_gn0_gn10_gn11_grid_desc);
    const auto c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc = *static_cast<const CGM10BM0BM1GN10BN0BN1GridDesc *>(p_c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc);
    const auto c_blockid_to_gm10_gn10_block_cluster_adaptor = *static_cast<const CBlockIdToGM10GN10BlockClusterAdaptor *>(p_c_blockid_to_gm10_gn10_block_cluster_adaptor); 

    const bool has_main_k_block_loop = GridwiseContraction::CalculateHasMainKBlockLoop(K);
    const bool has_double_tail_k_block_loop = GridwiseContraction::CalculateHasDoubleTailKBlockLoop(K);

    if(has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_contraction_v1r1<GridwiseContraction,
                                                            FloatAB,
                                                            FloatC,
                                                            remove_reference_t<AGKGM0GM10GM11GridDesc>,
                                                            remove_reference_t<BGKGN0GN10GN11GridDesc>,
                                                            remove_reference_t<CGM10BM0BM1GN10BN0BN1GridDesc>,
                                                            remove_reference_t<CBlockIdToGM10GN10BlockClusterAdaptor>,
                                                            true,
                                                            true>;

        kernel(static_cast<const FloatAB*>(p_a_grid), static_cast<const FloatAB*>(p_b_grid), static_cast<FloatC*>(p_c_grid),
			   a_gk_gm0_gm10_gm11_grid_desc, b_gk_gn0_gn10_gn11_grid_desc, c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc, c_blockid_to_gm10_gn10_block_cluster_adaptor);
    }
    else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_contraction_v1r1<GridwiseContraction,
                                                            FloatAB,
                                                            FloatC,
                                                            remove_reference_t<AGKGM0GM10GM11GridDesc>,
                                                            remove_reference_t<BGKGN0GN10GN11GridDesc>,
                                                            remove_reference_t<CGM10BM0BM1GN10BN0BN1GridDesc>,
                                                            remove_reference_t<CBlockIdToGM10GN10BlockClusterAdaptor>,
                                                            true,
                                                            false>;

        kernel(static_cast<const FloatAB*>(p_a_grid), static_cast<const FloatAB*>(p_b_grid), static_cast<FloatC*>(p_c_grid),
                           a_gk_gm0_gm10_gm11_grid_desc, b_gk_gn0_gn10_gn11_grid_desc, c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc, c_blockid_to_gm10_gn10_block_cluster_adaptor);
    }
    else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_contraction_v1r1<GridwiseContraction,
                                                            FloatAB,
                                                            FloatC,
                                                            remove_reference_t<AGKGM0GM10GM11GridDesc>,
                                                            remove_reference_t<BGKGN0GN10GN11GridDesc>,
                                                            remove_reference_t<CGM10BM0BM1GN10BN0BN1GridDesc>,
                                                            remove_reference_t<CBlockIdToGM10GN10BlockClusterAdaptor>,
                                                            false,
                                                            true>;

        kernel(static_cast<const FloatAB*>(p_a_grid), static_cast<const FloatAB*>(p_b_grid), static_cast<FloatC*>(p_c_grid),
                           a_gk_gm0_gm10_gm11_grid_desc, b_gk_gn0_gn10_gn11_grid_desc, c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc, c_blockid_to_gm10_gn10_block_cluster_adaptor);
    }
    else
    {
        const auto kernel = kernel_dynamic_contraction_v1r1<GridwiseContraction,
                                                            FloatAB,
                                                            FloatC,
                                                            remove_reference_t<AGKGM0GM10GM11GridDesc>,
                                                            remove_reference_t<BGKGN0GN10GN11GridDesc>,
                                                            remove_reference_t<CGM10BM0BM1GN10BN0BN1GridDesc>,
                                                            remove_reference_t<CBlockIdToGM10GN10BlockClusterAdaptor>,
                                                            false,
                                                            false>;

        kernel(static_cast<const FloatAB*>(p_a_grid), static_cast<const FloatAB*>(p_b_grid), static_cast<FloatC*>(p_c_grid),
                           a_gk_gm0_gm10_gm11_grid_desc, b_gk_gn0_gn10_gn11_grid_desc, c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc, c_blockid_to_gm10_gn10_block_cluster_adaptor);
    }
};

