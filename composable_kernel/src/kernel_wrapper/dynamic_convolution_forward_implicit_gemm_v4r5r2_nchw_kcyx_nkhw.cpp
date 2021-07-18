#include "common_header.hpp"
#include "type_helper.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_contraction_v1r1.hpp"
#include "transform_forward_convolution_into_gemm_v4r5_nchw_kcyx_nkhw.hpp"

using namespace ck;

using FloatAB  = typename get_type_from_type_id<static_cast<char>(CK_PARAM_IN_WEI_DATATYPE)>::type;
using FloatC   = typename get_type_from_type_id<static_cast<char>(CK_PARAM_OUT_DATATYPE)>::type;
using FloatAcc = typename get_type_from_type_id<static_cast<char>(CK_PARAM_CONV_COMPTYPE)>::type;

constexpr index_t BlockSize = CK_PARAM_BlockSize;
constexpr index_t N0        = CK_PARAM_N0;

constexpr index_t GM1PerBlockGM11            = CK_PARAM_GM1PerBlockGM11;
constexpr index_t GN1PerBlockGN11            = CK_PARAM_GN1PerBlockGN11;
constexpr index_t GK0PerBlock                = CK_PARAM_GK0PerBlock;
constexpr index_t BM1PerThreadBM11           = CK_PARAM_BM1PerThreadBM11;
constexpr index_t BN1PerThreadBN11           = CK_PARAM_BN1PerThreadBN11;
constexpr index_t BK0PerThread               = CK_PARAM_BK0PerThread;
constexpr index_t BM10BN10ThreadClusterBM100 = CK_PARAM_BM10BN10ThreadClusterBM100;
constexpr index_t BM10BN10ThreadClusterBN100 = CK_PARAM_BM10BN10ThreadClusterBN100;
constexpr index_t BM10BN10ThreadClusterBM101 = CK_PARAM_BM10BN10ThreadClusterBM101;
constexpr index_t BM10BN10ThreadClusterBN101 = CK_PARAM_BM10BN10ThreadClusterBN101;

using ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11 =
    Sequence<CK_PARAM_ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11>;
using ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11 =
    Sequence<CK_PARAM_ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11>;
using ABlockTransferThreadClusterArrangeOrder =
    Sequence<CK_PARAM_ABlockTransferThreadClusterArrangeOrder>;
using ABlockTransferSrcAccessOrder = Sequence<CK_PARAM_ABlockTransferSrcAccessOrder>;

constexpr index_t ABlockTransferSrcVectorDim       = CK_PARAM_ABlockTransferSrcVectorDim;
constexpr index_t ABlockTransferSrcScalarPerVector = CK_PARAM_ABlockTransferSrcScalarPerVector;
constexpr index_t ABlockTransferDstScalarPerVector_GM11 =
    CK_PARAM_ABlockTransferDstScalarPerVector_GM11;
constexpr bool AThreadTransferSrcResetCoordinateAfterRun =
    static_cast<bool>(CK_PARAM_AThreadTransferSrcResetCoordinateAfterRun);

using BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11 =
    Sequence<CK_PARAM_BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11>;
using BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11 =
    Sequence<CK_PARAM_BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11>;
using BBlockTransferThreadClusterArrangeOrder =
    Sequence<CK_PARAM_BBlockTransferThreadClusterArrangeOrder>;
using BBlockTransferSrcAccessOrder = Sequence<CK_PARAM_BBlockTransferSrcAccessOrder>;

constexpr index_t BBlockTransferSrcVectorDim       = CK_PARAM_BBlockTransferSrcVectorDim;
constexpr index_t BBlockTransferSrcScalarPerVector = CK_PARAM_BBlockTransferSrcScalarPerVector;
constexpr index_t BBlockTransferDstScalarPerVector_GN11 =
    CK_PARAM_BBlockTransferDstScalarPerVector_GN11;
constexpr bool BThreadTransferSrcResetCoordinateAfterRun =
    static_cast<bool>(CK_PARAM_BThreadTransferSrcResetCoordinateAfterRun);

using CThreadTransferSrcDstAccessOrder = Sequence<CK_PARAM_CThreadTransferSrcDstAccessOrder>;
constexpr index_t CThreadTransferSrcDstVectorDim    = CK_PARAM_CThreadTransferSrcDstVectorDim;
constexpr index_t CThreadTransferDstScalarPerVector = CK_PARAM_CThreadTransferDstScalarPerVector;

constexpr bool HasMainKBlockLoop       = static_cast<bool>(CK_PARAM_HAS_MAIN_KBLOCK_LOOP);
constexpr bool HasDoubleTailKBlockLoop = static_cast<bool>(CK_PARAM_HAS_DOUBLE_TAIL_KBLOCK_LOOP);

extern "C" __global__ void dynamic_convolution_forward_implicit_gemm_v4r5r2_nchw_kcyx_nkhw_prepare(
    index_t N,
    index_t C,
    index_t Hi,
    index_t Wi,
    index_t K,
    index_t Y,
    index_t X,
    index_t ConvStrideH,
    index_t ConvStrideW,
    index_t ConvDilationH,
    index_t ConvDilationW,
    index_t InLeftPadH,
    index_t InLeftPadW,
    index_t InRightPadH,
    index_t InRightPadW,
    void* p_a_gk_gm0_gm10_gm11_grid_desc,
    void* p_b_gk_gn0_gn10_gn11_grid_desc,
    void* p_c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
    void* p_c_blockid_to_gm10_gn10_block_cluster_adaptor)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    const index_t Ho =
        (Hi + InLeftPadH + InRightPadH - ConvDilationH * (Y - 1) - 1) / ConvStrideH + 1;
    const index_t Wo =
        (Wi + InLeftPadW + InRightPadW - ConvDilationW * (X - 1) - 1) / ConvStrideW + 1;

    const auto in_n_c_hi_wi_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, C, Hi, Wi));
    const auto wei_k_c_y_x_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, C, Y, X));
    const auto out_n_k_ho_wo_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, K, Ho, Wo));

    const auto descs = transform_forward_convolution_into_contraction_v4r5_nchw_kcyx_nkhw_pad<N0>(
        wei_k_c_y_x_desc,
        in_n_c_hi_wi_desc,
        out_n_k_ho_wo_desc,
        make_tuple(ConvStrideH, ConvStrideW),
        make_tuple(ConvDilationH, ConvDilationW),
        make_tuple(InLeftPadH, InLeftPadW),
        make_tuple(InRightPadH, InRightPadW));

    const auto a_gk_gm0_gm1_grid_desc      = descs[I0];
    const auto b_gk_gn0_gn1_grid_desc      = descs[I1];
    const auto c_gm0_gm1_gn0_gn1_grid_desc = descs[I2];

    using AGKGM0GM1GridDesc     = decltype(a_gk_gm0_gm1_grid_desc);
    using BGKGN0GN1GridDesc     = decltype(b_gk_gn0_gn1_grid_desc);
    using CGM0GM1GN0GN1GridDesc = decltype(c_gm0_gm1_gn0_gn1_grid_desc);

    using AGridIteratorHacks = decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{}),
                                                   make_tuple(Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{})));

    using BGridIteratorHacks =
        decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{}),
                            make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{})));

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
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{})));

    using AGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0>;

    using BGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0>;

    using GridwiseContraction = GridwiseDynamicContraction_km0m1_kn0n1_m0m1n0n1_v1r1<
        BlockSize,
        FloatAB,
        FloatAcc,
        FloatC,
        InMemoryDataOperation::Set, /* ToDo tunable */
        AGKGM0GM1GridDesc,
        BGKGN0GN1GridDesc,
        CGM0GM1GN0GN1GridDesc,
        GM1PerBlockGM11,
        GN1PerBlockGN11,
        GK0PerBlock,
        BM1PerThreadBM11,
        BN1PerThreadBN11,
        BK0PerThread,
        BM10BN10ThreadClusterBM100,
        BM10BN10ThreadClusterBN100,
        BM10BN10ThreadClusterBM101,
        BM10BN10ThreadClusterBN101,
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

    auto a_gk_gm0_gm10_gm11_grid_desc =
        GridwiseContraction::MakeAGKGM0GM10GM11GridDescriptor(a_gk_gm0_gm1_grid_desc);
    auto b_gk_gn0_gn10_gn11_grid_desc =
        GridwiseContraction::MakeBGKGN0GN10GN11GridDescriptor(b_gk_gn0_gn1_grid_desc);
    auto c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc =
        GridwiseContraction::MakeCGM10BM0BM1GN10BN0BN1GridDescriptor(c_gm0_gm1_gn0_gn1_grid_desc);
    auto c_blockid_to_gm10_gn10_block_cluster_adaptor =
        GridwiseContraction::MakeCBlockIdToGM10GN10BlockClusterAdaptor(c_gm0_gm1_gn0_gn1_grid_desc);

    if(hipThreadIdx_x == 0)
    {
        *static_cast<decltype(a_gk_gm0_gm10_gm11_grid_desc)*>(p_a_gk_gm0_gm10_gm11_grid_desc) =
            a_gk_gm0_gm10_gm11_grid_desc;
        *static_cast<decltype(b_gk_gn0_gn10_gn11_grid_desc)*>(p_b_gk_gn0_gn10_gn11_grid_desc) =
            b_gk_gn0_gn10_gn11_grid_desc;
        *static_cast<decltype(c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc)*>(
            p_c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc) = c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc;
        *static_cast<decltype(c_blockid_to_gm10_gn10_block_cluster_adaptor)*>(
            p_c_blockid_to_gm10_gn10_block_cluster_adaptor) =
            c_blockid_to_gm10_gn10_block_cluster_adaptor;
    };
};

extern "C" __global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        dynamic_convolution_forward_implicit_gemm_v4r5r2_nchw_kcyx_nkhw(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const void __CONSTANT__* p_a_gk_gm0_gm10_gm11_grid_desc,
            const void __CONSTANT__* p_b_gk_gn0_gn10_gn11_grid_desc,
            const void __CONSTANT__* p_c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
            const void __CONSTANT__* p_c_blockid_to_gm10_gn10_block_cluster_adaptor)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    constexpr auto in_n_c_hi_wi_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(256, 256, 28, 28));
    constexpr auto wei_k_c_y_x_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(256, 256, 3, 3));
    constexpr auto out_n_k_ho_wo_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(256, 256, 28, 28));

    constexpr auto descs =
        transform_forward_convolution_into_contraction_v4r5_nchw_kcyx_nkhw_pad<N0>(
            wei_k_c_y_x_desc,
            in_n_c_hi_wi_desc,
            out_n_k_ho_wo_desc,
            make_tuple(1, 1),
            make_tuple(1, 1),
            make_tuple(1, 1),
            make_tuple(1, 1));

    constexpr auto a_gk_gm0_gm1_grid_desc      = descs[I0];
    constexpr auto b_gk_gn0_gn1_grid_desc      = descs[I1];
    constexpr auto c_gm0_gm1_gn0_gn1_grid_desc = descs[I2];

    using AGKGM0GM1GridDesc     = decltype(a_gk_gm0_gm1_grid_desc);
    using BGKGN0GN1GridDesc     = decltype(b_gk_gn0_gn1_grid_desc);
    using CGM0GM1GN0GN1GridDesc = decltype(c_gm0_gm1_gn0_gn1_grid_desc);

    using AGridIteratorHacks = decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{}),
                                                   make_tuple(Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0, 0>{})));

    using BGridIteratorHacks =
        decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{}),
                            make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{})));

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
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>{})));

    using AGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0>;
    using BGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0>;

    using GridwiseContraction = GridwiseDynamicContraction_km0m1_kn0n1_m0m1n0n1_v1r1<
        BlockSize,
        FloatAB,
        FloatAcc,
        FloatC,
        InMemoryDataOperation::Set, /* ToDo tunable */
        AGKGM0GM1GridDesc,
        BGKGN0GN1GridDesc,
        CGM0GM1GN0GN1GridDesc,
        GM1PerBlockGM11,
        GN1PerBlockGN11,
        GK0PerBlock,
        BM1PerThreadBM11,
        BN1PerThreadBN11,
        BK0PerThread,
        BM10BN10ThreadClusterBM100,
        BM10BN10ThreadClusterBN100,
        BM10BN10ThreadClusterBM101,
        BM10BN10ThreadClusterBN101,
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

    using AGKGM0GM10GM11GridDesc =
        decltype(GridwiseContraction::MakeAGKGM0GM10GM11GridDescriptor(a_gk_gm0_gm1_grid_desc));
    using BGKGN0GN10GN11GridDesc =
        decltype(GridwiseContraction::MakeBGKGN0GN10GN11GridDescriptor(b_gk_gn0_gn1_grid_desc));
    using CGM10BM0BM1GN10BN0BN1GridDesc = decltype(
        GridwiseContraction::MakeCGM10BM0BM1GN10BN0BN1GridDescriptor(c_gm0_gm1_gn0_gn1_grid_desc));
    using CBlockIdToGM10GN10BlockClusterAdaptor =
        decltype(GridwiseContraction::MakeCBlockIdToGM10GN10BlockClusterAdaptor(
            c_gm0_gm1_gn0_gn1_grid_desc));

    const auto a_gk_gm0_gm10_gm11_grid_desc = *reinterpret_cast<const AGKGM0GM10GM11GridDesc*>(
        (const void*)p_a_gk_gm0_gm10_gm11_grid_desc);
    const auto b_gk_gn0_gn10_gn11_grid_desc = *reinterpret_cast<const BGKGN0GN10GN11GridDesc*>(
        (const void*)p_b_gk_gn0_gn10_gn11_grid_desc);
    const auto c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc =
        *reinterpret_cast<const CGM10BM0BM1GN10BN0BN1GridDesc*>(
            (const void*)p_c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc);
    const auto c_blockid_to_gm10_gn10_block_cluster_adaptor =
        *reinterpret_cast<const CBlockIdToGM10GN10BlockClusterAdaptor*>(
            (const void*)p_c_blockid_to_gm10_gn10_block_cluster_adaptor);

    constexpr index_t shared_block_size =
        GridwiseContraction::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseContraction::Run(p_a_grid,
                             p_b_grid,
                             p_c_grid,
                             p_shared_block,
                             a_gk_gm0_gm10_gm11_grid_desc,
                             b_gk_gn0_gn10_gn11_grid_desc,
                             c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
                             c_blockid_to_gm10_gn10_block_cluster_adaptor,
                             integral_constant<bool, HasMainKBlockLoop>{},
                             integral_constant<bool, HasDoubleTailKBlockLoop>{});
};
