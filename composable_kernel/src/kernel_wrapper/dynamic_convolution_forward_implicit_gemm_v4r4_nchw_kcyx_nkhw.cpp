#include "common_header.hpp"
#include "type_helper.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm_v1r2_olc.hpp"
#include "transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw.hpp"

using namespace ck; 

using FloatAB = typename get_type_from_type_id<static_cast<char>(CK_PARAM_IN_WEI_DATATYPE)>::type;
using FloatC = typename get_type_from_type_id<static_cast<char>(CK_PARAM_OUT_DATATYPE)>::type;
using FloatAcc = typename get_type_from_type_id<static_cast<char>(CK_PARAM_CONV_COMPTYPE)>::type;

constexpr index_t BlockSize = CK_PARAM_BlockSize; 

constexpr index_t MPerBlock = CK_PARAM_MPerBlock; 
constexpr index_t NPerBlock = CK_PARAM_NPerBlock; 
constexpr index_t KPerBlock = CK_PARAM_KPerBlock; 
constexpr index_t M1PerThread = CK_PARAM_M1PerThread; 
constexpr index_t N1PerThread = CK_PARAM_N1PerThread; 
constexpr index_t KPerThread = CK_PARAM_KPerThread; 
constexpr index_t M1N1ThreadClusterM10 = CK_PARAM_M1N1ThreadClusterM10;
constexpr index_t M1N1ThreadClusterN10 = CK_PARAM_M1N1ThreadClusterN10;
constexpr index_t M1N1ThreadClusterM11 = CK_PARAM_M1N1ThreadClusterM11;
constexpr index_t M1N1ThreadClusterN11 = CK_PARAM_M1N1ThreadClusterN11; 

using ABlockTransferThreadSliceLengths_K_M0_M1 = Sequence<CK_PARAM_ABlockTransferThreadSliceLengths_K_M0_M1>;
using ABlockTransferThreadClusterLengths_K_M0_M1 = Sequence<CK_PARAM_ABlockTransferThreadClusterLengths_K_M0_M1>; 
using ABlockTransferThreadClusterArrangeOrder = Sequence<CK_PARAM_ABlockTransferThreadClusterArrangeOrder>;
using ABlockTransferSrcAccessOrder = Sequence<CK_PARAM_ABlockTransferSrcAccessOrder>;

constexpr index_t ABlockTransferSrcVectorDim = CK_PARAM_ABlockTransferSrcVectorDim; 
constexpr index_t ABlockTransferSrcScalarPerVector = CK_PARAM_ABlockTransferSrcScalarPerVector; 
constexpr index_t ABlockTransferDstScalarPerVector_M1 = CK_PARAM_ABlockTransferDstScalarPerVector_M1;
constexpr bool AThreadTransferSrcResetCoordinateAfterRun = static_cast<bool>(CK_PARAM_AThreadTransferSrcResetCoordinateAfterRun);

using BBlockTransferThreadSliceLengths_K_N0_N1 = Sequence<CK_PARAM_BBlockTransferThreadSliceLengths_K_N0_N1>;
using BBlockTransferThreadClusterLengths_K_N0_N1 = Sequence<CK_PARAM_BBlockTransferThreadClusterLengths_K_N0_N1>;
using BBlockTransferThreadClusterArrangeOrder = Sequence<CK_PARAM_BBlockTransferThreadClusterArrangeOrder>; 
using BBlockTransferSrcAccessOrder = Sequence<CK_PARAM_BBlockTransferSrcAccessOrder>;

constexpr index_t BBlockTransferSrcVectorDim = CK_PARAM_BBlockTransferSrcVectorDim; 
constexpr index_t BBlockTransferSrcScalarPerVector = CK_PARAM_BBlockTransferSrcScalarPerVector; 
constexpr index_t BBlockTransferDstScalarPerVector_N1 = CK_PARAM_BBlockTransferDstScalarPerVector_N1;
constexpr bool BThreadTransferSrcResetCoordinateAfterRun = static_cast<bool>(CK_PARAM_BThreadTransferSrcResetCoordinateAfterRun);

using CThreadTransferSrcDstAccessOrder = Sequence<CK_PARAM_CThreadTransferSrcDstAccessOrder>;
constexpr index_t CThreadTransferSrcDstVectorDim = CK_PARAM_CThreadTransferSrcDstVectorDim;
constexpr index_t CThreadTransferDstScalarPerVector = CK_PARAM_CThreadTransferDstScalarPerVector; 

constexpr bool hasMainKBlockLoop = static_cast<bool>(CK_PARAM_HAS_MAIN_KBLOCK_LOOP); 
constexpr bool hasDoubleTailKBlockLoop = static_cast<bool>(CK_PARAM_HAS_DOUBLE_TAIL_KBLOCK_LOOP);  

extern "C" __global__ void dynamic_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw_prepare(
        int n, int c, int hi, int wi, int k, int y, int x,
        int convStrideH, int convStrideW, int convDilationY, int convDilationX,
        int leftPadH, int leftPadW, int rightPadH, int rightPadW,
        void* p_a_k_m0_m1_grid_desc, void* p_b_k_n0_n1_grid_desc, void* p_c_m0_m10_m11_n0_n10_n11_grid_desc, void* p_c_blockid_to_m0_n0_block_cluster_adaptor)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    const index_t ho = (hi + leftPadH + rightPadH - convDilationY * (y - 1) - 1) / convStrideH + 1;
    const index_t wo = (wi + leftPadW + rightPadW - convDilationX * (x - 1) - 1) / convStrideW + 1;

    const auto in_n_c_hi_wi_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(n, c, hi, wi));
    const auto wei_k_c_y_x_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(k, c, y, x));
    const auto out_n_k_ho_wo_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(n, k, ho, wo));

    const auto descs = transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw_pad(wei_k_c_y_x_desc,
                                                                                       in_n_c_hi_wi_desc,
                                                                                       out_n_k_ho_wo_desc,
                                                                                       make_tuple(convStrideH, convStrideW),
                                                                                       make_tuple(convDilationY, convDilationX),
                                                                                       make_tuple(leftPadH, leftPadW),
                                                                                       make_tuple(rightPadH, rightPadW));

    const auto a_k_m_grid_desc = descs[I0];
    const auto b_k_n_grid_desc = descs[I1];
    const auto c_m_n_grid_desc = descs[I2];

    using AKMGridDesc = decltype(a_k_m_grid_desc);
    using BKNGridDesc = decltype(b_k_n_grid_desc);
    using CMNGridDesc = decltype(c_m_n_grid_desc);

    using AGridIteratorHacks = decltype(
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{})) );

    using BGridIteratorHacks = decltype(
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{})) );

    using CGridIteratorHacks = decltype(
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{})) );

    using AGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0>;
    using BGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>;

    using GridwiseGemm = GridwiseDynamicGemm_km_kn_mn_v1r2<BlockSize,
                                          FloatAB,
                                          FloatAcc,
                                          FloatC,
                                          InMemoryDataOperation::Set,  /* ToDo tunable */ 
                                          AKMGridDesc,
                                          BKNGridDesc,
                                          CMNGridDesc,
                                          MPerBlock,
                                          NPerBlock,
                                          KPerBlock,
                                          M1PerThread,
                                          N1PerThread,
                                          KPerThread,
                                          M1N1ThreadClusterM10,
                                          M1N1ThreadClusterN10,
                                          M1N1ThreadClusterM11,
                                          M1N1ThreadClusterN11,
                                          ABlockTransferThreadSliceLengths_K_M0_M1,
                                          ABlockTransferThreadClusterLengths_K_M0_M1,
                                          ABlockTransferThreadClusterArrangeOrder,
                                          ABlockTransferSrcAccessOrder,
                                          ABlockTransferSrcVectorDim,
                                          ABlockTransferSrcScalarPerVector,
                                          ABlockTransferDstScalarPerVector_M1,
                                          AThreadTransferSrcResetCoordinateAfterRun,
                                          BBlockTransferThreadSliceLengths_K_N0_N1,
                                          BBlockTransferThreadClusterLengths_K_N0_N1,
                                          BBlockTransferThreadClusterArrangeOrder,
                                          BBlockTransferSrcAccessOrder,
                                          BBlockTransferSrcVectorDim,
                                          BBlockTransferSrcScalarPerVector,
                                          BBlockTransferDstScalarPerVector_N1,
                                          BThreadTransferSrcResetCoordinateAfterRun,
                                          CThreadTransferSrcDstAccessOrder,
                                          CThreadTransferSrcDstVectorDim,
                                          CThreadTransferDstScalarPerVector,
                                          AGridIteratorHacks,
                                          BGridIteratorHacks,
                                          CGridIteratorHacks,
                                          AGridMoveSliceWindowIteratorHacks,
                                          BGridMoveSliceWindowIteratorHacks>;

    auto a_k_m0_m1_grid_desc = GridwiseGemm::MakeAKM0M1GridDescriptor(a_k_m_grid_desc);
    auto b_k_n0_n1_grid_desc = GridwiseGemm::MakeBKN0N1GridDescriptor(b_k_n_grid_desc);
    auto c_m0_m10_m11_n0_n10_n11_grid_desc = GridwiseGemm::MakeCM0M10M11N0N10N11GridDescriptor(c_m_n_grid_desc);
    auto c_blockid_to_m0_n0_block_cluster_adaptor = GridwiseGemm::MakeCBlockIdToM0N0BlockClusterAdaptor(c_m_n_grid_desc);

    if ( hipThreadIdx_x == 0 ) {
        *static_cast<decltype(a_k_m0_m1_grid_desc)*>(p_a_k_m0_m1_grid_desc) = a_k_m0_m1_grid_desc; 
        *static_cast<decltype(b_k_n0_n1_grid_desc)*>(p_b_k_n0_n1_grid_desc) = b_k_n0_n1_grid_desc;
        *static_cast<decltype(c_m0_m10_m11_n0_n10_n11_grid_desc)*>(p_c_m0_m10_m11_n0_n10_n11_grid_desc) = c_m0_m10_m11_n0_n10_n11_grid_desc; 
        *static_cast<decltype(c_blockid_to_m0_n0_block_cluster_adaptor)*>(p_c_blockid_to_m0_n0_block_cluster_adaptor) = c_blockid_to_m0_n0_block_cluster_adaptor; 
    }; 
}; 


extern "C" __global__ void dynamic_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw(
        int n, int c, int hi, int wi, int k, int y, int x, 
        int convStrideH, int convStrideW, int convDilationY, int convDilationX, 
        int leftPadH, int leftPadW, int rightPadH, int rightPadW,	
        const void* p_a_grid, const void* p_b_grid, void* p_c_grid,
	const void __CONSTANT__ *p_a_k_m0_m1_grid_desc, const void __CONSTANT__ *p_b_k_n0_n1_grid_desc, 
	const void __CONSTANT__ *p_c_m0_m10_m11_n0_n10_n11_grid_desc, const void __CONSTANT__ *p_c_blockid_to_m0_n0_block_cluster_adaptor)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    constexpr auto in_n_c_hi_wi_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(64, 4, 35, 35));
    constexpr auto wei_k_c_y_x_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(8, 4, 3, 3));
    constexpr auto out_n_k_ho_wo_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(64, 8, 18, 18));

    constexpr auto descs = transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw_pad(wei_k_c_y_x_desc,
                                                                                       in_n_c_hi_wi_desc,
                                                                                       out_n_k_ho_wo_desc,
                                                                                       make_tuple(2, 2),
                                                                                       make_tuple(1, 1),
                                                                                       make_tuple(1, 1),
                                                                                       make_tuple(1, 1));

    constexpr auto a_k_m_grid_desc = descs[I0];
    constexpr auto b_k_n_grid_desc = descs[I1];
    constexpr auto c_m_n_grid_desc = descs[I2];

    using AKMGridDesc = decltype(a_k_m_grid_desc); 
    using BKNGridDesc = decltype(b_k_n_grid_desc); 
    using CMNGridDesc = decltype(c_m_n_grid_desc); 

    using AGridIteratorHacks = decltype(
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{})) );

    using BGridIteratorHacks = decltype(
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{})) );

    using CGridIteratorHacks = decltype(
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{})) );

    using AGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0>;
    using BGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>;

    using GridwiseGemm = GridwiseDynamicGemm_km_kn_mn_v1r2<BlockSize,
                                          FloatAB,
                                          FloatAcc,
                                          FloatC,
                                          InMemoryDataOperation::Set,  /* ToDo tunable */ 
                                          AKMGridDesc,
                                          BKNGridDesc,
                                          CMNGridDesc,
                                          MPerBlock,
                                          NPerBlock,
                                          KPerBlock,
                                          M1PerThread,
                                          N1PerThread,
                                          KPerThread,
                                          M1N1ThreadClusterM10,
                                          M1N1ThreadClusterN10,
                                          M1N1ThreadClusterM11,
                                          M1N1ThreadClusterN11,
                                          ABlockTransferThreadSliceLengths_K_M0_M1,
                                          ABlockTransferThreadClusterLengths_K_M0_M1,
                                          ABlockTransferThreadClusterArrangeOrder,
                                          ABlockTransferSrcAccessOrder,
                                          ABlockTransferSrcVectorDim,
                                          ABlockTransferSrcScalarPerVector,
                                          ABlockTransferDstScalarPerVector_M1,
                                          AThreadTransferSrcResetCoordinateAfterRun,
                                          BBlockTransferThreadSliceLengths_K_N0_N1,
                                          BBlockTransferThreadClusterLengths_K_N0_N1,
                                          BBlockTransferThreadClusterArrangeOrder,
                                          BBlockTransferSrcAccessOrder,
                                          BBlockTransferSrcVectorDim,
                                          BBlockTransferSrcScalarPerVector,
                                          BBlockTransferDstScalarPerVector_N1,
                                          BThreadTransferSrcResetCoordinateAfterRun,
                                          CThreadTransferSrcDstAccessOrder,
                                          CThreadTransferSrcDstVectorDim,
                                          CThreadTransferDstScalarPerVector,
                                          AGridIteratorHacks,
                                          BGridIteratorHacks,
                                          CGridIteratorHacks,
                                          AGridMoveSliceWindowIteratorHacks,
                                          BGridMoveSliceWindowIteratorHacks>;

    using AKM0M1GridDesc = decltype( GridwiseGemm::MakeAKM0M1GridDescriptor(a_k_m_grid_desc) );
    using BKN0N1GridDesc = decltype( GridwiseGemm::MakeBKN0N1GridDescriptor(b_k_n_grid_desc) ); 
    using CM0M10M11N0N10N11GridDesc = decltype( GridwiseGemm::MakeCM0M10M11N0N10N11GridDescriptor(c_m_n_grid_desc) );
    using CBlockIdToM0N0BlockClusterAdaptor = decltype( GridwiseGemm::MakeCBlockIdToM0N0BlockClusterAdaptor(c_m_n_grid_desc) ); 

    const auto a_k_m0_m1_grid_desc = *reinterpret_cast<const AKM0M1GridDesc *>((const void*)p_a_k_m0_m1_grid_desc); 
    const auto b_k_n0_n1_grid_desc = *reinterpret_cast<const BKN0N1GridDesc *>((const void*)p_b_k_n0_n1_grid_desc); 
    const auto c_m0_m10_m11_n0_n10_n11_grid_desc = *reinterpret_cast<const CM0M10M11N0N10N11GridDesc *>((const void*)p_c_m0_m10_m11_n0_n10_n11_grid_desc); 
    const auto c_blockid_to_m0_n0_block_cluster_adaptor = *reinterpret_cast<const CBlockIdToM0N0BlockClusterAdaptor *>((const void*)p_c_blockid_to_m0_n0_block_cluster_adaptor); 

    const auto kernel = kernel_dynamic_gemm_v1r2<GridwiseGemm, FloatAB, FloatC,
                                                 remove_reference_t<AKM0M1GridDesc>,
                                                 remove_reference_t<BKN0N1GridDesc>,
                                                 remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                                 remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                                 hasMainKBlockLoop, hasDoubleTailKBlockLoop>;

    kernel(static_cast<const FloatAB*>(p_a_grid), static_cast<const FloatAB*>(p_b_grid), static_cast<FloatC*>(p_c_grid),
            a_k_m0_m1_grid_desc, b_k_n0_n1_grid_desc, c_m0_m10_m11_n0_n10_n11_grid_desc, c_blockid_to_m0_n0_block_cluster_adaptor);
};

