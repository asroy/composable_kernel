#ifndef CK_DRIVER_DYNAMIC_GEMM_XDLOPS_V1
#define CK_DRIVER_DYNAMIC_GEMM_XDLOPS_V1

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm_xdlops.hpp"
#include "gridwise_operation_wrapper.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          typename AGlobalDesc,
          typename BGlobalDesc,
          typename CGlobalDesc,
          typename CBlockClusterDesc,
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
__host__ float launch_kernel_dynamic_gemm_xdlops_v1(const FloatAB* p_a_global,
                                                    const FloatAB* p_b_global,
                                                    FloatC* p_c_global,
                                                    const AGlobalDesc& a_k_m_global_desc,
                                                    const BGlobalDesc& b_k_n_global_desc,
                                                    const CGlobalDesc& c_m0_m1_n0_n1_global_desc,
                                                    const CBlockClusterDesc& c_block_cluster_desc,
                                                    AGlobalIteratorHacks,
                                                    BGlobalIteratorHacks,
                                                    CGlobalIteratorHacks,
                                                    AGlobalMoveSliceWindowIteratorHacks,
                                                    BGlobalMoveSliceWindowIteratorHacks,
                                                    index_t nrepeat)

{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto M = a_k_m_global_desc.GetLength(I1);
    const auto N = b_k_n_global_desc.GetLength(I1);
    const auto K = a_k_m_global_desc.GetLength(I0);

    if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0))
    {
        throw std::runtime_error("wrong! GEMM size no divisible");
    }

    constexpr auto M1 = Number<MPerThread * MLevel0Cluster * MLevel1Cluster>{};
    constexpr auto N1 = Number<NPerThread * NLevel0Cluster * NLevel1Cluster>{};

    if(!(MPerBlock % M1 == 0 && NPerBlock % N1 == 0))
    {
        throw std::runtime_error("wrong! GEMM size no divisible");
    }

    // GEMM
    using gridwise_gemm =
        GridwiseDynamicGemm_km_kn_m0m1n0n1_xdlops_v1<BlockSize,
                                                     FloatAB,
                                                     FloatAcc,
                                                     FloatC,
                                                     CGlobalMemoryDataOperation,
                                                     AGlobalDesc,
                                                     BGlobalDesc,
                                                     CGlobalDesc,
                                                     CBlockClusterDesc,
                                                     MPerBlock,
                                                     NPerBlock,
                                                     KPerBlock,
                                                     MPerThread,
                                                     NPerThread,
                                                     KPerThread,
                                                     MLevel0Cluster,
                                                     NLevel0Cluster,
                                                     MLevel1Cluster,
                                                     NLevel1Cluster,
                                                     ABlockTransferThreadSliceLengths_K_M,
                                                     ABlockTransferThreadClusterLengths_K_M,
                                                     ABlockTransferThreadClusterArrangeOrder,
                                                     ABlockTransferSrcAccessOrder,
                                                     ABlockTransferSrcVectorDim,
                                                     ABlockTransferSrcScalarPerVector,
                                                     ABlockTransferDstScalarPerVector_M,
                                                     AThreadTransferSrcResetCoordinateAfterRun,
                                                     BBlockTransferThreadSliceLengths_K_N,
                                                     BBlockTransferThreadClusterLengths_K_N,
                                                     BBlockTransferThreadClusterArrangeOrder,
                                                     BBlockTransferSrcAccessOrder,
                                                     BBlockTransferSrcVectorDim,
                                                     BBlockTransferSrcScalarPerVector,
                                                     BBlockTransferDstScalarPerVector_N,
                                                     BThreadTransferSrcResetCoordinateAfterRun,
                                                     CThreadTransferSrcDstAccessOrder,
                                                     CThreadTransferSrcDstVectorDim,
                                                     CThreadTransferDstScalarPerVector,
                                                     AGlobalIteratorHacks,
                                                     BGlobalIteratorHacks,
                                                     CGlobalIteratorHacks,
                                                     AGlobalMoveSliceWindowIteratorHacks,
                                                     BGlobalMoveSliceWindowIteratorHacks>;

    const auto GridSize = (M / MPerBlock) * (N / NPerBlock);

    const bool has_main_k_block_loop = (K + KPerBlock) / (2 * KPerBlock) > 1;

    const bool has_double_tail_k_block_loop = (K / KPerBlock) % 2 == 0;

    float ave_time = 0;

    if(has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                   remove_reference_t<AGlobalDesc>,
                                                   const FloatAB*,
                                                   remove_reference_t<BGlobalDesc>,
                                                   const FloatAB*,
                                                   remove_reference_t<CGlobalDesc>,
                                                   FloatC*,
                                                   remove_reference_t<CBlockClusterDesc>,
                                                   integral_constant<bool, true>,
                                                   integral_constant<bool, true>>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(GridSize),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          a_k_m_global_desc,
                                          p_a_global,
                                          b_k_n_global_desc,
                                          p_b_global,
                                          c_m0_m1_n0_n1_global_desc,
                                          p_c_global,
                                          c_block_cluster_desc,
                                          integral_constant<bool, true>{},
                                          integral_constant<bool, true>{});
    }
    else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
    {
        const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                   remove_reference_t<AGlobalDesc>,
                                                   const FloatAB*,
                                                   remove_reference_t<BGlobalDesc>,
                                                   const FloatAB*,
                                                   remove_reference_t<CGlobalDesc>,
                                                   FloatC*,
                                                   remove_reference_t<CBlockClusterDesc>,
                                                   integral_constant<bool, true>,
                                                   integral_constant<bool, false>>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(GridSize),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          a_k_m_global_desc,
                                          p_a_global,
                                          b_k_n_global_desc,
                                          p_b_global,
                                          c_m0_m1_n0_n1_global_desc,
                                          p_c_global,
                                          c_block_cluster_desc,
                                          integral_constant<bool, true>{},
                                          integral_constant<bool, false>{});
    }
    else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                   remove_reference_t<AGlobalDesc>,
                                                   const FloatAB*,
                                                   remove_reference_t<BGlobalDesc>,
                                                   const FloatAB*,
                                                   remove_reference_t<CGlobalDesc>,
                                                   FloatC*,
                                                   remove_reference_t<CBlockClusterDesc>,
                                                   integral_constant<bool, false>,
                                                   integral_constant<bool, true>>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(GridSize),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          a_k_m_global_desc,
                                          p_a_global,
                                          b_k_n_global_desc,
                                          p_b_global,
                                          c_m0_m1_n0_n1_global_desc,
                                          p_c_global,
                                          c_block_cluster_desc,
                                          integral_constant<bool, false>{},
                                          integral_constant<bool, true>{});
    }
    else
    {
        const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                   remove_reference_t<AGlobalDesc>,
                                                   const FloatAB*,
                                                   remove_reference_t<BGlobalDesc>,
                                                   const FloatAB*,
                                                   remove_reference_t<CGlobalDesc>,
                                                   FloatC*,
                                                   remove_reference_t<CBlockClusterDesc>,
                                                   integral_constant<bool, false>,
                                                   integral_constant<bool, false>>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(GridSize),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          a_k_m_global_desc,
                                          p_a_global,
                                          b_k_n_global_desc,
                                          p_b_global,
                                          c_m0_m1_n0_n1_global_desc,
                                          p_c_global,
                                          c_block_cluster_desc,
                                          integral_constant<bool, false>{},
                                          integral_constant<bool, false>{});
    }

    return ave_time;
}

} // namespace ck
#endif
