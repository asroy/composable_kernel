#ifndef CK_DRIVER_DYNAMIC_GEMM_V1R2
#define CK_DRIVER_DYNAMIC_GEMM_V1R2

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm_v1r2.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          typename AKMGridDesc,
          typename BKNGridDesc,
          typename CMNGridDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t M1PerThread,
          index_t N1PerThread,
          index_t KPerThread,
          index_t M1N1ThreadClusterM10,
          index_t M1N1ThreadClusterN10,
          index_t M1N1ThreadClusterM11,
          index_t M1N1ThreadClusterN11,
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
          typename AGridIteratorHacks,
          typename BGridIteratorHacks,
          typename CGridIteratorHacks,
          typename AGridMoveSliceWindowIteratorHacks,
          typename BGridMoveSliceWindowIteratorHacks>
__host__ float launch_kernel_dynamic_gemm_v1r2(const FloatAB* p_a_grid,
                                               const FloatAB* p_b_grid,
                                               FloatC* p_c_grid,
                                               const AKMGridDesc& a_k_m_grid_desc,
                                               const BKNGridDesc& b_k_n_grid_desc,
                                               const CMNGridDesc& c_m_n_grid_desc,
                                               AGridIteratorHacks,
                                               BGridIteratorHacks,
                                               CGridIteratorHacks,
                                               AGridMoveSliceWindowIteratorHacks,
                                               BGridMoveSliceWindowIteratorHacks,
                                               index_t nrepeat)

{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto M = a_k_m_grid_desc.GetLength(I1);
    const auto N = b_k_n_grid_desc.GetLength(I1);
    const auto K = a_k_m_grid_desc.GetLength(I0);

    if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0))
    {
        throw std::runtime_error("wrong! GEMM size no divisible");
    }

    const auto M1 = Number<M1PerThread * M1N1ThreadClusterM11 * M1N1ThreadClusterM10>{};
    const auto N1 = Number<N1PerThread * M1N1ThreadClusterN11 * M1N1ThreadClusterN10>{};

    if(!(MPerBlock % M1 == 0 && NPerBlock % N1 == 0))
    {
        throw std::runtime_error("wrong! GEMM size no divisible");
    }

    const auto M0 = M / M1;
    const auto N0 = N / N1;

    const auto c_m0_m1_n0_n1_grid_desc =
        transform_dynamic_tensor_descriptor(c_m_n_grid_desc,
                                            make_tuple(make_unmerge_transform(make_tuple(M0, M1)),
                                                       make_unmerge_transform(make_tuple(N0, N1))),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

    using CM0M1N0N1GridDesc = decltype(c_m0_m1_n0_n1_grid_desc);

    // out_gemm_block_cluster_desc
    const auto c_block_cluster_desc =
        make_cluster_descriptor_v2(make_tuple(M / Number<MPerBlock>{}, N / Number<NPerBlock>{}));

    using CBlockClusterDesc = decltype(c_block_cluster_desc);

    // GEMM
    using gridwise_gemm =
        GridwiseDynamicGemm_km_kn_m0m1n0n1_v1r2<BlockSize,
                                                FloatAB,
                                                FloatAcc,
                                                FloatC,
                                                CGlobalMemoryDataOperation,
                                                AKMGridDesc,
                                                BKNGridDesc,
                                                CM0M1N0N1GridDesc,
                                                CBlockClusterDesc,
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
                                                AGridIteratorHacks,
                                                BGridIteratorHacks,
                                                CGridIteratorHacks,
                                                AGridMoveSliceWindowIteratorHacks,
                                                BGridMoveSliceWindowIteratorHacks>;

    const auto GridSize = (M / MPerBlock) * (N / NPerBlock);

    const bool has_main_k_block_loop = (K + KPerBlock) / (2 * KPerBlock) > 1;

    const bool has_double_tail_k_block_loop = (K / KPerBlock) % 2 == 0;

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
    float ave_time = 0;

    if(has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_gemm_v1r2<gridwise_gemm,
                                                     FloatAB,
                                                     FloatAB,
                                                     FloatC,
                                                     remove_reference_t<AKMGridDesc>,
                                                     remove_reference_t<BKNGridDesc>,
                                                     remove_reference_t<CM0M1N0N1GridDesc>,
                                                     remove_reference_t<CBlockClusterDesc>,
                                                     true,
                                                     true>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(GridSize),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m_grid_desc,
                                          b_k_n_grid_desc,
                                          c_m0_m1_n0_n1_grid_desc,
                                          c_block_cluster_desc);
    }
    else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_gemm_v1r2<gridwise_gemm,
                                                     FloatAB,
                                                     FloatAB,
                                                     FloatC,
                                                     remove_reference_t<AKMGridDesc>,
                                                     remove_reference_t<BKNGridDesc>,
                                                     remove_reference_t<CM0M1N0N1GridDesc>,
                                                     remove_reference_t<CBlockClusterDesc>,
                                                     true,
                                                     false>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(GridSize),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m_grid_desc,
                                          b_k_n_grid_desc,
                                          c_m0_m1_n0_n1_grid_desc,
                                          c_block_cluster_desc);
    }
    else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_gemm_v1r2<gridwise_gemm,
                                                     FloatAB,
                                                     FloatAB,
                                                     FloatC,
                                                     remove_reference_t<AKMGridDesc>,
                                                     remove_reference_t<BKNGridDesc>,
                                                     remove_reference_t<CM0M1N0N1GridDesc>,
                                                     remove_reference_t<CBlockClusterDesc>,
                                                     false,
                                                     true>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(GridSize),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m_grid_desc,
                                          b_k_n_grid_desc,
                                          c_m0_m1_n0_n1_grid_desc,
                                          c_block_cluster_desc);
    }
    else
    {
        const auto kernel = kernel_dynamic_gemm_v1r2<gridwise_gemm,
                                                     FloatAB,
                                                     FloatAB,
                                                     FloatC,
                                                     remove_reference_t<AKMGridDesc>,
                                                     remove_reference_t<BKNGridDesc>,
                                                     remove_reference_t<CM0M1N0N1GridDesc>,
                                                     remove_reference_t<CBlockClusterDesc>,
                                                     false,
                                                     false>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(GridSize),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m_grid_desc,
                                          b_k_n_grid_desc,
                                          c_m0_m1_n0_n1_grid_desc,
                                          c_block_cluster_desc);
    }

    return ave_time;
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
    DeviceMem a_k_m_grid_desc_device_buf(sizeof(AKMGridDesc));
    DeviceMem b_k_n_grid_desc_device_buf(sizeof(BKNGridDesc));
    DeviceMem c_m0_m1_n0_n1_grid_desc_device_buf(sizeof(CM0M1N0N1GridDesc));
    DeviceMem c_block_cluster_desc_device_buf(sizeof(c_block_cluster_desc));

    a_k_m_grid_desc_device_buf.ToDevice(&a_k_m_grid_desc);
    b_k_n_grid_desc_device_buf.ToDevice(&b_k_n_grid_desc);
    c_m0_m1_n0_n1_grid_desc_device_buf.ToDevice(&c_m0_m1_n0_n1_grid_desc);
    c_block_cluster_desc_device_buf.ToDevice(&c_block_cluster_desc);

    float ave_time = 0;

    if(has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_gemm_v1r2<gridwise_gemm,
                                                     FloatAB,
                                                     FloatAB,
                                                     FloatC,
                                                     remove_reference_t<AKMGridDesc>,
                                                     remove_reference_t<BKNGridDesc>,
                                                     remove_reference_t<CM0M1N0N1GridDesc>,
                                                     remove_reference_t<CBlockClusterDesc>,
                                                     true,
                                                     true>;

        ave_time = launch_and_time_kernel(
            kernel,
            nrepeat,
            dim3(GridSize),
            dim3(BlockSize),
            0,
            0,
            p_a_grid,
            p_b_grid,
            p_c_grid,
            (void __CONSTANT__*)a_k_m_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)b_k_n_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_m0_m1_n0_n1_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_block_cluster_desc_device_buf.GetDeviceBuffer());
    }
    else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_gemm_v1r2<gridwise_gemm,
                                                     FloatAB,
                                                     FloatAB,
                                                     FloatC,
                                                     remove_reference_t<AKMGridDesc>,
                                                     remove_reference_t<BKNGridDesc>,
                                                     remove_reference_t<CM0M1N0N1GridDesc>,
                                                     remove_reference_t<CBlockClusterDesc>,
                                                     true,
                                                     false>;

        ave_time = launch_and_time_kernel(
            kernel,
            nrepeat,
            dim3(GridSize),
            dim3(BlockSize),
            0,
            0,
            p_a_grid,
            p_b_grid,
            p_c_grid,
            (void __CONSTANT__*)a_k_m_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)b_k_n_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_m0_m1_n0_n1_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_block_cluster_desc_device_buf.GetDeviceBuffer());
    }
    else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_gemm_v1r2<gridwise_gemm,
                                                     FloatAB,
                                                     FloatAB,
                                                     FloatC,
                                                     remove_reference_t<AKMGridDesc>,
                                                     remove_reference_t<BKNGridDesc>,
                                                     remove_reference_t<CM0M1N0N1GridDesc>,
                                                     remove_reference_t<CBlockClusterDesc>,
                                                     false,
                                                     true>;

        ave_time = launch_and_time_kernel(
            kernel,
            nrepeat,
            dim3(GridSize),
            dim3(BlockSize),
            0,
            0,
            p_a_grid,
            p_b_grid,
            p_c_grid,
            (void __CONSTANT__*)a_k_m_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)b_k_n_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_m0_m1_n0_n1_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_block_cluster_desc_device_buf.GetDeviceBuffer());
    }
    else
    {
        const auto kernel = kernel_dynamic_gemm_v1r2<gridwise_gemm,
                                                     FloatAB,
                                                     FloatAB,
                                                     FloatC,
                                                     remove_reference_t<AKMGridDesc>,
                                                     remove_reference_t<BKNGridDesc>,
                                                     remove_reference_t<CM0M1N0N1GridDesc>,
                                                     remove_reference_t<CBlockClusterDesc>,
                                                     false,
                                                     false>;

        ave_time = launch_and_time_kernel(
            kernel,
            nrepeat,
            dim3(GridSize),
            dim3(BlockSize),
            0,
            0,
            p_a_grid,
            p_b_grid,
            p_c_grid,
            (void __CONSTANT__*)a_k_m_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)b_k_n_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_m0_m1_n0_n1_grid_desc_device_buf.GetDeviceBuffer(),
            (void __CONSTANT__*)c_block_cluster_desc_device_buf.GetDeviceBuffer());
    }

    return ave_time;
#endif
}

} // namespace ck
#endif
