#ifndef CK_DRIVER_DYNAMIC_GEMM_XDLOPS_V2R3
#define CK_DRIVER_DYNAMIC_GEMM_XDLOPS_V2R3

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm_xdlops_v2r3.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPack,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadSliceLengths_K0_M_K1,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_KPack,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_K0_N_K1,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_KPack,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          bool CAccessOrderMRepeatNRepeat>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nhwc_kyxc_nhwk(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const void __CONSTANT__* p_a_k0_m_k1_grid_desc,
            const void __CONSTANT__* p_b_k0_n_k1_grid_desc,
            const void __CONSTANT__* p_c_m0_m1_m2_n_grid_desc,
            const void __CONSTANT__* p_c_blockid_to_m0_n0_block_cluster_adaptor)
{

#if 0

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_n_hi_wi_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(256, 28, 28, 256));
    constexpr auto wei_k_y_x_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(256, 3, 3, 256));
    constexpr auto out_n_ho_wo_k_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(256, 28, 28, 256));

    constexpr auto descs =
        transform_forward_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk_pad(in_n_hi_wi_c_desc,
                                                                          wei_k_y_x_c_desc,
                                                                          out_n_ho_wo_k_desc,
                                                                          make_tuple(I1, I1),
                                                                          make_tuple(I1, I1),
                                                                          make_tuple(I1, I1),
                                                                          make_tuple(I1, I1),
                                                                          Number<KPack>{});

    constexpr auto a_k0_m_k1_grid_desc_tmp = descs[I0];
    constexpr auto b_k0_n_k1_grid_desc_tmp = descs[I1];
    constexpr auto c_m_n_grid_desc         = descs[I2];

    using AK0MK1GridDesc = decltype(a_k0_m_k1_grid_desc_tmp);
    using BK0NK1GridDesc = decltype(b_k0_n_k1_grid_desc_tmp);
    using CMNGridDesc    = decltype(c_m_n_grid_desc);

    using BGridIteratorHacks = decltype(make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}),
        make_tuple(
            Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{})));

    using AGridIteratorHacks =
        decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{}),
                            make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},
                                       Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{})));

    using CGridIteratorHacks = decltype(make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 1, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 1, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 1, 0, 0>{}),
                                                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 2, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 2, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 0, 0, 0>{},
                                                              Sequence<0, 0, 2, 0, 0>{})));

    using AGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>;
    using BGridMoveSliceWindowIteratorHacks = Sequence<0, 0, 0, 0, 0>;

    using GridwiseGemm =
        GridwiseDynamicGemm_k0mk1_k0nk1_mn_xdlops_v2r3<BlockSize,
                                                       FloatAB,
                                                       FloatAcc,
                                                       FloatC,
                                                       InMemoryDataOperation::Set,
                                                       AK0MK1GridDesc,
                                                       BK0NK1GridDesc,
                                                       CMNGridDesc,
                                                       MPerBlock,
                                                       NPerBlock,
                                                       KPerBlock,
                                                       MPerWave,
                                                       NPerWave,
                                                       KPack,
                                                       MRepeat,
                                                       NRepeat,
                                                       ABlockTransferThreadSliceLengths_K0_M_K1,
                                                       ABlockTransferThreadClusterLengths_K0_M_K1,
                                                       ABlockTransferThreadClusterArrangeOrder,
                                                       ABlockTransferSrcAccessOrder,
                                                       ABlockTransferSrcVectorDim,
                                                       ABlockTransferSrcScalarPerVector,
                                                       ABlockTransferDstScalarPerVector_KPack,
                                                       AThreadTransferSrcResetCoordinateAfterRun,
                                                       BBlockTransferThreadSliceLengths_K0_N_K1,
                                                       BBlockTransferThreadClusterLengths_K0_N_K1,
                                                       BBlockTransferThreadClusterArrangeOrder,
                                                       BBlockTransferSrcAccessOrder,
                                                       BBlockTransferSrcVectorDim,
                                                       BBlockTransferSrcScalarPerVector,
                                                       BBlockTransferDstScalarPerVector_KPack,
                                                       BThreadTransferSrcResetCoordinateAfterRun,
                                                       CThreadTransferSrcDstAccessOrder,
                                                       CThreadTransferSrcDstVectorDim,
                                                       CThreadTransferDstScalarPerVector,
                                                       AGridIteratorHacks,
                                                       BGridIteratorHacks,
                                                       CGridIteratorHacks,
                                                       AGridMoveSliceWindowIteratorHacks,
                                                       BGridMoveSliceWindowIteratorHacks,
                                                       false>;

    constexpr auto c_m0_m1_m2_n_grid_desc_tmp =
        GridwiseGemm::MakeCM0M1M2NGridDescriptor(c_m_n_grid_desc);
    constexpr auto c_blockid_to_m0_n0_block_cluster_adaptor_tmp =
        GridwiseGemm::MakeCBlockClusterAdaptor(c_m_n_grid_desc);

    using CM0M1M2NGridDesc = decltype(c_m0_m1_m2_n_grid_desc_tmp);
    using CBlockIdToM0N0BlockClusterAdaptor =
        decltype(c_blockid_to_m0_n0_block_cluster_adaptor_tmp);

    const auto a_k0_m_k1_grid_desc =
        *reinterpret_cast<const AK0MK1GridDesc*>((const void*)p_a_k0_m_k1_grid_desc);
    const auto b_k0_n_k1_grid_desc =
        *reinterpret_cast<const BK0NK1GridDesc*>((const void*)p_b_k0_n_k1_grid_desc);
    const auto c_m0_m1_m2_n_grid_desc =
        *reinterpret_cast<const CM0M1M2NGridDesc*>((const void*)p_c_m0_m1_m2_n_grid_desc);
    const auto c_blockid_to_m0_n0_block_cluster_adaptor =
        *reinterpret_cast<const CBlockIdToM0N0BlockClusterAdaptor*>(
            (const void*)p_c_blockid_to_m0_n0_block_cluster_adaptor);

    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      p_shared_block,
                      a_k0_m_k1_grid_desc,
                      b_k0_n_k1_grid_desc,
                      c_m0_m1_m2_n_grid_desc,
                      c_blockid_to_m0_n0_block_cluster_adaptor);
#endif
}

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          typename AK0MK1GridDesc,
          typename BK0NK1GridDesc,
          typename CMNGridDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPack,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadSliceLengths_K0_M_K1,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_K0_N_K1,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGridIteratorHacks,
          typename BGridIteratorHacks,
          typename CGridIteratorHacks,
          typename AGridMoveSliceWindowIteratorHacks,
          typename BGridMoveSliceWindowIteratorHacks,
          bool CAccessOrderMRepeatNRepeat>
__host__ float driver_dynamic_gemm_xdlops_v2r3(const FloatAB* p_a_grid,
                                               const FloatAB* p_b_grid,
                                               FloatC* p_c_grid,
                                               const AK0MK1GridDesc& a_k0_m_k1_grid_desc,
                                               const BK0NK1GridDesc& b_k0_n_k1_grid_desc,
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
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};

    using GridwiseGemm =
        GridwiseDynamicGemm_k0mk1_k0nk1_mn_xdlops_v2r3<BlockSize,
                                                       FloatAB,
                                                       FloatAcc,
                                                       FloatC,
                                                       CGlobalMemoryDataOperation,
                                                       AK0MK1GridDesc,
                                                       BK0NK1GridDesc,
                                                       CMNGridDesc,
                                                       MPerBlock,
                                                       NPerBlock,
                                                       KPerBlock,
                                                       MPerWave,
                                                       NPerWave,
                                                       KPack,
                                                       MRepeat,
                                                       NRepeat,
                                                       ABlockTransferThreadSliceLengths_K0_M_K1,
                                                       ABlockTransferThreadClusterLengths_K0_M_K1,
                                                       ABlockTransferThreadClusterArrangeOrder,
                                                       ABlockTransferSrcAccessOrder,
                                                       ABlockTransferSrcVectorDim,
                                                       ABlockTransferSrcScalarPerVector,
                                                       ABlockTransferDstScalarPerVector_K1,
                                                       AThreadTransferSrcResetCoordinateAfterRun,
                                                       BBlockTransferThreadSliceLengths_K0_N_K1,
                                                       BBlockTransferThreadClusterLengths_K0_N_K1,
                                                       BBlockTransferThreadClusterArrangeOrder,
                                                       BBlockTransferSrcAccessOrder,
                                                       BBlockTransferSrcVectorDim,
                                                       BBlockTransferSrcScalarPerVector,
                                                       BBlockTransferDstScalarPerVector_K1,
                                                       BThreadTransferSrcResetCoordinateAfterRun,
                                                       CThreadTransferSrcDstAccessOrder,
                                                       CThreadTransferSrcDstVectorDim,
                                                       CThreadTransferDstScalarPerVector,
                                                       AGridIteratorHacks,
                                                       BGridIteratorHacks,
                                                       CGridIteratorHacks,
                                                       AGridMoveSliceWindowIteratorHacks,
                                                       BGridMoveSliceWindowIteratorHacks,
                                                       CAccessOrderMRepeatNRepeat>;

    {
        std::cout << "a_k0_m_k1_grid_desc{" << a_k0_m_k1_grid_desc.GetLength(I0) << ", "
                  << a_k0_m_k1_grid_desc.GetLength(I1) << ", " << a_k0_m_k1_grid_desc.GetLength(I2)
                  << "}" << std::endl;

        std::cout << "b_k0_n_k1_grid_desc{" << b_k0_n_k1_grid_desc.GetLength(I0) << ", "
                  << b_k0_n_k1_grid_desc.GetLength(I1) << ", " << b_k0_n_k1_grid_desc.GetLength(I2)
                  << "}" << std::endl;

        std::cout << "c_m_n_grid_desc{ " << c_m_n_grid_desc.GetLength(I0) << ", "
                  << c_m_n_grid_desc.GetLength(I1) << "}" << std::endl;
    }

    if(!GridwiseGemm::CheckValidity(a_k0_m_k1_grid_desc, b_k0_n_k1_grid_desc, c_m_n_grid_desc))
    {
        throw std::runtime_error(
            "wrong! GridwiseDynamicGemm_km_kn_m0m1n0n1_xdlops_v2r3 has invalid setting");
    }

    const auto c_m0_m1_m2_n_grid_desc = GridwiseGemm::MakeCM0M1M2NGridDescriptor(c_m_n_grid_desc);

    using CM0M1M2NGridDesc = decltype(c_m0_m1_m2_n_grid_desc);

    const auto c_block_cluster_adaptor = GridwiseGemm::MakeCBlockClusterAdaptor(c_m_n_grid_desc);

    using CBlockClusterAdaptor = decltype(c_block_cluster_adaptor);

    const index_t grid_size = GridwiseGemm::CalculateGridSize(c_m_n_grid_desc);

#if 0
    const auto kernel = kernel_dynamic_gemm_xdlops_v2r3<GridwiseGemm,
                                                        FloatAB,
                                                        FloatC,
                                                        remove_reference_t<AK0MK1GridDesc>,
                                                        remove_reference_t<BK0NK1GridDesc>,
                                                        remove_reference_t<CM0M1M2NGridDesc>,
                                                        remove_reference_t<CBlockClusterAdaptor>>;
#else
    const auto kernel = kernel_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nhwc_kyxc_nhwk<
        BlockSize,
        FloatAB,
        FloatAcc,
        FloatC,
        CGlobalMemoryDataOperation,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        MPerWave,
        NPerWave,
        KPack,
        MRepeat,
        NRepeat,
        ABlockTransferThreadSliceLengths_K0_M_K1,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        AThreadTransferSrcResetCoordinateAfterRun,
        BBlockTransferThreadSliceLengths_K0_N_K1,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        BThreadTransferSrcResetCoordinateAfterRun,
        CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector,
        AGridIteratorHacks,
        BGridIteratorHacks,
        CGridIteratorHacks,
        AGridMoveSliceWindowIteratorHacks,
        BGridMoveSliceWindowIteratorHacks,
        CAccessOrderMRepeatNRepeat>;
#endif

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
    float ave_time = launch_and_time_kernel(kernel,
                                            nrepeat,
                                            dim3(grid_size),
                                            dim3(BlockSize),
                                            0,
                                            0,
                                            p_a_grid,
                                            p_b_grid,
                                            p_c_grid,
                                            a_k0_m_k1_grid_desc,
                                            b_k0_n_k1_grid_desc,
                                            c_m0_m1_m2_n_grid_desc,
                                            c_block_cluster_adaptor);

#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER

    std::cerr << "CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER" << std::endl;

    DeviceMem a_k0_m_k1_grid_desc_dev_buf(sizeof(AK0MK1GridDesc));
    DeviceMem b_k0_n_k1_grid_desc_dev_buf(sizeof(BK0NK1GridDesc));
    DeviceMem c_m0_m1_m2_n_grid_desc_dev_buf(sizeof(CM0M1M2NGridDesc));
    DeviceMem c_block_cluster_adaptor_dev_buf(sizeof(CBlockClusterAdaptor));

    a_k0_m_k1_grid_desc_dev_buf.ToDevice(&a_k0_m_k1_grid_desc);
    b_k0_n_k1_grid_desc_dev_buf.ToDevice(&b_k0_n_k1_grid_desc);
    c_m0_m1_m2_n_grid_desc_dev_buf.ToDevice(&c_m0_m1_m2_n_grid_desc);
    c_block_cluster_adaptor_dev_buf.ToDevice(&c_block_cluster_adaptor);

    float ave_time = launch_and_time_kernel(
        kernel,
        nrepeat,
        dim3(grid_size),
        dim3(BlockSize),
        0,
        0,
        p_a_grid,
        p_b_grid,
        p_c_grid,
        (void __CONSTANT__*)a_k0_m_k1_grid_desc_dev_buf.GetDeviceBuffer(),
        (void __CONSTANT__*)b_k0_n_k1_grid_desc_dev_buf.GetDeviceBuffer(),
        (void __CONSTANT__*)c_m0_m1_m2_n_grid_desc_dev_buf.GetDeviceBuffer(),
        (void __CONSTANT__*)c_block_cluster_adaptor_dev_buf.GetDeviceBuffer());
#endif
    return ave_time;
}

} // namespace ck
#endif
