#ifndef CK_GRIDWISE_BATCHED_GEMM_XDLOPS_V2R3_HPP
#define CK_GRIDWISE_BATCHED_GEMM_XDLOPS_V2R3_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_gemm_xdlops.hpp"
#include "blockwise_tensor_slice_transfer_v4r1.hpp"
#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
template <typename GridwiseBatchedGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_G_K0_M_K1,
          typename BGridDesc_G_K0_N_K1,
          typename CGridDesc_G_M0_N0_M1_N1_M2_M3_M4_N2,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_xdlops_v2r3(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const AGridDesc_G_K0_M_K1 a_grid_desc_g_k0_m_k1,
            const BGridDesc_G_K0_N_K1 b_grid_desc_g_k0_n_k1,
            const CGridDesc_G_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op,
            const Block2CTileMap block_2_ctile_map)
{
    __shared__ char p_shared[GridwiseBatchedGemm::GetSharedMemoryNumberOfByte()];

    GridwiseBatchedGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                         p_b_grid,
                                                         p_c_grid,
                                                         p_shared,
                                                         a_grid_desc_g_k0_m_k1,
                                                         b_grid_desc_g_k0_n_k1,
                                                         c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2,
                                                         a_element_op,
                                                         b_element_op,
                                                         c_element_op,
                                                         block_2_ctile_map);
}
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
template <typename GridwiseBatchedGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_G_K0_M_K1,
          typename BGridDesc_G_K0_N_K1,
          typename CGridDesc_G_M0_N0_M1_N1_M2_M3_M4_N2,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_xdlops_v2r3(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const void CONSTANT* p_a_grid_desc_g_k0_m_k1,
            const void CONSTANT* p_b_grid_desc_g_k0_n_k1,
            const void CONSTANT* p_c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2,
            const void CONSTANT* p_a_element_op,
            const void CONSTANT* p_b_element_op,
            const void CONSTANT* p_c_element_op,
            const void CONSTANT* p_block_2_ctile_map)
{
    const auto a_grid_desc_g_k0_m_k1 = *reinterpret_cast<const AGridDesc_G_K0_M_K1*>(
        cast_pointer_to_generic_address_space(p_a_grid_desc_g_k0_m_k1));
    const auto b_grid_desc_g_k0_n_k1 = *reinterpret_cast<const BGridDesc_G_K0_N_K1*>(
        cast_pointer_to_generic_address_space(p_b_grid_desc_g_k0_n_k1));
    const auto c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2 =
        *reinterpret_cast<const CGridDesc_G_M0_N0_M1_N1_M2_M3_M4_N2*>(
            cast_pointer_to_generic_address_space(p_c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2));
    const auto block_2_ctile_map = *reinterpret_cast<const Block2CTileMap*>(
        cast_pointer_to_generic_address_space(p_block_2_ctile_map));
    const auto a_element_op = *reinterpret_cast<const AElementwiseOperation*>(
        cast_pointer_to_generic_address_space(p_a_element_op));
    const auto b_element_op = *reinterpret_cast<const BElementwiseOperation*>(
        cast_pointer_to_generic_address_space(p_b_element_op));
    const auto c_element_op = *reinterpret_cast<const CElementwiseOperation*>(
        cast_pointer_to_generic_address_space(p_c_element_op));

    __shared__ char p_shared[GridwiseBatchedGemm::GetSharedMemoryNumberOfByte()];

    GridwiseBatchedGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                         p_b_grid,
                                                         p_c_grid,
                                                         p_shared,
                                                         a_grid_desc_g_k0_m_k1,
                                                         b_grid_desc_g_k0_n_k1,
                                                         c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2,
                                                         a_element_op,
                                                         b_element_op,
                                                         c_element_op,
                                                         block_2_ctile_map);
}
#endif

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum_t CGlobalMemoryDataOperation,
          typename AGridDesc_G_K0_M_K1,
          typename BGridDesc_G_K0_N_K1,
          typename CGridDesc_G_M_N,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t K1Value,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_G_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_G_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          bool BBlockLdsExtraN,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector>
struct GridwiseBatchedGemm_gk0mk1_gk0nk1_gmn_xdlops_v2r3
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};
    static constexpr auto I8 = Number<8>{};

    // K1 should be Number<...>
    static constexpr auto K1 = Number<K1Value>{};

    __host__ __device__ static constexpr auto
    GetABlockDescriptor_BatchCount_K0PerBlock_MPerBlock_K1()
    {
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_g_k0_m_k1 = [&]() {
            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(I1, Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(Number<K0PerBlock>{} * Number<MPerBlock + 1>{} * K1,
                               Number<MPerBlock + 1>{} * K1,
                               K1,
                               I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(I1, Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);
            }
        }();

        return a_block_desc_g_k0_m_k1;
    }

    __host__ __device__ static constexpr auto
    GetBBlockDescriptor_BatchCount_K0PerBlock_NPerBlock_K1()
    {
        constexpr auto max_lds_align = K1;

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_g_k0_n_k1 = [&]() {
            if constexpr(BBlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(I1, Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(Number<K0PerBlock>{} * Number<NPerBlock + 1>{} * K1,
                               Number<NPerBlock + 1>{} * K1,
                               K1,
                               I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(I1, Number<K0PerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);
            }
        }();

        return b_block_desc_g_k0_n_k1;
    }

    __host__ __device__ static constexpr auto GetABlockDescriptor_K0PerBlock_MPerBlock_K1()
    {
        constexpr auto a_block_desc_g_k0_m_k1 =
            GetABlockDescriptor_BatchCount_K0PerBlock_MPerBlock_K1();

        constexpr auto K0 = a_block_desc_g_k0_m_k1.GetLength(I1);
        constexpr auto M  = a_block_desc_g_k0_m_k1.GetLength(I2);

        constexpr auto a_block_desc_k0_m_k1 = transform_tensor_descriptor(
            a_block_desc_g_k0_m_k1,
            make_tuple(make_freeze_transform(I0),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(M),
                       make_pass_through_transform(K1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        return a_block_desc_k0_m_k1;
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_K0PerBlock_NPerBlock_K1()
    {
        constexpr auto b_block_desc_g_k0_n_k1 =
            GetBBlockDescriptor_BatchCount_K0PerBlock_NPerBlock_K1();

        constexpr auto K0 = b_block_desc_g_k0_n_k1.GetLength(I1);
        constexpr auto N  = b_block_desc_g_k0_n_k1.GetLength(I2);

        constexpr auto b_block_desc_k0_n_k1 = transform_tensor_descriptor(
            b_block_desc_g_k0_n_k1,
            make_tuple(make_freeze_transform(I0),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(N),
                       make_pass_through_transform(K1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        return b_block_desc_k0_n_k1;
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_g_k0_m_k1 =
            GetABlockDescriptor_BatchCount_K0PerBlock_MPerBlock_K1();

        constexpr auto b_block_desc_g_k0_n_k1 =
            GetBBlockDescriptor_BatchCount_K0PerBlock_NPerBlock_K1();

        constexpr auto max_lds_align = K1;

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_g_k0_m_k1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_g_k0_n_k1.GetElementSpaceSize(), max_lds_align);

        return (a_block_space_size_aligned + b_block_space_size_aligned) * sizeof(FloatAB);
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_G_K0_M_K1& a_grid_desc_g_k0_m_k1,
                  const BGridDesc_G_K0_N_K1& b_grid_desc_g_k0_n_k1,
                  const CGridDesc_G_M_N& c_grid_desc_g_m_n,
                  index_t M01,
                  index_t N01)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerXDL * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXDL)) == 0,
                      "Invalid tuning param!");

        // const auto G  = a_grid_desc_g_k0_m_k1.GetLength(I0);
        const auto K0 = a_grid_desc_g_k0_m_k1.GetLength(I1);
        const auto M  = a_grid_desc_g_k0_m_k1.GetLength(I2);
        const auto N  = b_grid_desc_g_k0_n_k1.GetLength(I2);

        if(!(M == c_grid_desc_g_m_n.GetLength(I1) && N == c_grid_desc_g_m_n.GetLength(I2) &&
             K0 == b_grid_desc_g_k0_n_k1.GetLength(I1) &&
             K1 == a_grid_desc_g_k0_m_k1.GetLength(I3) &&
             K1 == b_grid_desc_g_k0_n_k1.GetLength(I3)))
            return false;

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K0 % K0PerBlock == 0))
            return false;

        // check M01, N01
        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        if(!(M0 % M01 == 0 && N0 % N01 == 0))
            return false;

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr index_t
    CalculateGridSize(const CGridDesc_G_M_N& c_grid_desc_g_m_n)
    {
        const auto G = c_grid_desc_g_m_n.GetLength(I0);
        const auto M = c_grid_desc_g_m_n.GetLength(I1);
        const auto N = c_grid_desc_g_m_n.GetLength(I2);

        const index_t grid_size = G * (M / MPerBlock) * (N / NPerBlock);

        return grid_size;
    }

    __host__ __device__ static constexpr bool CalculateHasMainK0BlockLoop(index_t K0)
    {
        const bool has_main_k0_block_loop = (K0 / K0PerBlock) > 1;

        return has_main_k0_block_loop;
    }

    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_G_M_N& c_grid_desc_g_m_n)
    {
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_k0_m_k1 = [&]() {
            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(Number<MPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);
            }
        }();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_k0_n_k1 = [&]() {
            if constexpr(BBlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(Number<NPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);
            }
        }();

        using BlockwiseGemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                FloatAB,
                                                                FloatAcc,
                                                                decltype(a_block_desc_k0_m_k1),
                                                                decltype(b_block_desc_k0_n_k1),
                                                                MPerXDL,
                                                                NPerXDL,
                                                                MXdlPerWave,
                                                                NXdlPerWave,
                                                                K1>;

        return BlockwiseGemm::MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_g_m_n);
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const CGridDesc_G_M_N& c_grid_desc_g_m_n, index_t M01, index_t N01)
    {
        const auto G = c_grid_desc_g_m_n.GetLength(I0);
        const auto M = c_grid_desc_g_m_n.GetLength(I1);
        const auto N = c_grid_desc_g_m_n.GetLength(I2);

        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        const auto M00 = M0 / M01;
        const auto N00 = N0 / N01;

        const auto g_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_pass_through_transform(G),
                           make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}));

        const auto cblockid_to_g_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(G, M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto cblockid_to_g_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(g_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  cblockid_to_g_m00_m01_n00_n01_block_cluster_adaptor);

        return cblockid_to_g_m0_n0_block_cluster_adaptor;
    }

    using CGridDesc_G_M0_N0_M1_N1_M2_M3_M4_N2 =
        decltype(MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(CGridDesc_G_M_N{}));
    using DefaultBlock2CTileMap = decltype(MakeDefaultBlock2CTileMap(CGridDesc_G_M_N{}, 1, 1));

    template <bool HasMainKBlockLoop, typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        void* __restrict__ p_shared,
        const AGridDesc_G_K0_M_K1& a_grid_desc_g_k0_m_k1,
        const BGridDesc_G_K0_N_K1& b_grid_desc_g_k0_n_k1,
        const CGridDesc_G_M0_N0_M1_N1_M2_M3_M4_N2& c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CElementwiseOperation& c_element_op,
        const Block2CTileMap& block_2_ctile_map)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_a_grid, a_grid_desc_g_k0_m_k1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_b_grid, b_grid_desc_g_k0_n_k1.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_c_grid, c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2.GetElementSpaceSize());

        const auto K0 = a_grid_desc_g_k0_m_k1.GetLength(I1);

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t g_idx_on_grid = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);

        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I2] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_g_k0_m_k1 =
            GetABlockDescriptor_BatchCount_K0PerBlock_MPerBlock_K1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_g_k0_n_k1 =
            GetBBlockDescriptor_BatchCount_K0PerBlock_NPerBlock_K1();

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4r1<BlockSize,
                                              AElementwiseOperation,
                                              ck::tensor_operation::element_wise::PassThrough,
                                              InMemoryDataOperationEnum_t::Set,
                                              Sequence<1, K0PerBlock, MPerBlock, K1>,
                                              ABlockTransferThreadClusterLengths_G_K0_M_K1,
                                              ABlockTransferThreadClusterArrangeOrder,
                                              FloatAB,
                                              FloatAB,
                                              decltype(a_grid_desc_g_k0_m_k1),
                                              decltype(a_block_desc_g_k0_m_k1),
                                              ABlockTransferSrcAccessOrder,
                                              Sequence<0, 2, 1, 3>,
                                              ABlockTransferSrcVectorDim,
                                              3,
                                              ABlockTransferSrcScalarPerVector,
                                              ABlockTransferDstScalarPerVector_K1,
                                              1,
                                              1,
                                              AThreadTransferSrcResetCoordinateAfterRun,
                                              true>(
                a_grid_desc_g_k0_m_k1,
                make_multi_index(g_idx_on_grid, 0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_g_k0_m_k1,
                make_multi_index(0, 0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4r1<BlockSize,
                                              BElementwiseOperation,
                                              ck::tensor_operation::element_wise::PassThrough,
                                              InMemoryDataOperationEnum_t::Set,
                                              Sequence<1, K0PerBlock, NPerBlock, K1>,
                                              BBlockTransferThreadClusterLengths_G_K0_N_K1,
                                              BBlockTransferThreadClusterArrangeOrder,
                                              FloatAB,
                                              FloatAB,
                                              decltype(b_grid_desc_g_k0_n_k1),
                                              decltype(b_block_desc_g_k0_n_k1),
                                              BBlockTransferSrcAccessOrder,
                                              Sequence<0, 2, 1, 3>,
                                              BBlockTransferSrcVectorDim,
                                              3,
                                              BBlockTransferSrcScalarPerVector,
                                              BBlockTransferDstScalarPerVector_K1,
                                              1,
                                              1,
                                              BThreadTransferSrcResetCoordinateAfterRun,
                                              true>(
                b_grid_desc_g_k0_n_k1,
                make_multi_index(g_idx_on_grid, 0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_g_k0_n_k1,
                make_multi_index(0, 0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check

        constexpr auto a_block_desc_k0_m_k1 = GetABlockDescriptor_K0PerBlock_MPerBlock_K1();
        constexpr auto b_block_desc_k0_n_k1 = GetBBlockDescriptor_K0PerBlock_NPerBlock_K1();

        auto blockwise_gemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                FloatAB,
                                                                FloatAcc,
                                                                decltype(a_block_desc_k0_m_k1),
                                                                decltype(b_block_desc_k0_n_k1),
                                                                MPerXDL,
                                                                NPerXDL,
                                                                MXdlPerWave,
                                                                NXdlPerWave,
                                                                K1>{};

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_g_k0_m_k1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            static_cast<FloatAB*>(p_shared), a_block_desc_g_k0_m_k1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            static_cast<FloatAB*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_g_k0_n_k1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(0, K0PerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(0, K0PerBlock, 0, 0);

        // preload data into LDS
        {
            a_blockwise_copy.RunRead(a_grid_desc_g_k0_m_k1, a_grid_buf);
            b_blockwise_copy.RunRead(b_grid_desc_g_k0_n_k1, b_grid_buf);

            a_blockwise_copy.RunWrite(a_block_desc_g_k0_m_k1, a_block_buf);
            b_blockwise_copy.RunWrite(b_block_desc_g_k0_n_k1, b_block_buf);
        }

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainKBlockLoop)
        {
            index_t k0_block_data_begin = 0;

            do
            {
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_g_k0_m_k1, a_block_slice_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_g_k0_n_k1, b_block_slice_copy_step);

                a_blockwise_copy.RunRead(a_grid_desc_g_k0_m_k1, a_grid_buf);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc_g_k0_n_k1, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.RunWrite(a_block_desc_g_k0_m_k1, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc_g_k0_n_k1, b_block_buf);

                k0_block_data_begin += K0PerBlock;
            } while(k0_block_data_begin < (K0 - K0PerBlock));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }

        // output: register to global memory
        {
            constexpr auto c_thread_desc_g_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm.GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto c_block_desc_g_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm.GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2();

            // constexpr auto G  = c_block_desc_g_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0);
            constexpr auto M0 = c_block_desc_g_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
            constexpr auto N0 = c_block_desc_g_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
            constexpr auto M1 = c_block_desc_g_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
            constexpr auto N1 = c_block_desc_g_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
            constexpr auto M2 = c_block_desc_g_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
            constexpr auto M3 = c_block_desc_g_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
            constexpr auto M4 = c_block_desc_g_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);
            constexpr auto N2 = c_block_desc_g_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I8);

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_grid =
                m_block_data_idx_on_grid + c_thread_mtx_on_block[I0];

            const index_t n_thread_data_on_grid =
                n_block_data_idx_on_grid + c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_grid_idx =
                m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_grid));

            const auto n_thread_data_on_grid_to_n0_n1_n2_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_grid_idx =
                n_thread_data_on_grid_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_grid));

            auto c_thread_copy = ThreadwiseTensorSliceTransfer_v1r3<
                FloatAcc,
                FloatC,
                decltype(c_thread_desc_g_m0_n0_m1_n1_m2_m3_m4_n2),
                decltype(c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2),
                CElementwiseOperation,
                Sequence<I1, M0, N0, I1, I1, M2, I1, M4, I1>,
                CThreadTransferSrcDstAccessOrder,
                CThreadTransferSrcDstVectorDim,
                CThreadTransferDstScalarPerVector,
                CGlobalMemoryDataOperation,
                1,
                true>{c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2,
                      make_multi_index(g_idx_on_grid,
                                       m_thread_data_on_grid_idx[I0],
                                       n_thread_data_on_grid_idx[I0],
                                       m_thread_data_on_grid_idx[I1],
                                       n_thread_data_on_grid_idx[I1],
                                       m_thread_data_on_grid_idx[I2],
                                       m_thread_data_on_grid_idx[I3],
                                       m_thread_data_on_grid_idx[I4],
                                       n_thread_data_on_grid_idx[I2]),
                      c_element_op};

            c_thread_copy.Run(c_thread_desc_g_m0_n0_m1_n1_m2_m3_m4_n2,
                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                              c_thread_buf,
                              c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2,
                              c_grid_buf);
        }
    }
};

} // namespace ck
#endif