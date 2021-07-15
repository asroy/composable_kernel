#ifndef CK_BLOCKWISE_GEMM_V2R3_HPP
#define CK_BLOCKWISE_GEMM_V2R3_HPP

#include "common_header.hpp"
#include "tensor_adaptor.hpp"
#include "threadwise_dynamic_tensor_slice_transfer_v2.hpp"
#include "threadwise_contraction.hpp"

namespace ck {

// C[M0, M1, N0, N1] += transpose(A[K, M0, M1]) * B[K, N0, N1]
// A and B are visable to the whole block, C is distributed among each thread
// Assume:
//   1. A:
//     1. ABlockDesc_BK0_BM_BK1 is known at compile-time
//     2. ABlockBuffer is DynamicBuffer
//   2. B:
//     1. BBlockDesc_BK0_BN_BK1 is known at compile-time
//     2. BBlockBuffer is DynamicBuffer
//   3. C:
//     1. CThreadDesc_BM0_BM11_BN0_BN11 is known at compile-time
//     2. CThreadBuffer is StaticBuffer
// Also assume:
//   M0 = N0 = 2. It will do 2x2 pipelined read and fma (ABBA optimization)
template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename ABlockDesc_BK0_BM_BK1,
          typename BBlockDesc_BK0_BN_BK1,
          index_t BM1PerThreadBM11,
          index_t BN1PerThreadBN11,
          index_t BK0PerThread,
          index_t BM10BN10ThreadClusterBM100,
          index_t BM10BN10ThreadClusterBN100,
          index_t BM10BN10ThreadClusterBM101,
          index_t BM10BN10ThreadClusterBN101,
          index_t AThreadCopyScalarPerVector_BM11,
          index_t BThreadCopyScalarPerVector_BN11,
          typename std::enable_if<ABlockDesc_BK0_BM_BK1::IsKnownAtCompileTime() &&
                                      BBlockDesc_BK0_BN_BK1::IsKnownAtCompileTime(),
                                  bool>::type = false>
struct BlockwiseGemm_k0mk1_k0nk1_m0m1n0n1_v2r3_pipeline_2x2
{
    using AIndex = MultiIndex<3>;
    using BIndex = MultiIndex<3>;
    using CIndex = MultiIndex<4>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr index_t K0 = ABlockDesc_BK0_BM_BK1{}.GetLength(I0);
    static constexpr index_t K1 = ABlockDesc_BK0_BM_BK1{}.GetLength(I2);
    static constexpr index_t M  = ABlockDesc_BK0_BM_BK1{}.GetLength(I1);
    static constexpr index_t N  = BBlockDesc_BK0_BN_BK1{}.GetLength(I1);

    static constexpr index_t M100 = BM10BN10ThreadClusterBM100;
    static constexpr index_t N100 = BM10BN10ThreadClusterBN100;

    static constexpr index_t M101 = BM10BN10ThreadClusterBM101;
    static constexpr index_t N101 = BM10BN10ThreadClusterBN101;

    static constexpr index_t M11 = BM1PerThreadBM11;
    static constexpr index_t N11 = BN1PerThreadBN11;

    static constexpr index_t M1 =
        BM10BN10ThreadClusterBM100 * BM10BN10ThreadClusterBM101 * BM1PerThreadBM11;
    static constexpr index_t N1 =
        BM10BN10ThreadClusterBN100 * BM10BN10ThreadClusterBN101 * BN1PerThreadBN11;

    static constexpr index_t M0 = M / M1;
    static constexpr index_t N0 = N / N1;

    __host__ __device__ static constexpr auto
    MakeABlockDescriptor_BK0_BM0_BM1_BK1(const ABlockDesc_BK0_BM_BK1& a_k0_m_k1_block_desc)
    {
        const auto a_k0_m0_m1_k1_block_desc = transform_dynamic_tensor_descriptor(
            a_k0_m_k1_block_desc,
            make_tuple(make_pass_through_transform(Number<K0>{}),
                       make_unmerge_transform(make_tuple(Number<M0>{}, Number<M1>{})),
                       make_pass_through_transform(Number<K1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        return a_k0_m0_m1_k1_block_desc;
    }

    __host__ __device__ static constexpr auto
    MakeBBlockDescriptor_BK0_BN0_BN1_BK1(const BBlockDesc_BK0_BN_BK1& b_k0_n_k1_block_desc)
    {
        const auto b_k0_n0_n1_k1_block_desc = transform_dynamic_tensor_descriptor(
            b_k0_n_k1_block_desc,
            make_tuple(make_pass_through_transform(Number<K0>{}),
                       make_unmerge_transform(make_tuple(Number<N0>{}, Number<N1>{})),
                       make_pass_through_transform(Number<K1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        return b_k0_n0_n1_k1_block_desc;
    }

    __host__ __device__ static constexpr auto
    MakeCBlockAdaptor_BM0_BM100_BM101_BM11_BN0_BN100_BN101_BN11_To_BM_BN()
    {
        // upper: [M0, M100, M101, M11, N0, N100, N101, N11]
        // lower: [M, N]
        constexpr auto c_m0_m100_m101_m11_n0_n100_n101_n11_to_m_n_block_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(
                               Number<M0>{}, Number<M100>{}, Number<M101>{}, Number<M11>{})),
                           make_unmerge_transform(make_tuple(
                               Number<N0>{}, Number<N100>{}, Number<N101>{}, Number<N11>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4, 5, 6, 7>{}));

        return c_m0_m100_m101_m11_n0_n100_n101_n11_to_m_n_block_adaptor;
    }

    __host__ __device__ static constexpr auto
    MakeCBlockAdaptor_BM0_BM100_BM101_BM11_BN0_BN100_BN101_BN11_To_BM0_BM1_BN0_BN1()
    {
        // upper: [M0, M100, M101, M11, N0, N100, N101, N11]
        // lower: [M0, M1, N0, N1]
        constexpr auto c_m0_m100_m101_m11_n0_n100_n101_n11_to_m0_m1_n0_n1_block_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_pass_through_transform(Number<M0>{}),
                           make_unmerge_transform(
                               make_tuple(Number<M100>{}, Number<M101>{}, Number<M11>{})),
                           make_pass_through_transform(Number<N0>{}),
                           make_unmerge_transform(
                               make_tuple(Number<N100>{}, Number<N101>{}, Number<N11>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}, Sequence<5, 6, 7>{}));

        return c_m0_m100_m101_m11_n0_n100_n101_n11_to_m0_m1_n0_n1_block_adaptor;
    }

    __host__ __device__ static constexpr auto GetCThreadTensorLengths_BM0_BM11_BN0_BN11()
    {
        return Sequence<M0, M11, N0, N11>{};
    }

    static constexpr auto a_k0_m0_m1_k1_block_desc_ =
        MakeABlockDescriptor_BK0_BM0_BM1_BK1(ABlockDesc_BK0_BM_BK1{});
    static constexpr auto b_k0_n0_n1_k1_block_desc_ =
        MakeBBlockDescriptor_BK0_BN0_BN1_BK1(BBlockDesc_BK0_BN_BK1{});

    public:
    __device__ BlockwiseGemm_k0mk1_k0nk1_m0m1n0n1_v2r3_pipeline_2x2()
        : c_thread_origin_data_idx_{CalculateCM0M1N0N1ThreadOriginOnBlock(
              get_thread_local_1d_id())},
          a_thread_copy_{
              make_tuple(0, c_thread_origin_data_idx_[I0], c_thread_origin_data_idx_[I1], 0)},
          b_thread_copy_{
              make_tuple(0, c_thread_origin_data_idx_[I2], c_thread_origin_data_idx_[I3], 0)}
    {
        static_assert(ABlockDesc_BK0_BM_BK1::IsKnownAtCompileTime() &&
                          BBlockDesc_BK0_BN_BK1::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(BlockSize == M101 * M100 * N101 * N100,
                      "wrong! blocksize and cluster size not consistent");

        static_assert(M % M1 == 0 && N % N1 == 0, "wrong!");

        static_assert(ABlockDesc_BK0_BM_BK1{}.GetLength(I0) ==
                          BBlockDesc_BK0_BN_BK1{}.GetLength(I0),
                      "wrong! K dimension not consistent");

        // TODO: remove this restriction
        static_assert(M0 == 2 && N0 == 2, "wrong");
    }

    __device__ static CIndex CalculateCM0M1N0N1ThreadOriginOnBlock(index_t thread_id)
    {
        // lower: [M0, M1, N0, N1]
        // upper: [M0, M100, M101, M11, N0, N100, N101, N11]
        constexpr auto adaptor0 =
            MakeCBlockAdaptor_BM0_BM100_BM101_BM11_BN0_BN100_BN101_BN11_To_BM0_BM1_BN0_BN1();

        // lower: [M0, M100, M101, M11, N0, N100, N101, N11]
        // upper: [Tid, M0, M11, N0, N11]
        constexpr auto adaptor1 = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(M100, N100, M101, N101)),
                       make_pass_through_transform(M0),
                       make_pass_through_transform(M11),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(N11)),
            make_tuple(
                Sequence<1, 5, 2, 6>{}, Sequence<0>{}, Sequence<3>{}, Sequence<4>{}, Sequence<7>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        constexpr auto adaptor = chain_tensor_adaptors(adaptor0, adaptor1);

        return adaptor.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id(), 0, 0, 0, 0));
    }

    template <typename CThreadDesc_BM0_BM11_BN0_BN11,
              typename ABlockBuffer,
              typename BBlockBuffer,
              typename CThreadBuffer>
    __device__ void Run(const CThreadDesc_BM0_BM11_BN0_BN11& c_m0_m1_n0_n1_thread_desc,
                        const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        static_assert(CThreadDesc_BM0_BM11_BN0_BN11::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        // TODO: remove this restriction
        static_assert(M0 == 2 && N0 == 2 && CThreadDesc_BM0_BM11_BN0_BN11{}.GetLength(I0) == M0 &&
                          CThreadDesc_BM0_BM11_BN0_BN11{}.GetLength(I2) == N0,
                      "wrong");

        auto a_thread_buf = make_static_buffer<AddressSpace::Vgpr, FloatA>(
            a_k0_m0_m1_k1_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpace::Vgpr, FloatB>(
            b_k0_n0_n1_k1_thread_desc_.GetElementSpaceSize());

        constexpr auto threadwise_contraction =
            ThreadwiseContraction_A_TK0_TM0_TM1_TK1_B_TK0_TN0_TN1_TK1_C_TM0_TM1_TN0_TN1<
                FloatA,
                FloatB,
                FloatC,
                decltype(a_k0_m0_m1_k1_thread_desc_),
                decltype(b_k0_n0_n1_k1_thread_desc_),
                CThreadDesc_BM0_BM11_BN0_BN11,
                Sequence<BK0PerThread, K1>,
                Sequence<1, BM1PerThreadBM11>,
                Sequence<1, BN1PerThreadBN11>>{};

        // read A_sub_0
        a_thread_copy_.Run(a_k0_m0_m1_k1_block_desc_,
                           make_tuple(I0, I0, I0, I0),
                           a_block_buf,
                           a_k0_m0_m1_k1_thread_desc_,
                           make_tuple(I0, I0, I0, I0),
                           a_thread_buf);

        // read B_sub_0
        b_thread_copy_.Run(b_k0_n0_n1_k1_block_desc_,
                           make_tuple(I0, I0, I0, I0),
                           b_block_buf,
                           b_k0_n0_n1_k1_thread_desc_,
                           make_tuple(I0, I0, I0, I0),
                           b_thread_buf);

        // read B_sub_1
        b_thread_copy_.Run(b_k0_n0_n1_k1_block_desc_,
                           make_tuple(I0, I1, I0, I0),
                           b_block_buf,
                           b_k0_n0_n1_k1_thread_desc_,
                           make_tuple(I0, I1, I0, I0),
                           b_thread_buf);

        // read A_sub_1
        a_thread_copy_.Run(a_k0_m0_m1_k1_block_desc_,
                           make_tuple(I0, I1, I0, I0),
                           a_block_buf,
                           a_k0_m0_m1_k1_thread_desc_,
                           make_tuple(I0, I1, I0, I0),
                           a_thread_buf);

        // C_sub_00 += transpose(A_sub_0) * B_sub_0
        threadwise_contraction.Run(a_thread_buf,
                            make_tuple(I0, I0, I0, I0),
                            b_thread_buf,
                            make_tuple(I0, I0, I0, I0),
                            c_thread_buf,
                            make_tuple(I0, I0, I0, I0));

        // C_sub_01 += transpose(A_sub_0) * B_sub_1
        threadwise_contraction.Run(a_thread_buf,
                            make_tuple(I0, I0, I0, I0),
                            b_thread_buf,
                            make_tuple(I0, I1, I0, I0),
                            c_thread_buf,
                            make_tuple(I0, I0, I1, I0));

        // loop over rest of k
        static_for<BK0PerThread, K0, BK0PerThread>{}([&](auto k) {
            // read A_sub_0
            a_thread_copy_.Run(a_k0_m0_m1_k1_block_desc_,
                               make_tuple(k, I0, I0, I0),
                               a_block_buf,
                               a_k0_m0_m1_k1_thread_desc_,
                               make_tuple(I0, I0, I0, I0),
                               a_thread_buf);

            // C_sub_10 += transpose(A_sub_1) * B_sub_0
            threadwise_contraction.Run(a_thread_buf,
                                make_tuple(I0, I1, I0, I0),
                                b_thread_buf,
                                make_tuple(I0, I0, I0, I0),
                                c_thread_buf,
                                make_tuple(I1, I0, I0, I0));

            // read B_sub_0
            b_thread_copy_.Run(b_k0_n0_n1_k1_block_desc_,
                               make_tuple(k, I0, I0, I0),
                               b_block_buf,
                               b_k0_n0_n1_k1_thread_desc_,
                               make_tuple(I0, I0, I0, I0),
                               b_thread_buf);

            // C_sub_11 += transpose(A_sub_1) * B_sub_1
            threadwise_contraction.Run(a_thread_buf,
                                make_tuple(I0, I1, I0, I0),
                                b_thread_buf,
                                make_tuple(I0, I1, I0, I0),
                                c_thread_buf,
                                make_tuple(I1, I0, I1, I0));

            // read B_sub_1
            b_thread_copy_.Run(b_k0_n0_n1_k1_block_desc_,
                               make_tuple(k, I1, I0, I0),
                               b_block_buf,
                               b_k0_n0_n1_k1_thread_desc_,
                               make_tuple(I0, I1, I0, I0),
                               b_thread_buf);

            // read A_sub_1
            a_thread_copy_.Run(a_k0_m0_m1_k1_block_desc_,
                               make_tuple(k, I1, I0, I0),
                               a_block_buf,
                               a_k0_m0_m1_k1_thread_desc_,
                               make_tuple(I0, I1, I0, I0),
                               a_thread_buf);

            // C_sub_00 += transpose(A_sub_0) * B_sub_0
            threadwise_contraction.Run(a_thread_buf,
                                make_tuple(I0, I0, I0, I0),
                                b_thread_buf,
                                make_tuple(I0, I0, I0, I0),
                                c_thread_buf,
                                make_tuple(I0, I0, I0, I0));

            // C_sub_01 += transpose(A_sub_0) * B_sub_1
            threadwise_contraction.Run(a_thread_buf,
                                make_tuple(I0, I0, I0, I0),
                                b_thread_buf,
                                make_tuple(I0, I1, I0, I0),
                                c_thread_buf,
                                make_tuple(I0, I0, I1, I0));
        });

        // C_sub_10 += transpose(A_sub_1) * B_sub_0
        threadwise_contraction.Run(a_thread_buf,
                            make_tuple(I0, I1, I0, I0),
                            b_thread_buf,
                            make_tuple(I0, I0, I0, I0),
                            c_thread_buf,
                            make_tuple(I1, I0, I0, I0));

        // C_sub_11 += transpose(A_sub_1) * B_sub_1
        threadwise_contraction.Run(a_thread_buf,
                            make_tuple(I0, I1, I0, I0),
                            b_thread_buf,
                            make_tuple(I0, I1, I0, I0),
                            c_thread_buf,
                            make_tuple(I1, I0, I1, I0));
    }

    private:
    // A[K0, M0, M1, K1]
    static constexpr auto a_k0_m0_m1_k1_thread_desc_ =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
            Number<BK0PerThread>{}, Number<M0>{}, Number<BM1PerThreadBM11>{}, Number<K1>{}));

    // B[K0, N0, N1, K1]
    static constexpr auto b_k0_n0_n1_k1_thread_desc_ =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(
            Number<BK0PerThread>{}, Number<N0>{}, Number<BN1PerThreadBN11>{}, Number<K1>{}));

    using AThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4r1<
        FloatA,
        FloatA,
        decltype(a_k0_m0_m1_k1_block_desc_),
        decltype(a_k0_m0_m1_k1_thread_desc_),
        Sequence<BK0PerThread, 1, BM1PerThreadBM11, K1>, // SliceLengths
        Sequence<0, 1, 2, 3>,                            // DimAccessOrder
        Sequence<1, 1, BM1PerThreadBM11, K1>,            // SrcVectorTensorLengths
        Sequence<0, 1, 2, 3>>;                           // SrcVectorTensorContiguousDimOrder

    using BThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4r1<
        FloatB,
        FloatB,
        decltype(b_k0_n0_n1_k1_block_desc_),
        decltype(b_k0_n0_n1_k1_thread_desc_),
        Sequence<BK0PerThread, 1, BN1PerThreadBN11, K1>, // SliceLengths
        Sequence<0, 1, 2, 3>,                            // DimAccessOrder
        Sequence<1, 1, BN1PerThreadBN11, K1>,            // SrcVectorTensorLengths
        Sequence<0, 1, 2, 3>>;                           // SrcVectorTensorContiguousDimOrder

    CIndex c_thread_origin_data_idx_;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
#endif
