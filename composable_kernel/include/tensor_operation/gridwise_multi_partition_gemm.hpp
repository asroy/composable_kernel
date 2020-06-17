#ifndef CK_GRIDWISE_MULTI_PARTITION_GEMM_HPP
#define CK_GRIDWISE_MULTI_PARTITION_GEMM_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_v2.hpp"

namespace ck {
template<index_t begin>
struct GemmBlockID
{
    /* data */
    __device__ const index_t get_gemm_block_id() const {
         return blockIdx.x - begin;
    }
};

template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccFloat,
          typename AGlobalDesc,
          typename BGlobalDesc,
          typename CGlobalDesc,
          InMemoryDataOperation CGlobalMemoryDataOperation,
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
          index_t ThreadGemmAThreadCopySrcDataPerRead_M,
          index_t ThreadGemmBThreadCopySrcDataPerRead_N,
          typename ABlockCopyThreadSliceLengths_K_M,
          typename ABlockCopyThreadClusterLengths_K_M,
          typename ABlockCopyThreadClusterArrangeOrder,
          typename ABlockCopySrcAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          index_t ABlockCopySrcDataPerRead,
          index_t ABlockCopyDstDataPerWrite_M,
          typename BBlockCopyThreadSliceLengths_K_N,
          typename BBlockCopyThreadClusterLengths_K_N,
          typename BBlockCopyThreadClusterArrangeOrder,
          typename BBlockCopySrcAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          index_t BBlockCopySrcDataPerRead,
          index_t BBlockCopyDstDataPerWrite_N,
          typename CThreadCopySrcDstAccessOrder,
          index_t CThreadCopySrcDstVectorReadWriteDim,
          index_t CThreadCopyDstDataPerWrite>
struct GridwiseMultiPartitionGemmTransposedANormalBNormalC_v1
{
    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr index_t max_lds_align = math::lcm(ABlockCopyDstDataPerWrite_M,
                                                    BBlockCopyDstDataPerWrite_N,
                                                    ThreadGemmAThreadCopySrcDataPerRead_M,
                                                    ThreadGemmBThreadCopySrcDataPerRead_N);

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, MPerBlock>{}, Number<max_lds_align>{});

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<KPerBlock, NPerBlock>{}, Number<max_lds_align>{});

        // LDS allocation for A and B: be careful of alignment
        constexpr index_t a_block_space =
            math::integer_least_multiple(a_k_m_block_desc.GetElementSpace(), max_lds_align);

        constexpr index_t b_block_space =
            math::integer_least_multiple(b_k_n_block_desc.GetElementSpace(), max_lds_align);

        return 2 * (a_block_space + b_block_space) * sizeof(Float);
    }

    __device__ void Run(const Float* __restrict__ p_wei_global,
                        const Float* __restrict__ p_in_global,
                        Float* __restrict__ p_out_global) const
    {
        constexpr index_t shared_block_size = GetSharedMemoryNumberOfByte() / sizeof(Float);
        __shared__ Float p_shared_block[shared_block_size];

        constexpr auto wei_e_k_global_desc = AGlobalDesc{};
        constexpr auto in_e_b_global_desc  = BGlobalDesc{};
        constexpr auto out_k_b_global_desc = CGlobalDesc{};

        constexpr auto GemmK = wei_e_k_global_desc.GetLengths()[0];
        constexpr auto GemmM = wei_e_k_global_desc.GetLengths()[1];
        constexpr auto GemmN = in_e_b_global_desc.GetLengths()[1];
        //just for simple, let GemmM and GemmN >=128
        static_assert(GemmM >= 128 && (GemmM % 128 == 0 || GemmM % 128 == 32),"GemmM >= 128 && (GemmM % 128 == 0 || GemmM % 128 == 32)");
        static_assert(GemmN >= 128 && (GemmN % 128 == 0 || GemmN % 128 == 32),"GemmN >= 128 && (GemmN % 128 == 0 || GemmN % 128 == 32) ");

        constexpr index_t GemmM128BlockNum = GemmM / 128;
        constexpr index_t GemmN128BlockNum = GemmN / 128;

        constexpr index_t GemmMBlockNum = (GemmM + 127) / 128;
        constexpr index_t GemmNBlockNum = (GemmN + 127) / 128;

        constexpr index_t GemmOBeginM = GemmM128BlockNum * 128;
        constexpr index_t GemmOBeginN = GemmN128BlockNum * 128;

        // partition W
        constexpr auto wei_e_k_global_1st_desc = transform_tensor_descriptor(
            wei_e_k_global_desc,
            make_tuple(Slice<Sequence<GemmM>, Sequence<0>, Sequence<GemmOBeginM>>{},
                       PassThrough<GemmK>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        constexpr auto wei_e_k_global_2nd_desc = transform_tensor_descriptor(
            wei_e_k_global_desc,
            make_tuple(Slice<Sequence<GemmM>, Sequence<GemmOBeginM>, Sequence<GemmM>>{},
                       PassThrough<GemmK>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));
        // partition in
        constexpr auto in_e_b_global_1st_desc = transform_tensor_descriptor(
            in_e_b_global_desc,
            make_tuple(PassThrough<GemmK>{},
                       Slice<Sequence<GemmN>, Sequence<0>, Sequence<GemmOBeginN>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
        constexpr auto in_e_b_global_2nd_desc = transform_tensor_descriptor(
            in_e_b_global_desc,
            make_tuple(PassThrough<GemmK>{},
                       Slice<Sequence<GemmN>, Sequence<GemmOBeginN>, Sequence<GemmN>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        constexpr auto block_work_desc =
            make_cluster_descriptor(Sequence<GemmMBlockNum, GemmNBlockNum>{});

// GEMM
        constexpr index_t Gemm128BlockNum = GemmM128BlockNum * GemmN128BlockNum;
        constexpr bool bIsHave1stPartition = Gemm128BlockNum > 0;
        constexpr bool bIsHave2ndPartition = GemmMBlockNum > GemmM128BlockNum && GemmN >= 128;
        constexpr bool bIsHave3rdPartition = GemmNBlockNum > GemmN128BlockNum && GemmM >= 128;
        constexpr bool bIsHave4thPartition = bIsHave2ndPartition && bIsHave3rdPartition;

        

        constexpr index_t bolck_begin_3rd =
            bIsHave2ndPartition ? Gemm128BlockNum + GemmN128BlockNum : Gemm128BlockNum;

        const auto getwaveid = []() { return get_thread_local_1d_id() / 64; };

        if(bIsHave1stPartition && get_block_1d_id() < Gemm128BlockNum)
        {
            static_if<bIsHave1stPartition>{}([&](auto){
                constexpr auto out_k_b_global_1st_desc = transform_tensor_descriptor(
                    out_k_b_global_desc,
                    make_tuple(Slice<Sequence<GemmM>, Sequence<0>, Sequence<GemmOBeginM>>{},
                               Slice<Sequence<GemmN>, Sequence<0>, Sequence<GemmOBeginN>>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                constexpr auto gridwise_gemm =
                    GridwiseGemmTransposedANormalBNormalC_v2<GridSize,
                                                             BlockSize,
                                                             Float,
                                                             AccFloat,
                                                             decltype(wei_e_k_global_1st_desc),
                                                             decltype(in_e_b_global_1st_desc),
                                                             decltype(out_k_b_global_1st_desc),
                                                             InMemoryDataOperation::Set,
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
                                                             ThreadGemmAThreadCopySrcDataPerRead_M,
                                                             ThreadGemmBThreadCopySrcDataPerRead_N,
                                                             ABlockCopyThreadSliceLengths_K_M,
                                                             ABlockCopyThreadClusterLengths_K_M,
                                                             ABlockCopyThreadClusterArrangeOrder,
                                                             ABlockCopySrcAccessOrder,
                                                             ABlockCopySrcVectorReadDim,
                                                             ABlockCopySrcDataPerRead,
                                                             ABlockCopyDstDataPerWrite_M,
                                                             BBlockCopyThreadSliceLengths_K_N,
                                                             BBlockCopyThreadClusterLengths_K_N,
                                                             BBlockCopyThreadClusterArrangeOrder,
                                                             BBlockCopySrcAccessOrder,
                                                             BBlockCopySrcVectorReadDim,
                                                             BBlockCopySrcDataPerRead,
                                                             BBlockCopyDstDataPerWrite_N,
                                                             CThreadCopySrcDstAccessOrder,
                                                             CThreadCopySrcDstVectorReadWriteDim,
                                                             CThreadCopyDstDataPerWrite,
                                                             GemmBlockID<0>>{};

                gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global, p_shared_block);
            });
        }
        else if(bIsHave2ndPartition && (get_block_1d_id() < Gemm128BlockNum + GemmN128BlockNum))
        {
            static_if<bIsHave2ndPartition>{}([&](auto){
                // BlockSize = 128, GemmKPerBlock = 8  32X128
                constexpr index_t BlockSize1 = 256;

                constexpr index_t GemmMPerBlock1 = 32;
                constexpr index_t GemmNPerBlock1 = 128;
                constexpr index_t GemmKPerBlock1 = 8;

                constexpr index_t GemmMPerThreadSubC1     = 4;
                constexpr index_t GemmNPerThreadSubC1     = 4;
                constexpr index_t GemmMLevel0Cluster1     = 4;
                constexpr index_t GemmNLevel0Cluster1     = 4;
                constexpr index_t GemmMLevel1Cluster1     = 2;
                constexpr index_t GemmNLevel1Cluster1     = 8;
                constexpr index_t GemmKPerThreadLoop1     = 1;
                constexpr index_t ThreadGemmDataPerReadM1 = 1;
                constexpr index_t ThreadGemmDataPerReadN1 = 1;

                using GemmABlockCopyThreadSliceLengths_GemmK_GemmM1   = Sequence<1, 1>;
                using GemmABlockCopyThreadClusterLengths_GemmK_GemmM1 = Sequence<8, 32>;

                constexpr index_t GemmABlockCopySrcDataPerRead_GemmK1  = 1;
                constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM1 = 1;

                using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN1   = Sequence<1, 4>;
                using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN1 = Sequence<8, 32>;

                constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN1  = 1;
                constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN1 = 1;

                constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN11 = 1;

                constexpr auto out_k_b_global_2nd_desc = transform_tensor_descriptor(
                    out_k_b_global_desc,
                    make_tuple(Slice<Sequence<GemmM>, Sequence<GemmOBeginM>, Sequence<GemmM>>{},
                               Slice<Sequence<GemmN>, Sequence<0>, Sequence<GemmOBeginN>>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
                constexpr auto gridwise_gemm1 = GridwiseGemmTransposedANormalBNormalC_v2<
                    GridSize,
                    BlockSize1,
                    Float,
                    AccFloat,
                    decltype(wei_e_k_global_2nd_desc),
                    decltype(in_e_b_global_1st_desc),
                    decltype(out_k_b_global_2nd_desc),
                    InMemoryDataOperation::Set,
                    GemmMPerBlock1,
                    GemmNPerBlock1,
                    GemmKPerBlock1,
                    GemmMPerThreadSubC1,
                    GemmNPerThreadSubC1,
                    GemmKPerThreadLoop1,
                    GemmMLevel0Cluster1,
                    GemmNLevel0Cluster1,
                    GemmMLevel1Cluster1,
                    GemmNLevel1Cluster1,
                    ThreadGemmDataPerReadM1,
                    ThreadGemmDataPerReadN1,
                    GemmABlockCopyThreadSliceLengths_GemmK_GemmM1,
                    GemmABlockCopyThreadClusterLengths_GemmK_GemmM1,
                    Sequence<1, 0>,
                    Sequence<1, 0>,
                    0,
                    GemmABlockCopySrcDataPerRead_GemmK1,
                    GemmABlockCopyDstDataPerWrite_GemmM1,
                    GemmBBlockCopyThreadSliceLengths_GemmK_GemmN1,
                    GemmBBlockCopyThreadClusterLengths_GemmK_GemmN1,
                    Sequence<0, 1>,
                    Sequence<0, 1>,
                    1,
                    GemmBBlockCopySrcDataPerRead_GemmN1,
                    GemmBBlockCopyDstDataPerWrite_GemmN1,
                    Sequence<0, 1, 2, 3>,
                    3,
                    GemmCThreadCopyDstDataPerWrite_GemmN11,
                    GemmBlockID<Gemm128BlockNum>>{};

                gridwise_gemm1.Run(p_wei_global, p_in_global, p_out_global, p_shared_block);
            });
        }
        else if(bIsHave3rdPartition && (get_block_1d_id() < bolck_begin_3rd + GemmM128BlockNum))
        {
            static_if<bIsHave3rdPartition>{}([&](auto){
                // BlockSize = 128, GemmKPerBlock = 8  128X32
                constexpr index_t BlockSize2 = 256;

                constexpr index_t GemmMPerBlock2 = 128;
                constexpr index_t GemmNPerBlock2 = 32;
                constexpr index_t GemmKPerBlock2 = 8;

                constexpr index_t GemmMPerThreadSubC2     = 4;
                constexpr index_t GemmNPerThreadSubC2     = 4;
                constexpr index_t GemmMLevel0Cluster2     = 4;
                constexpr index_t GemmNLevel0Cluster2     = 4;
                constexpr index_t GemmMLevel1Cluster2     = 8;
                constexpr index_t GemmNLevel1Cluster2     = 2;
                constexpr index_t GemmKPerThreadLoop2     = 1;
                constexpr index_t ThreadGemmDataPerReadM2 = 1;
                constexpr index_t ThreadGemmDataPerReadN2 = 1;

                using GemmABlockCopyThreadSliceLengths_GemmK_GemmM2   = Sequence<1, 4>;
                using GemmABlockCopyThreadClusterLengths_GemmK_GemmM2 = Sequence<8, 32>;

                constexpr index_t GemmABlockCopySrcDataPerRead_GemmK2  = 1;
                constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM2 = 1;

                using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN2   = Sequence<1, 1>;
                using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN2 = Sequence<8, 32>;

                constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN2  = 1;
                constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN2 = 1;

                constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN12 = 1;

                constexpr auto out_k_b_global_3rd_desc = transform_tensor_descriptor(
                    out_k_b_global_desc,
                    make_tuple(Slice<Sequence<GemmM>, Sequence<0>, Sequence<GemmOBeginM>>{},
                               Slice<Sequence<GemmN>, Sequence<GemmOBeginN>, Sequence<GemmN>>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                constexpr auto gridwise_gemm2 = GridwiseGemmTransposedANormalBNormalC_v2<
                    GridSize,
                    BlockSize2,
                    Float,
                    AccFloat,
                    decltype(wei_e_k_global_1st_desc),
                    decltype(in_e_b_global_2nd_desc),
                    decltype(out_k_b_global_3rd_desc),
                    InMemoryDataOperation::Set,
                    GemmMPerBlock2,
                    GemmNPerBlock2,
                    GemmKPerBlock2,
                    GemmMPerThreadSubC2,
                    GemmNPerThreadSubC2,
                    GemmKPerThreadLoop2,
                    GemmMLevel0Cluster2,
                    GemmNLevel0Cluster2,
                    GemmMLevel1Cluster2,
                    GemmNLevel1Cluster2,
                    ThreadGemmDataPerReadM2,
                    ThreadGemmDataPerReadN2,
                    GemmABlockCopyThreadSliceLengths_GemmK_GemmM2,
                    GemmABlockCopyThreadClusterLengths_GemmK_GemmM2,
                    Sequence<1, 0>,
                    Sequence<1, 0>,
                    0,
                    GemmABlockCopySrcDataPerRead_GemmK2,
                    GemmABlockCopyDstDataPerWrite_GemmM2,
                    GemmBBlockCopyThreadSliceLengths_GemmK_GemmN2,
                    GemmBBlockCopyThreadClusterLengths_GemmK_GemmN2,
                    Sequence<0, 1>,
                    Sequence<0, 1>,
                    1,
                    GemmBBlockCopySrcDataPerRead_GemmN2,
                    GemmBBlockCopyDstDataPerWrite_GemmN2,
                    Sequence<0, 1, 2, 3>,
                    3,
                    GemmCThreadCopyDstDataPerWrite_GemmN12,
                    GemmBlockID<bolck_begin_3rd>>{};

                gridwise_gemm2.Run(p_wei_global, p_in_global, p_out_global, p_shared_block);
            });
            
        }
        else if(bIsHave4thPartition)
        {
            const index_t waveid = getwaveid();
            if(waveid >= 1)
                return;

            static_if<bIsHave4thPartition>{}([&](auto){
                // BlockSize = 64, GemmKPerBlock = 8
                constexpr index_t BlockSize3 = 64;

                constexpr index_t GemmMPerBlock3 = 32;
                constexpr index_t GemmNPerBlock3 = 32;
                constexpr index_t GemmKPerBlock3 = 8;

                constexpr index_t GemmMPerThreadSubC3     = 4;
                constexpr index_t GemmNPerThreadSubC3     = 4;
                constexpr index_t GemmMLevel0Cluster3     = 4;
                constexpr index_t GemmNLevel0Cluster3     = 4;
                constexpr index_t GemmMLevel1Cluster3     = 2;
                constexpr index_t GemmNLevel1Cluster3     = 2;
                constexpr index_t GemmKPerThreadLoop3     = 1;
                constexpr index_t ThreadGemmDataPerReadM3 = 4;
                constexpr index_t ThreadGemmDataPerReadN3 = 4;

                using GemmABlockCopyThreadSliceLengths_GemmK_GemmM3   = Sequence<4, 1>;
                using GemmABlockCopyThreadClusterLengths_GemmK_GemmM3 = Sequence<2, 32>;

                constexpr index_t GemmABlockCopySrcDataPerRead_GemmK3  = 1;
                constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM3 = 1;

                using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN3   = Sequence<4, 1>;
                using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN3 = Sequence<2, 32>;

                constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN3  = 1;
                constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN3 = 1;

                constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN13 = 1;

                constexpr auto out_k_b_global_4th_desc = transform_tensor_descriptor(
                    out_k_b_global_desc,
                    make_tuple(Slice<Sequence<GemmM>, Sequence<GemmOBeginM>, Sequence<GemmM>>{},
                               Slice<Sequence<GemmN>, Sequence<GemmOBeginN>, Sequence<GemmN>>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                constexpr auto gridwise_gemm3 = GridwiseGemmTransposedANormalBNormalC_v2<
                    GridSize,
                    BlockSize3,
                    Float,
                    AccFloat,
                    decltype(wei_e_k_global_2nd_desc),
                    decltype(in_e_b_global_2nd_desc),
                    decltype(out_k_b_global_4th_desc),
                    InMemoryDataOperation::Set,
                    GemmMPerBlock3,
                    GemmNPerBlock3,
                    GemmKPerBlock3,
                    GemmMPerThreadSubC3,
                    GemmNPerThreadSubC3,
                    GemmKPerThreadLoop3,
                    GemmMLevel0Cluster3,
                    GemmNLevel0Cluster3,
                    GemmMLevel1Cluster3,
                    GemmNLevel1Cluster3,
                    ThreadGemmDataPerReadM3,
                    ThreadGemmDataPerReadN3,
                    GemmABlockCopyThreadSliceLengths_GemmK_GemmM3,
                    GemmABlockCopyThreadClusterLengths_GemmK_GemmM3,
                    Sequence<1, 0>,
                    Sequence<1, 0>,
                    0,
                    GemmABlockCopySrcDataPerRead_GemmK3,
                    GemmABlockCopyDstDataPerWrite_GemmM3,
                    GemmBBlockCopyThreadSliceLengths_GemmK_GemmN3,
                    GemmBBlockCopyThreadClusterLengths_GemmK_GemmN3,
                    Sequence<0, 1>,
                    Sequence<0, 1>,
                    1,
                    GemmBBlockCopySrcDataPerRead_GemmN3,
                    GemmBBlockCopyDstDataPerWrite_GemmN3,
                    Sequence<0, 1, 2, 3>,
                    3,
                    GemmCThreadCopyDstDataPerWrite_GemmN13,
                    GemmBlockID<Gemm128BlockNum + GemmN128BlockNum + GemmM128BlockNum>>{};

                gridwise_gemm3.Run(p_wei_global, p_in_global, p_out_global, p_shared_block);
            });
        }
    }
};

} // namespace ck
#endif
