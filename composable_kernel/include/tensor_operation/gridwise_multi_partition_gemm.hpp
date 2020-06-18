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
          typename ABlockCopyThreadClusterArrangeOrder,
          typename ABlockCopySrcAccessOrder,
          index_t ABlockCopySrcVectorReadDim,
          typename BBlockCopyThreadClusterArrangeOrder,
          typename BBlockCopySrcAccessOrder,
          index_t BBlockCopySrcVectorReadDim,
          typename CThreadCopySrcDstAccessOrder,
          index_t CThreadCopySrcDstVectorReadWriteDim,
          typename GemmParameters1,
          typename GemmParameters2,
          typename GemmParameters3,
          typename GemmParameters4>
struct GridwiseMultiPartitionGemmTransposedANormalBNormalC_v1
{
    static constexpr auto partition1 = GemmParameters1{};
    static constexpr auto partition2 = GemmParameters2{};
    static constexpr auto partition3 = GemmParameters3{};
    static constexpr auto partition4 = GemmParameters4{};
    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr index_t max_lds_align = math::lcm(partition1.ABlockCopyDstDataPerWrite_M,
                                                    partition1.BBlockCopyDstDataPerWrite_N,
                                                    partition1.ThreadGemmAThreadCopySrcDataPerRead_M,
                                                    partition1.ThreadGemmBThreadCopySrcDataPerRead_N);

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<partition1.KPerBlock, partition1.MPerBlock>{}, Number<max_lds_align>{});

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<partition1.KPerBlock, partition1.NPerBlock>{}, Number<max_lds_align>{});

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

                constexpr auto gridwise_gemm = GridwiseGemmTransposedANormalBNormalC_v2<
                    GridSize,
                    partition1.BlockSize,
                    Float,
                    AccFloat,
                    decltype(wei_e_k_global_1st_desc),
                    decltype(in_e_b_global_1st_desc),
                    decltype(out_k_b_global_1st_desc),
                    CGlobalMemoryDataOperation,
                    partition1.MPerBlock,
                    partition1.NPerBlock,
                    partition1.KPerBlock,
                    partition1.MPerThread,
                    partition1.NPerThread,
                    partition1.KPerThread,
                    partition1.MLevel0Cluster,
                    partition1.NLevel0Cluster,
                    partition1.MLevel1Cluster,
                    partition1.NLevel1Cluster,
                    partition1.ThreadGemmAThreadCopySrcDataPerRead_M,
                    partition1.ThreadGemmBThreadCopySrcDataPerRead_N,
                    typename GemmParameters1::ABlockCopyThreadSliceLengths_K_M,
                    typename GemmParameters1::ABlockCopyThreadClusterLengths_K_M,
                    ABlockCopyThreadClusterArrangeOrder,
                    ABlockCopySrcAccessOrder,
                    ABlockCopySrcVectorReadDim,
                    partition1.ABlockCopySrcDataPerRead_K,
                    partition1.ABlockCopyDstDataPerWrite_M,
                    typename GemmParameters1::BBlockCopyThreadSliceLengths_K_N,
                    typename GemmParameters1::BBlockCopyThreadClusterLengths_K_N,
                    BBlockCopyThreadClusterArrangeOrder,
                    BBlockCopySrcAccessOrder,
                    BBlockCopySrcVectorReadDim,
                    partition1.BBlockCopySrcDataPerRead_N,
                    partition1.BBlockCopyDstDataPerWrite_N,
                    CThreadCopySrcDstAccessOrder,
                    CThreadCopySrcDstVectorReadWriteDim,
                    partition1.CThreadCopyDstDataPerWrite,
                    GemmBlockID<0>>{};

                gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global, p_shared_block);
            });
        }
        else if(bIsHave2ndPartition && (get_block_1d_id() < Gemm128BlockNum + GemmN128BlockNum))
        {
            static_if<bIsHave2ndPartition>{}([&](auto){
                constexpr auto out_k_b_global_2nd_desc = transform_tensor_descriptor(
                    out_k_b_global_desc,
                    make_tuple(Slice<Sequence<GemmM>, Sequence<GemmOBeginM>, Sequence<GemmM>>{},
                               Slice<Sequence<GemmN>, Sequence<0>, Sequence<GemmOBeginN>>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
                constexpr auto gridwise_gemm = GridwiseGemmTransposedANormalBNormalC_v2<
                    GridSize,
                    partition2.BlockSize,
                    Float,
                    AccFloat,
                    decltype(wei_e_k_global_2nd_desc),
                    decltype(in_e_b_global_1st_desc),
                    decltype(out_k_b_global_2nd_desc),
                    CGlobalMemoryDataOperation,
                    partition2.MPerBlock,
                    partition2.NPerBlock,
                    partition2.KPerBlock,
                    partition2.MPerThread,
                    partition2.NPerThread,
                    partition2.KPerThread,
                    partition2.MLevel0Cluster,
                    partition2.NLevel0Cluster,
                    partition2.MLevel1Cluster,
                    partition2.NLevel1Cluster,
                    partition2.ThreadGemmAThreadCopySrcDataPerRead_M,
                    partition2.ThreadGemmBThreadCopySrcDataPerRead_N,
                    typename GemmParameters2::ABlockCopyThreadSliceLengths_K_M,
                    typename GemmParameters2::ABlockCopyThreadClusterLengths_K_M,
                    ABlockCopyThreadClusterArrangeOrder,
                    ABlockCopySrcAccessOrder,
                    ABlockCopySrcVectorReadDim,
                    partition2.ABlockCopySrcDataPerRead_K,
                    partition2.ABlockCopyDstDataPerWrite_M,
                    typename GemmParameters2::BBlockCopyThreadSliceLengths_K_N,
                    typename GemmParameters2::BBlockCopyThreadClusterLengths_K_N,
                    BBlockCopyThreadClusterArrangeOrder,
                    BBlockCopySrcAccessOrder,
                    BBlockCopySrcVectorReadDim,
                    partition2.BBlockCopySrcDataPerRead_N,
                    partition2.BBlockCopyDstDataPerWrite_N,
                    CThreadCopySrcDstAccessOrder,
                    CThreadCopySrcDstVectorReadWriteDim,
                    partition2.CThreadCopyDstDataPerWrite,
                    GemmBlockID<Gemm128BlockNum>>{};

                gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global, p_shared_block);
            });
        }
        else if(bIsHave3rdPartition && (get_block_1d_id() < bolck_begin_3rd + GemmM128BlockNum))
        {
            static_if<bIsHave3rdPartition>{}([&](auto){
                constexpr auto out_k_b_global_3rd_desc = transform_tensor_descriptor(
                    out_k_b_global_desc,
                    make_tuple(Slice<Sequence<GemmM>, Sequence<0>, Sequence<GemmOBeginM>>{},
                               Slice<Sequence<GemmN>, Sequence<GemmOBeginN>, Sequence<GemmN>>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                constexpr auto gridwise_gemm = GridwiseGemmTransposedANormalBNormalC_v2<
                    GridSize,
                    partition3.BlockSize,
                    Float,
                    AccFloat,
                    decltype(wei_e_k_global_1st_desc),
                    decltype(in_e_b_global_2nd_desc),
                    decltype(out_k_b_global_3rd_desc),
                    CGlobalMemoryDataOperation,
                    partition3.MPerBlock,
                    partition3.NPerBlock,
                    partition3.KPerBlock,
                    partition3.MPerThread,
                    partition3.NPerThread,
                    partition3.KPerThread,
                    partition3.MLevel0Cluster,
                    partition3.NLevel0Cluster,
                    partition3.MLevel1Cluster,
                    partition3.NLevel1Cluster,
                    partition3.ThreadGemmAThreadCopySrcDataPerRead_M,
                    partition3.ThreadGemmBThreadCopySrcDataPerRead_N,
                    typename GemmParameters3::ABlockCopyThreadSliceLengths_K_M,
                    typename GemmParameters3::ABlockCopyThreadClusterLengths_K_M,
                    ABlockCopyThreadClusterArrangeOrder,
                    ABlockCopySrcAccessOrder,
                    ABlockCopySrcVectorReadDim,
                    partition3.ABlockCopySrcDataPerRead_K,
                    partition3.ABlockCopyDstDataPerWrite_M,
                    typename GemmParameters3::BBlockCopyThreadSliceLengths_K_N,
                    typename GemmParameters3::BBlockCopyThreadClusterLengths_K_N,
                    BBlockCopyThreadClusterArrangeOrder,
                    BBlockCopySrcAccessOrder,
                    BBlockCopySrcVectorReadDim,
                    partition3.BBlockCopySrcDataPerRead_N,
                    partition3.BBlockCopyDstDataPerWrite_N,
                    CThreadCopySrcDstAccessOrder,
                    CThreadCopySrcDstVectorReadWriteDim,
                    partition3.CThreadCopyDstDataPerWrite,
                    GemmBlockID<bolck_begin_3rd>>{};

                gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global, p_shared_block);
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

                constexpr auto gridwise_gemm = GridwiseGemmTransposedANormalBNormalC_v2<
                    GridSize,
                    partition4.BlockSize,
                    Float,
                    AccFloat,
                    decltype(wei_e_k_global_2nd_desc),
                    decltype(in_e_b_global_2nd_desc),
                    decltype(out_k_b_global_4th_desc),
                    CGlobalMemoryDataOperation,
                    partition4.MPerBlock,
                    partition4.NPerBlock,
                    partition4.KPerBlock,
                    partition4.MPerThread,
                    partition4.NPerThread,
                    partition4.KPerThread,
                    partition4.MLevel0Cluster,
                    partition4.NLevel0Cluster,
                    partition4.MLevel1Cluster,
                    partition4.NLevel1Cluster,
                    partition4.ThreadGemmAThreadCopySrcDataPerRead_M,
                    partition4.ThreadGemmBThreadCopySrcDataPerRead_N,
                    typename GemmParameters4::ABlockCopyThreadSliceLengths_K_M,
                    typename GemmParameters4::ABlockCopyThreadClusterLengths_K_M,
                    ABlockCopyThreadClusterArrangeOrder,
                    ABlockCopySrcAccessOrder,
                    ABlockCopySrcVectorReadDim,
                    partition4.ABlockCopySrcDataPerRead_K,
                    partition4.ABlockCopyDstDataPerWrite_M,
                    typename GemmParameters4::BBlockCopyThreadSliceLengths_K_N,
                    typename GemmParameters4::BBlockCopyThreadClusterLengths_K_N,
                    BBlockCopyThreadClusterArrangeOrder,
                    BBlockCopySrcAccessOrder,
                    BBlockCopySrcVectorReadDim,
                    partition4.BBlockCopySrcDataPerRead_N,
                    partition4.BBlockCopyDstDataPerWrite_N,
                    CThreadCopySrcDstAccessOrder,
                    CThreadCopySrcDstVectorReadWriteDim,
                    partition4.CThreadCopyDstDataPerWrite,
                    GemmBlockID<Gemm128BlockNum + GemmN128BlockNum + GemmM128BlockNum>>{};

                gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global, p_shared_block);
            });
        }
    }
};

} // namespace ck
#endif
