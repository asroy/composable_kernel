#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_NCHW_KCYX_NKHW_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
//#include "gridwise_gemm.hpp"
#include "gridwise_multi_partition_gemm.hpp"

namespace ck {
template <bool    GemmIsValid,
          index_t GemmBlockSize,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerThread,
          index_t GemmNPerThread,
          index_t GemmKPerThread,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t ThreadGemmDataPerRead_GemmM,
          index_t ThreadGemmDataPerRead_GemmN,
          typename GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
          typename GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
          index_t GemmABlockCopySrcDataPerRead_GemmK,
          index_t GemmABlockCopyDstDataPerWrite_GemmM,
          typename GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
          typename GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN,
          index_t GemmCThreadCopyDstDataPerWrite_GemmN1,
          index_t GemmBlockBeginId,
          index_t GemmBlockEndId>
struct GemmParameters{
    using ABlockCopyThreadSliceLengths_K_M                           = GemmABlockCopyThreadSliceLengths_GemmK_GemmM;
    using ABlockCopyThreadClusterLengths_K_M                         = GemmABlockCopyThreadClusterLengths_GemmK_GemmM;
    using BBlockCopyThreadSliceLengths_K_N                           = GemmBBlockCopyThreadSliceLengths_GemmK_GemmN;
    using BBlockCopyThreadClusterLengths_K_N                         = GemmBBlockCopyThreadClusterLengths_GemmK_GemmN;

     static constexpr index_t BlockSize                              = GemmBlockSize;
     static constexpr index_t MPerBlock                              = GemmMPerBlock;
     static constexpr index_t NPerBlock                              = GemmNPerBlock;
     static constexpr index_t KPerBlock                              = GemmKPerBlock;
     static constexpr index_t MPerThread                             = GemmMPerThread;
     static constexpr index_t NPerThread                             = GemmNPerThread;
     static constexpr index_t KPerThread                             = GemmKPerThread;
     static constexpr index_t MLevel0Cluster                         = GemmMLevel0Cluster;
     static constexpr index_t NLevel0Cluster                         = GemmNLevel0Cluster;
     static constexpr index_t MLevel1Cluster                         = GemmMLevel1Cluster;
     static constexpr index_t NLevel1Cluster                         = GemmNLevel1Cluster;
     static constexpr index_t ThreadGemmAThreadCopySrcDataPerRead_M  = ThreadGemmDataPerRead_GemmM;
     static constexpr index_t ThreadGemmBThreadCopySrcDataPerRead_N  = ThreadGemmDataPerRead_GemmN;
     static constexpr index_t ABlockCopySrcDataPerRead_K             = GemmABlockCopySrcDataPerRead_GemmK;
     static constexpr index_t ABlockCopyDstDataPerWrite_M            = GemmABlockCopyDstDataPerWrite_GemmM;
     static constexpr index_t BBlockCopySrcDataPerRead_N             = GemmBBlockCopySrcDataPerRead_GemmN;
     static constexpr index_t BBlockCopyDstDataPerWrite_N            = GemmBBlockCopyDstDataPerWrite_GemmN;
     static constexpr index_t CThreadCopyDstDataPerWrite             = GemmCThreadCopyDstDataPerWrite_GemmN1;
     static constexpr index_t BlockBeginId                           = GemmBlockBeginId;
     static constexpr index_t BlockEndId                             = GemmBlockEndId;

     __host__ __device__ static constexpr bool IsValid(){
         return GemmIsValid;
     }
};
// GemmM = K
// GemmN = N * Ho * Wo
// GemmK = C * Y * X
template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccFloat,
          typename InGlobalDesc,
          typename WeiGlobalDesc,
          typename OutGlobalDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          typename GemmParamters1,
          typename GemmParamters2,
          typename GemmParamters3,
          typename GemmParamters4,
          index_t  GemmOBeginM,
          index_t  GemmOBeginN>
struct GridwiseConvolutionImplicitGemm_v4r4_nchw_kcyx_nkhw_mp
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto in_n_c_hi_wi_global_desc  = InGlobalDesc{};
        constexpr auto wei_k_c_y_x_global_desc   = WeiGlobalDesc{};
        constexpr auto out_n_k_ho_wo_global_desc = OutGlobalDesc{};

        constexpr index_t N  = in_n_c_hi_wi_global_desc.GetLengths()[0];
        constexpr index_t C  = in_n_c_hi_wi_global_desc.GetLengths()[1];
        constexpr index_t Hi = in_n_c_hi_wi_global_desc.GetLengths()[2];
        constexpr index_t Wi = in_n_c_hi_wi_global_desc.GetLengths()[3];

        constexpr index_t K  = out_n_k_ho_wo_global_desc.GetLengths()[1];
        constexpr index_t Ho = out_n_k_ho_wo_global_desc.GetLengths()[2];
        constexpr index_t Wo = out_n_k_ho_wo_global_desc.GetLengths()[3];

        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLengths()[2];
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLengths()[3];

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        // sanity-check for vectorized memory load
        constexpr auto partition1 = GemmParamters1{};
        static_assert((Wo == 1 || (ConvStrideW == 1 || partition1.BBlockCopySrcDataPerRead_N == 1)) &&
                          (X == 1 || ConvDilationW % partition1.BBlockCopySrcDataPerRead_N == 0) &&
                          InLeftPads{}[1] % partition1.BBlockCopySrcDataPerRead_N == 0 &&
                          InRightPads{}[1] % partition1.BBlockCopySrcDataPerRead_N == 0,
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // weight tensor
        constexpr auto wei_e_k_global_desc = reorder_tensor_descriptor_given_upper2lower(
            unfold_tensor_descriptor(wei_k_c_y_x_global_desc, I1, I3), Sequence<1, 0>{});

        // input tensor
        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Pad<Sequence<Hi, Wi>, InLeftPads, InRightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr index_t Hip = in_n_c_hip_wip_global_desc.GetLengths()[2];
        constexpr index_t Wip = in_n_c_hip_wip_global_desc.GetLengths()[3];

        constexpr auto in_n_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Hip, Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wip, Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto in_e_b_global_desc = transform_tensor_descriptor(
            in_n_c_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, Y, X>>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // output tensor
        constexpr auto out_k_b_global_desc =
            transform_tensor_descriptor(unfold_tensor_descriptor(out_n_k_ho_wo_global_desc, I2, I3),
                                        make_tuple(PassThrough<K>{}, Merge<Sequence<N, Ho * Wo>>{}),
                                        make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        // GEMM
        constexpr auto gridwise_gemm =
            GridwiseMultiPartitionGemmTransposedANormalBNormalC_v1<GridSize,
                                                     BlockSize,
                                                     Float,
                                                     AccFloat,
                                                     decltype(wei_e_k_global_desc),
                                                     decltype(in_e_b_global_desc),
                                                     decltype(out_k_b_global_desc),
                                                     InMemoryDataOperation::Set,
                                                     Sequence<1, 0>,
                                                     Sequence<1, 0>,
                                                     0,
                                                     Sequence<0, 1>,
                                                     Sequence<0, 1>,
                                                     1,
                                                     Sequence<0, 1, 2, 3>,
                                                     3,
                                                     GemmParamters1,
                                                     GemmParamters2,
                                                     GemmParamters3,
                                                     GemmParamters4,
                                                     GemmOBeginM,
                                                     GemmOBeginN>{};

        gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global);
        
    }
};

} // namespace ck
#endif
