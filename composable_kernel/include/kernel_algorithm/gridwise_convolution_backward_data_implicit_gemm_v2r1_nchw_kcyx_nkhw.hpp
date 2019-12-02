#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V2R1_NCHW_KCYX_NKHW_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V2R1_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccFloat,
          typename InGlobalDesc,
          typename WeiGlobalDesc,
          typename OutGlobalDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmThreadGemmDataPerReadM,
          index_t GemmThreadGemmDataPerReadN,
          typename GemmABlockCopySubLengths,     // Gemm-K, Gemm-M
          typename GemmABlockCopyClusterLengths, // Gemm-K, Gemm-M
          index_t GemmABlockCopyDataPerAccess,   // Gemm-M
          typename GemmBBlockCopySubLengths,     // Gemm-K, Gemm-N
          typename GemmBBlockCopyClusterLengths, // Gemm-K, Gemm-N
          index_t GemmBBlockCopyDataPerAccess,   // Gemm-N
          index_t GemmCThreadCopyDataPerAccess   // Gemm-N
          >
struct GridwiseConvolutionBackwardDataImplicitGemm_v2r1_nchw_kcyx_nkhw
{
    __device__ void Run(Float* __restrict__ p_in_global,
                        const Float* __restrict__ p_wei_global,
                        const Float* __restrict__ p_out_global) const
    {
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
        static_assert((Wo == 1 || (ConvStrideW == 1 || GemmCThreadCopyDataPerAccess == 1)) &&
                          (X == 1 || ConvDilationW % GemmCThreadCopyDataPerAccess == 0),
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // TODO: this algo support any stride and dilation. But for now, let's fix them to be 1 for
        // simplicity
        static_assert(ConvStrideH == 1 && ConvStrideW == 1 && ConvDilationH == 1 &&
                          ConvDilationW == 1,
                      "wrong! not supported yet");

        // TODO: these logic are only for stride = 1, dilation = 1
        constexpr index_t Ydot   = Y;
        constexpr index_t Ytilda = 1;
        constexpr index_t Htilda = Ho + Y - 1;

        constexpr index_t Xdot   = X;
        constexpr index_t Xtilda = 1;
        constexpr index_t Wtilda = Wo + X - 1;

        constexpr index_t GemmK = K * Ydot * Xdot;
        constexpr index_t GemmM = C * Ytilda * Xtilda;
        constexpr index_t GemmN = N * Htilda * Wtilda;

        // weight tensor
        constexpr auto wei_k_c_ydot_ytilda_xdot_xtilda_global_desc = transform_tensor_descriptor(
            wei_k_c_y_x_global_desc,
            make_tuple(
                PassThrough<K>{},
                PassThrough<C>{},
                Embed<Sequence<Ydot, Ytilda>, Sequence<1, 1, 0>>{},  // coefficient may be wrong
                Embed<Sequence<Xdot, Xtilda>, Sequence<1, 1, 0>>{}), // coefficient may be wrong
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto wei_gemmk_gemmm_global_desc = transform_tensor_descriptor(
            wei_k_c_ydot_ytilda_xdot_xtilda_global_desc,
            make_tuple(Merge<Sequence<K, Ydot, Xdot>>{}, Merge<Sequence<C, Ytilda, Xtilda>>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // output tensor
        constexpr auto out_n_k_hop_wop_global_desc = transform_tensor_descriptor(
            out_n_k_ho_wo_global_desc,
            make_tuple(
                PassThrough<N>{},
                PassThrough<K>{},
                Pad<Sequence<Ho, Wo>, Sequence<0, 0>, Sequence<Y - 1, X - 1>>{}), // coefficient may
                                                                                  // be wrong
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto out_n_k_ydot_htilda_xdot_wtilda_global_desc = transform_tensor_descriptor(
            out_n_k_hop_wop_global_desc,
            make_tuple(
                PassThrough<N>{},
                PassThrough<K>{},
                Embed<Sequence<Ydot, Htilda>, Sequence<0, 1, 0>>{},  // coefficient may be wrong
                Embed<Sequence<Xdot, Wtilda>, Sequence<0, 1, 0>>{}), // coefficient may be wrong
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto out_gemmk_gemmn_global_desc = transform_tensor_descriptor(
            out_n_k_ydot_htilda_xdot_wtilda_global_desc,
            make_tuple(Merge<Sequence<K, Ydot, Xdot>>{}, Merge<Sequence<N, Htilda, Wtilda>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // input tensor
        constexpr auto eff_left_pads  = LeftPads{} + Sequence<Y - 1, X - 1>{};
        constexpr auto eff_right_pads = RightPads{} + Sequence<Y - 1, X - 1>{};

        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Pad<Sequence<Hi, Wi>, decltype(eff_left_pads), decltype(eff_right_pads)>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto in_n_c_ytilda_htilda_xtilda_wtilda_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Sequence<Ytilda, Htilda>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Sequence<Xtilda, Wtilda>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto in_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            in_n_c_ytilda_htilda_xtilda_wtilda_global_desc,
            make_tuple(Merge<Sequence<C, Ytilda, Xtilda>>{}, Merge<Sequence<N, Htilda, Wtilda>>{}),
            make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // GEMM
        constexpr auto gridwise_gemm =
            GridwiseGemmTransposedANormalBNormalC_v1r1<GridSize,
                                                       BlockSize,
                                                       Float,
                                                       AccFloat,
                                                       decltype(wei_gemmk_gemmm_global_desc),
                                                       decltype(out_gemmk_gemmn_global_desc),
                                                       decltype(in_gemmm_gemmn_global_desc),
                                                       InMemoryDataOperation::none,
                                                       GemmMPerBlock,
                                                       GemmNPerBlock,
                                                       GemmKPerBlock,
                                                       GemmMPerThreadSubC,
                                                       GemmNPerThreadSubC,
                                                       GemmMLevel0Cluster,
                                                       GemmNLevel0Cluster,
                                                       GemmMLevel1Cluster,
                                                       GemmNLevel1Cluster,
                                                       GemmKPerThreadLoop,
                                                       GemmThreadGemmDataPerReadM,
                                                       GemmThreadGemmDataPerReadN,
                                                       GemmABlockCopySubLengths,
                                                       GemmABlockCopyClusterLengths,
                                                       GemmABlockCopyDataPerAccess,
                                                       GemmBBlockCopySubLengths,
                                                       GemmBBlockCopyClusterLengths,
                                                       GemmBBlockCopyDataPerAccess,
                                                       GemmCThreadCopyDataPerAccess>{};

        gridwise_gemm.Run(p_wei_global, p_out_global, p_in_global);
    }
};

} // namespace ck
#endif
