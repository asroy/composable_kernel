#ifndef CK_DRIVER_DYNAMIC_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_NCHW_KCYX_NKHW_HPP
#define CK_DRIVER_DYNAMIC_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm.hpp"
#include "gridwise_operation_wrapper.hpp"

namespace ck {

// GemmM = K
// GemmN = N * Ho * Wo
// GemmK = C * Y * X
template <index_t BlockSize,
          typename Float,
          typename AccFloat,
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
          typename GemmABlockTransferThreadSliceLengths_GemmK_GemmM,
          typename GemmABlockTransferThreadClusterLengths_GemmK_GemmM,
          index_t GemmABlockTransferSrcScalarPerVector_GemmK,
          index_t GemmABlockTransferDstScalarPerVector_GemmM,
          typename GemmBBlockTransferThreadSliceLengths_GemmK_GemmN,
          typename GemmBBlockTransferThreadClusterLengths_GemmK_GemmN,
          index_t GemmBBlockTransferSrcScalarPerVector_GemmN,
          index_t GemmBBlockTransferDstScalarPerVector_GemmN,
          index_t GemmCThreadTransferDstScalarPerVector_GemmN1>
struct DriverDynamicConvolutionForwardImplicitGemm_v4r4_nchw_kcyx_nkhw
{
    template <typename... Wei, typename... In, typename... Out>
    __host__ void Run(const DynamicTensorDescriptor<Wei...>& wei_k_c_y_x_global_desc,
                      const DynamicTensorDescriptor<In...>& in_n_c_hi_wi_global_desc,
                      const DynamicTensorDescriptor<Out...>& out_n_k_ho_wo_global_desc,
                      const MultiIndex<2> conv_strides,
                      const MultiIndex<2> conv_dilations,
                      const MultiIndex<2> in_left_pads,
                      const MultiIndex<2> in_right_pads,
                      const Float* __restrict__ p_wei_global,
                      const Float* __restrict__ p_in_global,
                      Float* __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        const index_t N = in_n_c_hi_wi_global_desc.GetLength(I0);
        const index_t C = in_n_c_hi_wi_global_desc.GetLength(I1);
        const index_t K = out_n_k_ho_wo_global_desc.GetLength(I1);

        const index_t Hi = in_n_c_hi_wi_global_desc.GetLength(I2);
        const index_t Wi = in_n_c_hi_wi_global_desc.GetLength(I3);

        const index_t Ho = out_n_k_ho_wo_global_desc.GetLength(I2);
        const index_t Wo = out_n_k_ho_wo_global_desc.GetLength(I3);

        const index_t Y = wei_k_c_y_x_global_desc.GetLength(I2);
        const index_t X = wei_k_c_y_x_global_desc.GetLength(I3);

        const index_t ConvStrideH = conv_strides[I0];
        const index_t ConvStrideW = conv_strides[I1];

        const index_t ConvDilationH = conv_dilations[I0];
        const index_t ConvDilationW = conv_dilations[I1];

        const index_t InLeftPadH = in_left_pads[I0];
        const index_t InLeftPadW = in_left_pads[I1];

        const index_t InRightPadH = in_right_pads[I0];
        const index_t InRightPadW = in_right_pads[I1];

        // weight tensor
#if 0
        // TODO implement graph optimization of tensor descriptor transformation
        const auto wei_gemmk_gemmm_global_desc = transform_dynamic_tensor_descriptor(
            wei_k_c_y_x_global_desc,
            make_tuple(DynamicPassThrough{K}, DynamicMerge<3>{make_multi_index(C, Y, X)}),
            make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));
#else
        const auto wei_gemmk_gemmm_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed<2>(make_multi_index(K, C * Y * X)),
            make_tuple(DynamicPassThrough{K}, DynamicPassThrough{C * Y * X}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));
#endif

        // input tensor
        const auto in_n_c_hip_wip_global_desc = transform_dynamic_tensor_descriptor(
            transform_dynamic_tensor_descriptor(
                in_n_c_hi_wi_global_desc,
                make_tuple(DynamicPassThrough{N},
                           DynamicPassThrough{C},
                           DynamicLeftPad{Hi, InLeftPadH},
                           DynamicLeftPad{Wi, InLeftPadW}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{})),
            make_tuple(DynamicPassThrough{N},
                       DynamicPassThrough{C},
                       DynamicRightPad{Hi + InLeftPadH, InRightPadH},
                       DynamicRightPad{Wi + InLeftPadW, InRightPadW}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const index_t Hip = in_n_c_hip_wip_global_desc.GetLength(I2);
        const index_t Wip = in_n_c_hip_wip_global_desc.GetLength(I3);

        const auto in_n_c_y_ho_x_wo_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(DynamicPassThrough{N},
                       DynamicPassThrough{C},
                       DynamicEmbed<2>{make_multi_index(Y, Ho),
                                       make_multi_index(ConvDilationH, ConvStrideH)},
                       DynamicEmbed<2>{make_multi_index(X, Wo),
                                       make_multi_index(ConvDilationW, ConvStrideW)}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        const auto in_gemmk_gemmn_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_y_ho_x_wo_global_desc,
            make_tuple(DynamicMerge<3>{make_multi_index(C, Y, X)},
                       DynamicMerge<3>{make_multi_index(N, Ho, Wo)}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // output tensor
#if 0
        //TODO: implement graph optimization of tensor descriptor transformation
        const auto out_gemmm_gemmn_global_desc =
            transform_dynamic_tensor_descriptor(out_n_k_ho_wo_global_desc,
                                        make_tuple(DynamicPassThrough{K}, DynamicMerge<3>{make_mult_index(N, Ho, Wo)}),
                                        make_tuple(Sequence<1>{}, Sequence<0, 2, 3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
#else
        const auto out_gemmm_gemmn_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed<3>(make_multi_index(N, K, Ho * Wo)),
            make_tuple(DynamicPassThrough{K}, DynamicMerge<2>{make_multi_index(N, Ho * Wo)}),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
#endif

        const index_t GemmM = out_gemmm_gemmn_global_desc.GetLength(I0);
        const index_t GemmN = out_gemmm_gemmn_global_desc.GetLength(I1);
        const index_t GemmK = wei_gemmk_gemmm_global_desc.GetLength(I0);

        if(!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
             GemmK % GemmKPerBlock == 0))
        {
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        constexpr index_t GemmM1 = GemmMPerThread * GemmMLevel0Cluster * GemmMLevel1Cluster;
        constexpr index_t GemmN1 = GemmNPerThread * GemmNLevel0Cluster * GemmNLevel1Cluster;

        const index_t GemmM0 = GemmM / GemmM1;
        const index_t GemmN0 = GemmN / GemmN1;

        const auto out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc =
            transform_dynamic_tensor_descriptor(
                out_gemmm_gemmn_global_desc,
                make_tuple(DynamicUnMerge<2>{make_multi_index(GemmM0, GemmM1)},
                           DynamicUnMerge<2>{make_multi_index(GemmN0, GemmN1)}),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        // GEMM
#if 1
        using gridwise_gemm =
            GridwiseDynamicGemm_km_kn_mn_v1<BlockSize,
                                            Float,
                                            AccFloat,
                                            InMemoryDataOperation::Set,
                                            GemmMPerBlock,
                                            GemmNPerBlock,
                                            GemmKPerBlock,
                                            GemmMPerThread,
                                            GemmNPerThread,
                                            GemmKPerThread,
                                            GemmMLevel0Cluster,
                                            GemmNLevel0Cluster,
                                            GemmMLevel1Cluster,
                                            GemmNLevel1Cluster,
                                            GemmABlockTransferThreadSliceLengths_GemmK_GemmM,
                                            GemmABlockTransferThreadClusterLengths_GemmK_GemmM,
                                            Sequence<1, 0>,
                                            Sequence<1, 0>,
                                            0,
                                            GemmABlockTransferSrcScalarPerVector_GemmK,
                                            GemmABlockTransferDstScalarPerVector_GemmM,
                                            GemmBBlockTransferThreadSliceLengths_GemmK_GemmN,
                                            GemmBBlockTransferThreadClusterLengths_GemmK_GemmN,
                                            Sequence<0, 1>,
                                            Sequence<0, 1>,
                                            1,
                                            GemmBBlockTransferSrcScalarPerVector_GemmN,
                                            GemmBBlockTransferDstScalarPerVector_GemmN,
                                            Sequence<2, 3, 0, 1>,
                                            3,
                                            GemmCThreadTransferDstScalarPerVector_GemmN1>;

        const index_t GridSize = (GemmM / GemmMPerBlock) * (GemmN / GemmNPerBlock);

        const bool is_even_number_k_block_loop = (GemmK / GemmKPerBlock) % 2 == 0;

        if(is_even_number_k_block_loop)
        {
            const auto kernel =
                run_gridwise_operation<gridwise_gemm,
                                       decltype(wei_gemmk_gemmm_global_desc),
                                       const Float*,
                                       decltype(in_gemmk_gemmn_global_desc),
                                       const Float*,
                                       decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                       Float*,
                                       integral_constant<bool, true>>;

            launch_kernel(kernel,
                          dim3(GridSize),
                          dim3(BlockSize),
                          0,
                          0,
                          wei_gemmk_gemmm_global_desc,
                          p_wei_global,
                          in_gemmk_gemmn_global_desc,
                          p_in_global,
                          out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc,
                          p_out_global,
                          integral_constant<bool, true>{});
        }
        else
        {
            const auto kernel =
                run_gridwise_operation<gridwise_gemm,
                                       decltype(wei_gemmk_gemmm_global_desc),
                                       const Float*,
                                       decltype(in_gemmk_gemmn_global_desc),
                                       const Float*,
                                       decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                       Float*,
                                       integral_constant<bool, false>>;

            launch_kernel(kernel,
                          dim3(GridSize),
                          dim3(BlockSize),
                          0,
                          0,
                          wei_gemmk_gemmm_global_desc,
                          p_wei_global,
                          in_gemmk_gemmn_global_desc,
                          p_in_global,
                          out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc,
                          p_out_global,
                          integral_constant<bool, false>{});
        }
#else
        using gridwise_gemm =
            GridwiseDynamicGemm_km_kn_mn_v2<BlockSize,
                                            Float,
                                            AccFloat,
                                            InMemoryDataOperation::Set,
                                            GemmMPerBlock,
                                            GemmNPerBlock,
                                            GemmKPerBlock,
                                            GemmMPerThread,
                                            GemmNPerThread,
                                            GemmKPerThread,
                                            GemmMLevel0Cluster,
                                            GemmNLevel0Cluster,
                                            GemmMLevel1Cluster,
                                            GemmNLevel1Cluster,
                                            GemmABlockTransferThreadSliceLengths_GemmK_GemmM,
                                            GemmABlockTransferThreadClusterLengths_GemmK_GemmM,
                                            Sequence<1, 0>,
                                            Sequence<1, 0>,
                                            0,
                                            GemmABlockTransferSrcScalarPerVector_GemmK,
                                            GemmABlockTransferDstScalarPerVector_GemmM,
                                            GemmBBlockTransferThreadSliceLengths_GemmK_GemmN,
                                            GemmBBlockTransferThreadClusterLengths_GemmK_GemmN,
                                            Sequence<0, 1>,
                                            Sequence<0, 1>,
                                            1,
                                            GemmBBlockTransferSrcScalarPerVector_GemmN,
                                            GemmBBlockTransferDstScalarPerVector_GemmN,
                                            Sequence<2, 3, 0, 1>,
                                            3,
                                            GemmCThreadTransferDstScalarPerVector_GemmN1>;

        const index_t GridSize = (GemmM / GemmMPerBlock) * (GemmN / GemmNPerBlock);

        const auto kernel =
            run_gridwise_operation<gridwise_gemm,
                                   decltype(wei_gemmk_gemmm_global_desc),
                                   const Float*,
                                   decltype(in_gemmk_gemmn_global_desc),
                                   const Float*,
                                   decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                   Float*>;

        launch_kernel(kernel,
                      dim3(GridSize),
                      dim3(BlockSize),
                      0,
                      0,
                      wei_gemmk_gemmm_global_desc,
                      p_wei_global,
                      in_gemmk_gemmn_global_desc,
                      p_in_global,
                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc,
                      p_out_global);
#endif
    }
};

} // namespace ck
#endif
