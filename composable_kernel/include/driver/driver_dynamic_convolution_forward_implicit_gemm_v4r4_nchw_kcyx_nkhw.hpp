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
struct DriverDynamicConvolutionForwardImplicitGemm_v4r4_nchw_kcyx_nkhw_pad
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
        const auto wei_gemmk_gemmm_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(K, C * Y * X)),
            make_tuple(DynamicPassThrough{K}, DynamicPassThrough{C * Y * X}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        // input tensor
        const auto in_n_c_hip_wip_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(DynamicPassThrough{N},
                       DynamicPassThrough{C},
                       DynamicPad{Hi, InLeftPadH, InRightPadH},
                       DynamicPad{Wi, InLeftPadW, InRightPadW}),
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
        const auto out_gemmm_gemmn_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(N, K, Ho * Wo)),
            make_tuple(DynamicPassThrough{K}, DynamicMerge<2>{make_multi_index(N, Ho * Wo)}),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

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

#if 0
        const auto out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc =
            transform_dynamic_tensor_descriptor(
                out_gemmm_gemmn_global_desc,
                make_tuple(DynamicUnMerge<2>{make_multi_index(GemmM0, GemmM1)},
                           DynamicUnMerge<2>{make_multi_index(GemmN0, GemmN1)}),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
#else
        const auto GemmM0_GemmM1 = make_tuple(GemmM0, Number<GemmM1>{});
        const auto GemmN0_GemmN1 = make_tuple(GemmN0, Number<GemmN1>{});

        const auto out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc =
            transform_dynamic_tensor_descriptor(
                out_gemmm_gemmn_global_desc,
                make_tuple(
                    DynamicUnMerge<2, false, remove_cv_t<decltype(GemmM0_GemmM1)>>{GemmM0_GemmM1},
                    DynamicUnMerge<2, false, remove_cv_t<decltype(GemmN0_GemmN1)>>{GemmN0_GemmN1}),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
#endif

        // hack to control index calculation when iterating over a_k_m_global tensor
        constexpr auto a_k_m_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        constexpr auto a_k_m_global_move_slice_window_iterator_hack = Sequence<0, 0, 0>{};

        // hack to control index calculation when iterating over b_k_n_global tensor
        constexpr auto b_k_n_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2>{}));

        constexpr auto b_k_n_global_move_slice_window_iterator_hack =
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2>{};

        // hack to control index calculation when iterating over c_m0_m1_n0_n1_global tensor
        // hack for NKHW format
        constexpr auto c_m0_m1_n0_n1_global_tensor_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 1, 0, 0>{},
                                  Sequence<0, 0, 1, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 2, 0, 0>{},
                                  Sequence<0, 0, 2, 0, 0>{}));

        // GEMM
        using gridwise_gemm = GridwiseDynamicGemm_km_kn_mn_v1<
            BlockSize,
            Float,
            AccFloat,
            InMemoryDataOperation::Set,
            decltype(wei_gemmk_gemmm_global_desc),
            decltype(in_gemmk_gemmn_global_desc),
            decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
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
            false, // don't move back src coordinate after threadwise copy
            GemmBBlockTransferThreadSliceLengths_GemmK_GemmN,
            GemmBBlockTransferThreadClusterLengths_GemmK_GemmN,
            Sequence<0, 1>,
            Sequence<0, 1>,
            1,
            GemmBBlockTransferSrcScalarPerVector_GemmN,
            GemmBBlockTransferDstScalarPerVector_GemmN,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<2, 3, 0, 1>,
            3,
            GemmCThreadTransferDstScalarPerVector_GemmN1,
            decltype(a_k_m_global_iterator_hacks),
            decltype(b_k_n_global_iterator_hacks),
            decltype(c_m0_m1_n0_n1_global_tensor_iterator_hacks),
            decltype(a_k_m_global_move_slice_window_iterator_hack),
            decltype(b_k_n_global_move_slice_window_iterator_hack)>;

        const index_t GridSize = (GemmM / GemmMPerBlock) * (GemmN / GemmNPerBlock);

        const bool has_main_k_block_loop = (GemmK + GemmKPerBlock) / (2 * GemmKPerBlock) > 1;

        const bool has_double_tail_k_block_loop = (GemmK / GemmKPerBlock) % 2 == 0;

#if 1 // pass tensor descriptors by their reference
        index_t nrepeat = 100;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc),
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc),
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, true>,
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
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc),
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc),
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, true>,
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
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc),
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc),
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, false>,
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
                                  integral_constant<bool, false>{},
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
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, false>,
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
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k_ho_wo_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#elif 1 // pass tensor descriptors by their pointers
        using ADesc = decltype(wei_gemmk_gemmm_global_desc);
        using BDesc = decltype(in_gemmk_gemmn_global_desc);
        using CDesc = decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        DeviceMem wei_gemmk_gemmm_global_desc_device_buf(sizeof(ADesc));
        DeviceMem in_gemmk_gemmn_global_desc_device_buf(sizeof(BDesc));
        DeviceMem out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf(sizeof(CDesc));

        wei_gemmk_gemmm_global_desc_device_buf.ToDevice(&wei_gemmk_gemmm_global_desc);
        in_gemmk_gemmn_global_desc_device_buf.ToDevice(&in_gemmk_gemmn_global_desc);
        out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf.ToDevice(
            &out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        index_t nrepeat = 100;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, true>,
                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, true>,
                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, false>,
                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, true>{});
                }
                else
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, false>,
                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k_ho_wo_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#elif 1 // pass tensor descriptor by void*
        using ADesc = decltype(wei_gemmk_gemmm_global_desc);
        using BDesc = decltype(in_gemmk_gemmn_global_desc);
        using CDesc = decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        DeviceMem wei_gemmk_gemmm_global_desc_device_buf(sizeof(ADesc));
        DeviceMem in_gemmk_gemmn_global_desc_device_buf(sizeof(BDesc));
        DeviceMem out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf(sizeof(CDesc));

        wei_gemmk_gemmm_global_desc_device_buf.ToDevice(&wei_gemmk_gemmm_global_desc);
        in_gemmk_gemmn_global_desc_device_buf.ToDevice(&in_gemmk_gemmn_global_desc);
        out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf.ToDevice(
            &out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        index_t nrepeat = 100;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, true>{});
                }
                else
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k_ho_wo_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#endif
    }
};

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
struct DriverDynamicConvolutionForwardImplicitGemm_v4r4_nchw_kcyx_nkhw_no_pad
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

        if(!(InLeftPadH == 0 && InLeftPadW == 0 && InRightPadH == 0 && InRightPadW == 0))
        {
            throw std::runtime_error("wrong! 1x1, stride 1, no padding");
        }

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
            make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(K, C * Y * X)),
            make_tuple(DynamicPassThrough{K}, DynamicPassThrough{C * Y * X}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));
#endif

        // input tensor
        // debug: don't do padding
        const auto in_n_c_hip_wip_global_desc = in_n_c_hi_wi_global_desc;

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
            make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(N, K, Ho * Wo)),
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

        // hack to control index calculation when iterating over a_k_m_global tensor
        constexpr auto a_k_m_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        constexpr auto a_k_m_global_move_slice_window_iterator_hack = Sequence<0, 0, 0>{};

        // hack to control index calculation when iterating over b_k_n_global tensor
        constexpr auto b_k_n_global_iterator_hacks = make_tuple(
            make_tuple(Sequence<0, 0, 0, 0, 0, 1, 0>{}, Sequence<0, 0, 0, 0, 0, 0, 1>{}),
            make_tuple(Sequence<0, 0, 0, 0, 0, 2, 0>{}, Sequence<0, 0, 0, 0, 0, 0, 2>{}));

        constexpr auto b_k_n_global_move_slice_window_iterator_hack =
            Sequence<0, 0, 0, 0, 0, 1, 2>{};

        // hack to control index calculation when iterating over c_m0_m1_n0_n1_global tensor
        // hack for NKHW format
        constexpr auto c_m0_m1_n0_n1_global_tensor_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 1, 0, 0>{},
                                  Sequence<0, 0, 1, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 2, 0, 0>{},
                                  Sequence<0, 0, 2, 0, 0>{}));

        // GEMM
        using gridwise_gemm = GridwiseDynamicGemm_km_kn_mn_v1<
            BlockSize,
            Float,
            AccFloat,
            InMemoryDataOperation::Set,
            decltype(wei_gemmk_gemmm_global_desc),
            decltype(in_gemmk_gemmn_global_desc),
            decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
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
            false, // don't move back src coordinate after threadwise copy
            GemmBBlockTransferThreadSliceLengths_GemmK_GemmN,
            GemmBBlockTransferThreadClusterLengths_GemmK_GemmN,
            Sequence<0, 1>,
            Sequence<0, 1>,
            1,
            GemmBBlockTransferSrcScalarPerVector_GemmN,
            GemmBBlockTransferDstScalarPerVector_GemmN,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<2, 3, 0, 1>,
            3,
            GemmCThreadTransferDstScalarPerVector_GemmN1,
            decltype(a_k_m_global_iterator_hacks),
            decltype(b_k_n_global_iterator_hacks),
            decltype(c_m0_m1_n0_n1_global_tensor_iterator_hacks),
            decltype(a_k_m_global_move_slice_window_iterator_hack),
            decltype(b_k_n_global_move_slice_window_iterator_hack)>;

        const index_t GridSize = (GemmM / GemmMPerBlock) * (GemmN / GemmNPerBlock);

        const bool has_main_k_block_loop = (GemmK + GemmKPerBlock) / (2 * GemmKPerBlock) > 1;

        const bool has_double_tail_k_block_loop = (GemmK / GemmKPerBlock) % 2 == 0;

#if 1 // pass tensor descriptors by their reference
        index_t nrepeat = 100;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc),
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc),
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, true>,
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
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc),
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc),
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, true>,
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
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc),
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc),
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, false>,
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
                                  integral_constant<bool, false>{},
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
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, false>,
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
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k_ho_wo_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#elif 1 // pass tensor descriptors by their pointers
        using ADesc = decltype(wei_gemmk_gemmm_global_desc);
        using BDesc = decltype(in_gemmk_gemmn_global_desc);
        using CDesc = decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        DeviceMem wei_gemmk_gemmm_global_desc_device_buf(sizeof(ADesc));
        DeviceMem in_gemmk_gemmn_global_desc_device_buf(sizeof(BDesc));
        DeviceMem out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf(sizeof(CDesc));

        wei_gemmk_gemmm_global_desc_device_buf.ToDevice(&wei_gemmk_gemmm_global_desc);
        in_gemmk_gemmn_global_desc_device_buf.ToDevice(&in_gemmk_gemmn_global_desc);
        out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf.ToDevice(
            &out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        index_t nrepeat = 100;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, true>,
                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, true>,
                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, false>,
                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, true>{});
                }
                else
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, false>,
                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k_ho_wo_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#elif 1 // pass tensor descriptor by void*
        using ADesc = decltype(wei_gemmk_gemmm_global_desc);
        using BDesc = decltype(in_gemmk_gemmn_global_desc);
        using CDesc = decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        DeviceMem wei_gemmk_gemmm_global_desc_device_buf(sizeof(ADesc));
        DeviceMem in_gemmk_gemmn_global_desc_device_buf(sizeof(BDesc));
        DeviceMem out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf(sizeof(CDesc));

        wei_gemmk_gemmm_global_desc_device_buf.ToDevice(&wei_gemmk_gemmm_global_desc);
        in_gemmk_gemmn_global_desc_device_buf.ToDevice(&in_gemmk_gemmn_global_desc);
        out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf.ToDevice(
            &out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        index_t nrepeat = 100;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, true>{});
                }
                else
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k_ho_wo_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#endif
    }
};

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
struct DriverDynamicConvolutionForwardImplicitGemm_v4r4_nchw_kcyx_nkhw_1x1
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

        if(!(Y == 1 && X == 1 && ConvStrideH == 1 && ConvStrideW == 1 && ConvDilationH == 1 &&
             ConvDilationW == 1 && InLeftPadH == 0 && InLeftPadW == 0 && InRightPadH == 0 &&
             InRightPadW == 0))
        {
            throw std::runtime_error("wrong! 1x1, stride 1, no padding");
        }

        // weight tensor
        const auto wei_gemmk_gemmm_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(K, C)),
            make_tuple(DynamicPassThrough{K}, DynamicPassThrough{C}),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        // input tensor
        const auto in_gemmk_gemmn_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(DynamicPassThrough{C}, DynamicMerge<3>{make_multi_index(N, Ho, Wo)}),
            make_tuple(Sequence<1>{}, Sequence<0, 2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // output tensor
        const auto out_gemmm_gemmn_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(N, K, Ho * Wo)),
            make_tuple(DynamicPassThrough{K}, DynamicMerge<2>{make_multi_index(N, Ho * Wo)}),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

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

        // hack to control index calculation when iterating over a_k_m_global tensor
        constexpr auto a_k_m_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        constexpr auto a_k_m_global_move_slice_window_iterator_hack = Sequence<0, 0, 0>{};

        // hack to control index calculation when iterating over b_k_n_global tensor
        constexpr auto b_k_n_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 1, 0>{}, Sequence<0, 0, 1>{}),
                       make_tuple(Sequence<0, 2, 0>{}, Sequence<0, 0, 2>{}));

        constexpr auto b_k_n_global_move_slice_window_iterator_hack = Sequence<0, 1, 2>{};

        // hack to control index calculation when iterating over c_m0_m1_n0_n1_global tensor
        constexpr auto c_m0_m1_n0_n1_global_tensor_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 1, 0, 0>{},
                                  Sequence<0, 0, 1, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 2, 0, 0>{},
                                  Sequence<0, 0, 2, 0, 0>{}));

        // GEMM
        using gridwise_gemm = GridwiseDynamicGemm_km_kn_mn_v1<
            BlockSize,
            Float,
            AccFloat,
            InMemoryDataOperation::Set,
            decltype(wei_gemmk_gemmm_global_desc),
            decltype(in_gemmk_gemmn_global_desc),
            decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
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
            false, // don't move back src coordinate after threadwise copy
            GemmBBlockTransferThreadSliceLengths_GemmK_GemmN,
            GemmBBlockTransferThreadClusterLengths_GemmK_GemmN,
            Sequence<0, 1>,
            Sequence<0, 1>,
            1,
            GemmBBlockTransferSrcScalarPerVector_GemmN,
            GemmBBlockTransferDstScalarPerVector_GemmN,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<2, 3, 0, 1>,
            3,
            GemmCThreadTransferDstScalarPerVector_GemmN1,
            decltype(a_k_m_global_iterator_hacks),
            decltype(b_k_n_global_iterator_hacks),
            decltype(c_m0_m1_n0_n1_global_tensor_iterator_hacks),
            decltype(a_k_m_global_move_slice_window_iterator_hack),
            decltype(b_k_n_global_move_slice_window_iterator_hack)>;

        const index_t GridSize = (GemmM / GemmMPerBlock) * (GemmN / GemmNPerBlock);

        const bool has_main_k_block_loop = (GemmK + GemmKPerBlock) / (2 * GemmKPerBlock) > 1;

        const bool has_double_tail_k_block_loop = (GemmK / GemmKPerBlock) % 2 == 0;

#if 1 // pass tensor descriptors by their reference
        index_t nrepeat = 100;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc),
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc),
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, true>,
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
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc),
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc),
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, true>,
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
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc),
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc),
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, false>,
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
                                  integral_constant<bool, false>{},
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
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc),
                                               Float*,
                                               integral_constant<bool, false>,
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
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k_ho_wo_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#elif 1 // pass tensor descriptors by their pointers
        using ADesc = decltype(wei_gemmk_gemmm_global_desc);
        using BDesc = decltype(in_gemmk_gemmn_global_desc);
        using CDesc = decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        DeviceMem wei_gemmk_gemmm_global_desc_device_buf(sizeof(ADesc));
        DeviceMem in_gemmk_gemmn_global_desc_device_buf(sizeof(BDesc));
        DeviceMem out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf(sizeof(CDesc));

        wei_gemmk_gemmm_global_desc_device_buf.ToDevice(&wei_gemmk_gemmm_global_desc);
        in_gemmk_gemmn_global_desc_device_buf.ToDevice(&in_gemmk_gemmn_global_desc);
        out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf.ToDevice(
            &out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        index_t nrepeat = 100;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, true>,
                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, true>,
                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, false>,
                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, true>{});
                }
                else
                {
                    const auto kernel =
                        run_gridwise_operation<gridwise_gemm,
                                               decltype(wei_gemmk_gemmm_global_desc)*,
                                               const Float*,
                                               decltype(in_gemmk_gemmn_global_desc)*,
                                               const Float*,
                                               decltype(
                                                   out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc)*,
                                               Float*,
                                               integral_constant<bool, false>,
                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  reinterpret_cast<const ADesc*>(
                                      wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer()),
                                  p_wei_global,
                                  reinterpret_cast<const BDesc*>(
                                      in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer()),
                                  p_in_global,
                                  reinterpret_cast<const CDesc*>(
                                      out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                          .GetDeviceBuffer()),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k_ho_wo_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#elif 1 // pass tensor descriptor by void*
        using ADesc = decltype(wei_gemmk_gemmm_global_desc);
        using BDesc = decltype(in_gemmk_gemmn_global_desc);
        using CDesc = decltype(out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        DeviceMem wei_gemmk_gemmm_global_desc_device_buf(sizeof(ADesc));
        DeviceMem in_gemmk_gemmn_global_desc_device_buf(sizeof(BDesc));
        DeviceMem out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf(sizeof(CDesc));

        wei_gemmk_gemmm_global_desc_device_buf.ToDevice(&wei_gemmk_gemmm_global_desc);
        in_gemmk_gemmn_global_desc_device_buf.ToDevice(&in_gemmk_gemmn_global_desc);
        out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf.ToDevice(
            &out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc);

        index_t nrepeat = 100;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, true>{});
                }
                else
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               const Float*,
                                                               const void*,
                                                               Float*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  wei_gemmk_gemmm_global_desc_device_buf.GetDeviceBuffer(),
                                  p_wei_global,
                                  in_gemmk_gemmn_global_desc_device_buf.GetDeviceBuffer(),
                                  p_in_global,
                                  out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc_desc_device_buf
                                      .GetDeviceBuffer(),
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k_ho_wo_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#endif
    }
};

} // namespace ck
#endif
