#ifndef CK_DRIVER_DYNAMIC_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_CHWN_CYXK_KHWN_HPP
#define CK_DRIVER_DYNAMIC_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_CHWN_CYXK_KHWN_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm.hpp"
#include "gridwise_operation_wrapper.hpp"

namespace ck {

// GemmM = K
// GemmN = N * Ho * Wo
// GemmK = Y * X * C
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
          index_t GemmCThreadTransferDstScalarPerVector_GemmM1>
struct DriverDynamicConvolutionForwardImplicitGemm_v4r4_chwn_cyxk_khwn_pad
{
    template <typename... Wei,
              typename... In,
              typename... Out,
              typename ConvStrides,
              typename ConvDilations,
              typename InLeftPads,
              typename InRightPads>
    __host__ void Run(const DynamicTensorDescriptor<Wei...>& wei_c_y_x_k_global_desc,
                      const DynamicTensorDescriptor<In...>& in_c_hi_wi_n_global_desc,
                      const DynamicTensorDescriptor<Out...>& out_k_ho_wo_n_global_desc,
                      const ConvStrides& conv_strides,
                      const ConvDilations& conv_dilations,
                      const InLeftPads& in_left_pads,
                      const InRightPads& in_right_pads,
                      const Float* __restrict__ p_wei_global,
                      const Float* __restrict__ p_in_global,
                      Float* __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        const auto N = in_c_hi_wi_n_global_desc.GetLength(I3);
        const auto C = in_c_hi_wi_n_global_desc.GetLength(I0);
        const auto K = out_k_ho_wo_n_global_desc.GetLength(I0);

        const auto Hi = in_c_hi_wi_n_global_desc.GetLength(I1);
        const auto Wi = in_c_hi_wi_n_global_desc.GetLength(I2);

        const auto Ho = out_k_ho_wo_n_global_desc.GetLength(I1);
        const auto Wo = out_k_ho_wo_n_global_desc.GetLength(I2);

        const auto Y = wei_c_y_x_k_global_desc.GetLength(I1);
        const auto X = wei_c_y_x_k_global_desc.GetLength(I2);

        const auto ConvStrideH = conv_strides[I0];
        const auto ConvStrideW = conv_strides[I1];

        const auto ConvDilationH = conv_dilations[I0];
        const auto ConvDilationW = conv_dilations[I1];

        const auto InLeftPadH = in_left_pads[I0];
        const auto InLeftPadW = in_left_pads[I1];

        const auto InRightPadH = in_right_pads[I0];
        const auto InRightPadW = in_right_pads[I1];

        // weight tensor
        const auto wei_gemmk_gemmm_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(C * Y * X, K)),
            make_tuple(make_pass_through_transform(C * Y * X), make_pass_through_transform(K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // input tensor
        const auto in_c_hip_wip_n_global_desc = transform_dynamic_tensor_descriptor(
            in_c_hi_wi_n_global_desc,
            make_tuple(make_pass_through_transform(C),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW),
                       make_pass_through_transform(N)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto in_c_y_ho_x_wo_n_global_desc = transform_dynamic_tensor_descriptor(
            in_c_hip_wip_n_global_desc,
            make_tuple(
                make_pass_through_transform(C),
                make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                make_pass_through_transform(N)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        const auto in_gemmk_gemmn_global_desc = transform_dynamic_tensor_descriptor(
            in_c_y_ho_x_wo_n_global_desc,
            make_tuple(make_merge_transform(make_tuple(C, Y, X)),
                       make_merge_transform(make_tuple(Ho, Wo, N))),
            make_tuple(Sequence<0, 1, 3>{}, Sequence<2, 4, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // output tensor
        const auto out_gemmm_gemmn_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, Ho * Wo * N)),
            make_tuple(make_pass_through_transform(K), make_pass_through_transform(Ho * Wo * N)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto GemmM = out_gemmm_gemmn_global_desc.GetLength(I0);
        const auto GemmN = out_gemmm_gemmn_global_desc.GetLength(I1);
        const auto GemmK = wei_gemmk_gemmm_global_desc.GetLength(I0);

        if(!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
             GemmK % GemmKPerBlock == 0))
        {
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        constexpr auto GemmM1 = Number<GemmMPerThread * GemmMLevel0Cluster * GemmMLevel1Cluster>{};
        constexpr auto GemmN1 = Number<GemmNPerThread * GemmNLevel0Cluster * GemmNLevel1Cluster>{};

        const auto GemmM0 = GemmM / GemmM1;
        const auto GemmN0 = GemmN / GemmN1;

        const auto out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc =
            transform_dynamic_tensor_descriptor(
                out_gemmm_gemmn_global_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmM0, GemmM1)),
                           make_unmerge_transform(make_tuple(GemmN0, GemmN1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

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
            Sequence<1, 0>,
            Sequence<1, 0>,
            1,
            GemmBBlockTransferSrcScalarPerVector_GemmN,
            GemmBBlockTransferDstScalarPerVector_GemmN,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<2, 3, 0, 1>,
            1,
            GemmCThreadTransferDstScalarPerVector_GemmM1,
            decltype(a_k_m_global_iterator_hacks),
            decltype(b_k_n_global_iterator_hacks),
            decltype(c_m0_m1_n0_n1_global_tensor_iterator_hacks),
            decltype(a_k_m_global_move_slice_window_iterator_hack),
            decltype(b_k_n_global_move_slice_window_iterator_hack)>;

        const auto GridSize = (GemmM / GemmMPerBlock) * (GemmN / GemmNPerBlock);

        const bool has_main_k_block_loop = (GemmK + GemmKPerBlock) / (2 * GemmKPerBlock) > 1;

        const bool has_double_tail_k_block_loop = (GemmK / GemmKPerBlock) % 2 == 0;

        printf("%s: BlockSize %d, GridSize %d \n", __func__, BlockSize, GridSize);

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

            float perf = (float)(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
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

            float perf = (float)(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
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

            float perf = (float)(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#endif
    }
};

#if 0
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
          index_t GemmCThreadTransferDstScalarPerVector_GemmM1>
struct DriverDynamicConvolutionForwardImplicitGemm_v4r4_chwn_cyxk_khwn_1x1
{
    template <typename... Wei,
              typename... In,
              typename... Out,
              typename ConvStrides,
              typename ConvDilations,
              typename InLeftPads,
              typename InRightPads>
    __host__ void Run(const DynamicTensorDescriptor<Wei...>& wei_c_y_x_k_global_desc,
                      const DynamicTensorDescriptor<In...>& in_c_hi_wi_n_global_desc,
                      const DynamicTensorDescriptor<Out...>& out_k_ho_wo_n_global_desc,
                      const ConvStrides& conv_strides,
                      const ConvDilations& conv_dilations,
                      const InLeftPads& in_left_pads,
                      const InRightPads& in_right_pads,
                      const Float* __restrict__ p_wei_global,
                      const Float* __restrict__ p_in_global,
                      Float* __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        const auto N = in_c_hi_wi_n_global_desc.GetLength(I3);
        const auto C = in_c_hi_wi_n_global_desc.GetLength(I0);
        const auto K = out_k_ho_wo_n_global_desc.GetLength(I0);

        const auto Hi = in_c_hi_wi_n_global_desc.GetLength(I1);
        const auto Wi = in_c_hi_wi_n_global_desc.GetLength(I2);

        const auto Ho = out_k_ho_wo_n_global_desc.GetLength(I1);
        const auto Wo = out_k_ho_wo_n_global_desc.GetLength(I2);

        const auto Y = wei_c_y_x_k_global_desc.GetLength(I1);
        const auto X = wei_c_y_x_k_global_desc.GetLength(I2);

        const auto ConvStrideH = conv_strides[I0];
        const auto ConvStrideW = conv_strides[I1];

        const auto ConvDilationH = conv_dilations[I0];
        const auto ConvDilationW = conv_dilations[I1];

        const auto InLeftPadH = in_left_pads[I0];
        const auto InLeftPadW = in_left_pads[I1];

        const auto InRightPadH = in_right_pads[I0];
        const auto InRightPadW = in_right_pads[I1];

        if(!(Y == 1 && X == 1 && ConvStrideH == 1 && ConvStrideW == 1 && ConvDilationH == 1 &&
             ConvDilationW == 1 && InLeftPadH == 0 && InLeftPadW == 0 && InRightPadH == 0 &&
             InRightPadW == 0))
        {
            throw std::runtime_error("wrong! 1x1, stride 1, no padding");
        }

        // weight tensor
        const auto wei_gemmk_gemmm_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, C)),
            make_tuple(make_pass_through_transform(K), make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        // input tensor
        const auto in_gemmk_gemmn_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N * Ho * Wo, C)),
            make_tuple(make_pass_through_transform(N * Ho * Wo), make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        // output tensor
        const auto out_gemmm_gemmn_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N * Ho * Wo, K)),
            make_tuple(make_pass_through_transform(N * Ho * Wo), make_pass_through_transform(K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        const auto GemmM = out_gemmm_gemmn_global_desc.GetLength(I0);
        const auto GemmN = out_gemmm_gemmn_global_desc.GetLength(I1);
        const auto GemmK = wei_gemmk_gemmm_global_desc.GetLength(I0);

        if(!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
             GemmK % GemmKPerBlock == 0))
        {
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        constexpr auto GemmM1 = Number<GemmMPerThread * GemmMLevel0Cluster * GemmMLevel1Cluster>{};
        constexpr auto GemmN1 = Number<GemmNPerThread * GemmNLevel0Cluster * GemmNLevel1Cluster>{};

        const auto GemmM0 = GemmM / GemmM1;
        const auto GemmN0 = GemmN / GemmN1;

        const auto out_gemmm0_gemmm1_gemmn0_gemmn1_global_desc =
            transform_dynamic_tensor_descriptor(
                out_gemmm_gemmn_global_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmM0, GemmM1)),
                           make_unmerge_transform(make_tuple(GemmN0, GemmN1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        // hack to control index calculation when iterating over a_k_m_global tensor
        constexpr auto a_k_m_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        constexpr auto a_k_m_global_move_slice_window_iterator_hack = Sequence<0, 0, 0>{};

        // hack to control index calculation when iterating over b_k_n_global tensor
        constexpr auto b_k_n_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        constexpr auto b_k_n_global_move_slice_window_iterator_hack = Sequence<0, 0, 0>{};

        // hack to control index calculation when iterating over c_m0_m1_n0_n1_global tensor
        constexpr auto c_m0_m1_n0_n1_global_tensor_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}));

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
            Sequence<1, 0>,
            Sequence<1, 0>,
            1,
            GemmBBlockTransferSrcScalarPerVector_GemmN,
            GemmBBlockTransferDstScalarPerVector_GemmN,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<2, 3, 0, 1>,
            1,
            GemmCThreadTransferDstScalarPerVector_GemmM1,
            decltype(a_k_m_global_iterator_hacks),
            decltype(b_k_n_global_iterator_hacks),
            decltype(c_m0_m1_n0_n1_global_tensor_iterator_hacks),
            decltype(a_k_m_global_move_slice_window_iterator_hack),
            decltype(b_k_n_global_move_slice_window_iterator_hack)>;

        const auto GridSize = (GemmM / GemmMPerBlock) * (GemmN / GemmNPerBlock);

        const bool has_main_k_block_loop = (GemmK + GemmKPerBlock) / (2 * GemmKPerBlock) > 1;

        const bool has_double_tail_k_block_loop = (GemmK / GemmKPerBlock) % 2 == 0;

        printf("%s: BlockSize %d, GridSize %d \n", __func__, BlockSize, GridSize);

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

            float perf = (float)(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
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

            float perf = (float)(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
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

            float perf = (float)(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
#endif
    }
};
#endif

} // namespace ck
#endif
