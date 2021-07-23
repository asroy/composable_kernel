#ifndef CK_DRIVER_STATIC_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V5R1_NCHW_KCYX_NKHW_OUTPAD_HPP
#define CK_DRIVER_STATIC_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V5R1_NCHW_KCYX_NKHW_OUTPAD_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_static_gemm_v2.hpp"
#include "gridwise_operation_wrapper.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          index_t KPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t EPerBlock,
          index_t KPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E_K,
          typename ABlockTransferThreadClusterLengths_E_K,
          index_t ABlockTransferSrcScalarPerVector_E,
          index_t ABlockTransferDstScalarPerVector_K,
          index_t BThreadTransferSrcScalarPerVector_W,
          index_t CThreadTransferDstScalarPerVector_W>
struct DriverStaticConvolutionForwardImplicitGemm_v5r1_nchw_kcyx_nkhw_outpad
{
    template <typename... Wei,
              typename... In,
              typename... Out,
              typename ConvStrides,
              typename ConvDilations,
              typename InLeftPads,
              typename InRightPads>
    __host__ void Run(const DynamicTensorDescriptor<Wei...>& wei_k_c_y_x_global_desc,
                      const DynamicTensorDescriptor<In...>& in_n_c_hi_wi_global_desc,
                      const DynamicTensorDescriptor<Out...>& out_n_k0_ho_wo_k1_global_desc,
                      const ConvStrides& conv_strides,
                      const ConvDilations& conv_dilations,
                      const InLeftPads& in_left_pads,
                      const InRightPads& in_right_pads_,
                      const FloatAB* __restrict__ p_wei_global,
                      const FloatAB* __restrict__ p_in_global,
                      FloatC* __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        const auto N_ = in_n_c_hi_wi_global_desc.GetLength(I0);
        const auto C_ = in_n_c_hi_wi_global_desc.GetLength(I1);

        const auto Hi_ = in_n_c_hi_wi_global_desc.GetLength(I2);
        const auto Wi_ = in_n_c_hi_wi_global_desc.GetLength(I3);

        const auto K0_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I1);
        const auto Ho_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I2);
        const auto Wo_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I3);
        const auto K1_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I4);

        const auto K_ = wei_k_c_y_x_global_desc.GetLength(I0);
        const auto Y_ = wei_k_c_y_x_global_desc.GetLength(I2);
        const auto X_ = wei_k_c_y_x_global_desc.GetLength(I3);

        constexpr auto N  = Number<N_>{};
        constexpr auto C  = Number<C_>{};
        constexpr auto K0 = Number<K0_>{};
        constexpr auto K1 = Number<K1_>{};

        constexpr auto Hi = Number<Hi_>{};
        constexpr auto Wi = Number<Wi_>{};

        constexpr auto Ho = Number<Ho_>{};
        constexpr auto Wo = Number<Wo_>{};

        constexpr auto K = Number<K_>{};
        constexpr auto Y = Number<Y_>{};
        constexpr auto X = Number<X_>{};

        const auto ConvStrideH_ = conv_strides[I0];
        const auto ConvStrideW_ = conv_strides[I1];

        const auto ConvDilationH_ = conv_dilations[I0];
        const auto ConvDilationW_ = conv_dilations[I1];

        constexpr auto ConvStrideH = Number<ConvStrideH_>{};
        constexpr auto ConvStrideW = Number<ConvStrideW_>{};

        constexpr auto ConvDilationH = Number<ConvDilationH_>{};
        constexpr auto ConvDilationW = Number<ConvDilationW_>{};

        constexpr auto Hop = Number<(Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock>{};
        constexpr auto Wop = Number<(Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock>{};

        constexpr auto OutRightPadH = Hop - Ho;
        constexpr auto OutRightPadW = Wop - Wo;

        const auto InLeftPadH_ = in_left_pads[I0];
        const auto InLeftPadW_ = in_left_pads[I1];

        constexpr auto InLeftPadH = Number<InLeftPadH_>{};
        constexpr auto InLeftPadW = Number<InLeftPadW_>{};

        constexpr auto in_right_pads = InRightPads{};

        const auto InRightPadH_ = in_right_pads[I0] + OutRightPadH * ConvStrideH;
        const auto InRightPadW_ = in_right_pads[I1] + OutRightPadW * ConvStrideW;

        constexpr auto InRightPadH = Number<InRightPadH_>{};
        constexpr auto InRightPadW = Number<InRightPadW_>{};

        std::cerr << "OutRightPadH = " << OutRightPadH << " OutRightPadW = " << OutRightPadW
                  << std::endl;
        std::cerr << "InRightPadH = " << InRightPadH << " InRightPadW = " << InRightPadW
                  << std::endl;

        // weight tensor
        const auto wei_e_k_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, C * Y * X)),
            make_tuple(make_pass_through_transform(K), make_pass_through_transform(C * Y * X)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        static_assert(wei_e_k_global_desc.IsKnownAtCompileTime(),
                      "wrong! wei_e_k_global_desc need to known at compile-time");

        // input tensor
        const auto in_n_c_hip_wip_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(make_pass_through_transform(N),
                       make_pass_through_transform(C),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        static_assert(in_n_c_hip_wip_global_desc.IsKnownAtCompileTime(),
                      "wrong! in_n_c_hip_wip_global_desc need to known at compile-time");

        const auto in_n_c_y_ho_x_wo_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(
                make_pass_through_transform(N),
                make_pass_through_transform(C),
                make_embed_transform(make_tuple(Y, Hop), make_tuple(ConvDilationH, ConvStrideH)),
                make_embed_transform(make_tuple(X, Wop), make_tuple(ConvDilationW, ConvStrideW))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        static_assert(in_n_c_y_ho_x_wo_global_desc.IsKnownAtCompileTime(),
                      "wrong! in_n_c_y_ho_x_wo_global_desc need to known at compile-time");

        const auto in_e_n_ho_wo_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_y_ho_x_wo_global_desc,
            make_tuple(make_merge_transform(make_tuple(C, Y, X)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Hop),
                       make_pass_through_transform(Wop)),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0>{}, Sequence<3>{}, Sequence<5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        static_assert(in_e_n_ho_wo_global_desc.IsKnownAtCompileTime(),
                      "wrong! in_e_n_ho_wo_global_desc need to known at compile-time");

        // output tensor
        const auto out_k_n_hop_wop_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, K0, Ho, Wo, K1)),
            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_right_pad_transform(Ho, OutRightPadH),
                       make_right_pad_transform(Wo, OutRightPadW)),
            make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        static_assert(out_k_n_hop_wop_global_desc.IsKnownAtCompileTime(),
                      "wrong! out_k_n_hop_wop_global_desc need to known at compile-time");

        const auto E = C * Y * X;

        std::cerr << "Hop = " << Hop << " Wop = " << Wop << std::endl;

        if(!((K % KPerBlock) == 0 && (Hop % HoPerBlock) == 0 && (Wop % WoPerBlock) == 0 &&
             (E % EPerBlock) == 0))
        {
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        // hack to control index calculation when iterating over a_k_m_global tensor
        constexpr auto a_e_k_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        constexpr auto a_e_k_global_move_slice_window_iterator_hack = Sequence<0, 0, 0>{};

        constexpr auto b_e_n_ho_wo_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}));

        constexpr auto b_e_n_ho_wo_global_move_slice_window_iterator_hack =
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{};

        // hack to control index calculation when iterating over c_m0_m1_n0_n1_global tensor
        // hack for NKHW format
        constexpr auto c_k_n_ho_wo_global_tensor_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 1, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 2, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}));

        // GEMM
        using gridwise_gemm = GridwiseStaticGemm_km_kn_mn_v3<
            BlockSize,
            FloatAB,
            FloatAcc,
            FloatC,
            InMemoryDataOperation::Set,
            decltype(wei_e_k_global_desc),
            decltype(in_e_n_ho_wo_global_desc),
            decltype(out_k_n_hop_wop_global_desc),
            KPerBlock,
            HoPerBlock,
            WoPerBlock,
            EPerBlock,
            KPerThread,
            HoPerThread,
            WoPerThread,
            EPerThread,
            ABlockTransferThreadSliceLengths_E_K,
            ABlockTransferThreadClusterLengths_E_K,
            Sequence<1, 0>,
            Sequence<1, 0>,
            0,
            ABlockTransferSrcScalarPerVector_E,
            ABlockTransferDstScalarPerVector_K,
            false, // don't move back src coordinate after threadwise copy
            Sequence<0, 2, 3, 1>,
            3,
            BThreadTransferSrcScalarPerVector_W,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<0, 2, 3, 1>,
            0,
            CThreadTransferDstScalarPerVector_W,
            decltype(a_e_k_global_iterator_hacks),
            decltype(b_e_n_ho_wo_global_iterator_hacks),
            decltype(c_k_n_ho_wo_global_tensor_iterator_hacks),
            decltype(a_e_k_global_move_slice_window_iterator_hack),
            decltype(b_e_n_ho_wo_global_move_slice_window_iterator_hack)>;

        const auto GridSize = (K / KPerBlock) * (Hop / HoPerBlock) * (Wop / WoPerBlock) * N;

        constexpr bool has_main_k_block_loop = (E + EPerBlock) / (2 * EPerBlock) > 1;

        constexpr bool has_double_tail_k_block_loop = (E / EPerBlock) % 2 == 0;

        index_t nrepeat = 100;

        std::cout << "conv_v5r1__NCHWc" << K1 << "_n" << N << "c" << C << "h" << Hi << "w" << Wi
                  << "-k" << K << "c" << C << "y" << Y << "x" << X << "-u" << conv_strides[I0]
                  << "v" << conv_strides[I1] << "l" << conv_dilations[I0] << "j"
                  << conv_dilations[I1] << "q" << in_left_pads[I0] << "p" << in_right_pads[I0]
                  << std::endl;

        std::cout << "GridSize = " << GridSize << " BlockSize = " << BlockSize << std::endl;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();
            std::cout << "has_main_k_block_loop: " << has_main_k_block_loop
                      << " has_double_tail_k_block_loop: " << has_double_tail_k_block_loop
                      << std::endl;

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if constexpr(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if constexpr(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if constexpr(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, true>{});
                }
                else
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k0_ho_wo_k1_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }
};

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          index_t KPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t EPerBlock,
          index_t KPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E_K,
          typename ABlockTransferThreadClusterLengths_E_K,
          index_t ABlockTransferSrcScalarPerVector_E,
          index_t ABlockTransferDstScalarPerVector_K,
          index_t BThreadTransferSrcScalarPerVector_W,
          index_t CThreadTransferDstScalarPerVector_W>
struct DriverStaticConvolutionForwardImplicitGemm_v5r1_nchw_kcyx_nkhw_outpad_1x1
{
    template <typename... Wei,
              typename... In,
              typename... Out,
              typename ConvStrides,
              typename ConvDilations,
              typename InLeftPads,
              typename InRightPads>
    __host__ void Run(const DynamicTensorDescriptor<Wei...>& wei_k_c_y_x_global_desc,
                      const DynamicTensorDescriptor<In...>& in_n_c_hi_wi_global_desc,
                      const DynamicTensorDescriptor<Out...>& out_n_k0_ho_wo_k1_global_desc,
                      const ConvStrides& conv_strides,
                      const ConvDilations& conv_dilations,
                      const InLeftPads& in_left_pads,
                      const InRightPads& in_right_pads_,
                      const FloatAB* __restrict__ p_wei_global,
                      const FloatAB* __restrict__ p_in_global,
                      FloatC* __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        const auto N_ = in_n_c_hi_wi_global_desc.GetLength(I0);
        const auto C_ = in_n_c_hi_wi_global_desc.GetLength(I1);

        const auto Hi_ = in_n_c_hi_wi_global_desc.GetLength(I2);
        const auto Wi_ = in_n_c_hi_wi_global_desc.GetLength(I3);

        const auto K0_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I1);
        const auto Ho_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I2);
        const auto Wo_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I3);
        const auto K1_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I4);

        const auto K_ = wei_k_c_y_x_global_desc.GetLength(I0);
        const auto Y_ = wei_k_c_y_x_global_desc.GetLength(I2);
        const auto X_ = wei_k_c_y_x_global_desc.GetLength(I3);

        constexpr auto N  = Number<N_>{};
        constexpr auto C  = Number<C_>{};
        constexpr auto K0 = Number<K0_>{};
        constexpr auto K1 = Number<K1_>{};

        constexpr auto Hi = Number<Hi_>{};
        constexpr auto Wi = Number<Wi_>{};

        constexpr auto Ho = Number<Ho_>{};
        constexpr auto Wo = Number<Wo_>{};

        constexpr auto K = Number<K_>{};
        constexpr auto Y = Number<Y_>{};
        constexpr auto X = Number<X_>{};

        const auto ConvStrideH_ = conv_strides[I0];
        const auto ConvStrideW_ = conv_strides[I1];

        const auto ConvDilationH_ = conv_dilations[I0];
        const auto ConvDilationW_ = conv_dilations[I1];

        constexpr auto ConvStrideH = Number<ConvStrideH_>{};
        constexpr auto ConvStrideW = Number<ConvStrideW_>{};

        constexpr auto ConvDilationH = Number<ConvDilationH_>{};
        constexpr auto ConvDilationW = Number<ConvDilationW_>{};

        constexpr auto Hop = Number<(Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock>{};
        constexpr auto Wop = Number<(Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock>{};

        constexpr auto OutRightPadH = Hop - Ho;
        constexpr auto OutRightPadW = Wop - Wo;

        const auto InLeftPadH_ = in_left_pads[I0];
        const auto InLeftPadW_ = in_left_pads[I1];

        constexpr auto InLeftPadH = Number<InLeftPadH_>{};
        constexpr auto InLeftPadW = Number<InLeftPadW_>{};

        static_assert(InLeftPadH == 0 and InLeftPadW == 0, "");

        constexpr auto in_right_pads = InRightPads{};

        const auto InRightPadH_ = in_right_pads[I0] + OutRightPadH * ConvStrideH;
        const auto InRightPadW_ = in_right_pads[I1] + OutRightPadW * ConvStrideW;

        constexpr auto InRightPadH = Number<InRightPadH_>{};
        constexpr auto InRightPadW = Number<InRightPadW_>{};

        std::cerr << "OutRightPadH = " << OutRightPadH << " OutRightPadW = " << OutRightPadW
                  << std::endl;
        std::cerr << "InRightPadH = " << InRightPadH << " InRightPadW = " << InRightPadW
                  << std::endl;

        // weight tensor
        const auto wei_e_k_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, C)),
            make_tuple(make_pass_through_transform(K), make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        static_assert(wei_e_k_global_desc.IsKnownAtCompileTime(),
                      "wrong! wei_e_k_global_desc need to known at compile-time");

        // input tensor
        const auto in_e_n_ho_wo_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(make_pass_through_transform(C),
                       make_pass_through_transform(N),
                       make_right_pad_transform(Hi, InRightPadH),
                       make_right_pad_transform(Wi, InRightPadW)),
            make_tuple(Sequence<1>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        static_assert(in_e_n_ho_wo_global_desc.IsKnownAtCompileTime(),
                      "wrong! in_e_n_ho_wo_global_desc need to known at compile-time");

        // output tensor
        const auto out_k_n_hop_wop_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, K0, Ho, Wo, K1)),
            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_right_pad_transform(Ho, OutRightPadH),
                       make_right_pad_transform(Wo, OutRightPadW)),
            make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        static_assert(out_k_n_hop_wop_global_desc.IsKnownAtCompileTime(),
                      "wrong! out_k_n_hop_wop_global_desc need to known at compile-time");

        const auto E = C;

        std::cerr << "Hop = " << Hop << " Wop = " << Wop << std::endl;

        if(!((K % KPerBlock) == 0 && (Hop % HoPerBlock) == 0 && (Wop % WoPerBlock) == 0 &&
             (E % EPerBlock) == 0))
        {
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        // hack to control index calculation when iterating over a_k_m_global tensor
        constexpr auto a_e_k_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        constexpr auto a_e_k_global_move_slice_window_iterator_hack = Sequence<0, 0, 0>{};

        constexpr auto b_e_n_ho_wo_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}));

        constexpr auto b_e_n_ho_wo_global_move_slice_window_iterator_hack =
            Sequence<0, 0, 0, 0, 0>{};

        // hack to control index calculation when iterating over c_m0_m1_n0_n1_global tensor
        // hack for NKHW format
        constexpr auto c_k_n_ho_wo_global_tensor_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 1, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 2, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}));

        // GEMM
        using gridwise_gemm = GridwiseStaticGemm_km_kn_mn_v3<
            BlockSize,
            FloatAB,
            FloatAcc,
            FloatC,
            InMemoryDataOperation::Set,
            decltype(wei_e_k_global_desc),
            decltype(in_e_n_ho_wo_global_desc),
            decltype(out_k_n_hop_wop_global_desc),
            KPerBlock,
            HoPerBlock,
            WoPerBlock,
            EPerBlock,
            KPerThread,
            HoPerThread,
            WoPerThread,
            EPerThread,
            ABlockTransferThreadSliceLengths_E_K,
            ABlockTransferThreadClusterLengths_E_K,
            Sequence<1, 0>,
            Sequence<1, 0>,
            0,
            ABlockTransferSrcScalarPerVector_E,
            ABlockTransferDstScalarPerVector_K,
            false, // don't move back src coordinate after threadwise copy
            Sequence<0, 2, 3, 1>,
            3,
            BThreadTransferSrcScalarPerVector_W,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<0, 2, 3, 1>,
            0,
            CThreadTransferDstScalarPerVector_W,
            decltype(a_e_k_global_iterator_hacks),
            decltype(b_e_n_ho_wo_global_iterator_hacks),
            decltype(c_k_n_ho_wo_global_tensor_iterator_hacks),
            decltype(a_e_k_global_move_slice_window_iterator_hack),
            decltype(b_e_n_ho_wo_global_move_slice_window_iterator_hack)>;

        const auto GridSize = (K / KPerBlock) * (Hop / HoPerBlock) * (Wop / WoPerBlock) * N;

        constexpr bool has_main_k_block_loop = (E + EPerBlock) / (2 * EPerBlock) > 1;

        constexpr bool has_double_tail_k_block_loop = (E / EPerBlock) % 2 == 0;

        index_t nrepeat = 100;

        std::cout << "conv_v5r1__NCHWc" << K1 << "_n" << N << "c" << C << "h" << Hi << "w" << Wi
                  << "-k" << K << "c" << C << "y" << Y << "x" << X << "-u" << conv_strides[I0]
                  << "v" << conv_strides[I1] << "l" << conv_dilations[I0] << "j"
                  << conv_dilations[I1] << "q" << in_left_pads[I0] << "p" << in_right_pads[I0]
                  << std::endl;

        std::cout << "GridSize = " << GridSize << " BlockSize = " << BlockSize << std::endl;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();
            std::cout << "has_main_k_block_loop: " << has_main_k_block_loop
                      << " has_double_tail_k_block_loop: " << has_double_tail_k_block_loop
                      << std::endl;

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if constexpr(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if constexpr(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if constexpr(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, true>{});
                }
                else
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k0_ho_wo_k1_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }
};

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          index_t KPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t EPerBlock,
          index_t KPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E_K,
          typename ABlockTransferThreadClusterLengths_E_K,
          index_t ABlockTransferSrcScalarPerVector_E,
          index_t ABlockTransferDstScalarPerVector_K,
          index_t BThreadTransferSrcScalarPerVector_W,
          index_t CThreadTransferDstScalarPerVector_W>
struct DriverStaticConvolutionForwardImplicitGemm_v5r1_nchw_kcyx_nkhw_1x1
{
    template <typename... Wei,
              typename... In,
              typename... Out,
              typename ConvStrides,
              typename ConvDilations,
              typename InLeftPads,
              typename InRightPads>
    __host__ void Run(const DynamicTensorDescriptor<Wei...>& wei_k_c_y_x_global_desc,
                      const DynamicTensorDescriptor<In...>& in_n_c_hi_wi_global_desc,
                      const DynamicTensorDescriptor<Out...>& out_n_k0_ho_wo_k1_global_desc,
                      const ConvStrides& conv_strides,
                      const ConvDilations& conv_dilations,
                      const InLeftPads& in_left_pads,
                      const InRightPads& in_right_pads_,
                      const FloatAB* __restrict__ p_wei_global,
                      const FloatAB* __restrict__ p_in_global,
                      FloatC* __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        const auto N_ = in_n_c_hi_wi_global_desc.GetLength(I0);
        const auto C_ = in_n_c_hi_wi_global_desc.GetLength(I1);

        const auto Hi_ = in_n_c_hi_wi_global_desc.GetLength(I2);
        const auto Wi_ = in_n_c_hi_wi_global_desc.GetLength(I3);

        const auto K0_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I1);
        const auto Ho_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I2);
        const auto Wo_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I3);
        const auto K1_ = out_n_k0_ho_wo_k1_global_desc.GetLength(I4);

        const auto K_ = wei_k_c_y_x_global_desc.GetLength(I0);
        const auto Y_ = wei_k_c_y_x_global_desc.GetLength(I2);
        const auto X_ = wei_k_c_y_x_global_desc.GetLength(I3);

        constexpr auto N  = Number<N_>{};
        constexpr auto C  = Number<C_>{};
        constexpr auto K0 = Number<K0_>{};
        constexpr auto K1 = Number<K1_>{};

        constexpr auto Hi = Number<Hi_>{};
        constexpr auto Wi = Number<Wi_>{};

        constexpr auto Ho = Number<Ho_>{};
        constexpr auto Wo = Number<Wo_>{};

        constexpr auto K = Number<K_>{};
        constexpr auto Y = Number<Y_>{};
        constexpr auto X = Number<X_>{};

        const auto ConvStrideH_ = conv_strides[I0];
        const auto ConvStrideW_ = conv_strides[I1];

        const auto ConvDilationH_ = conv_dilations[I0];
        const auto ConvDilationW_ = conv_dilations[I1];

        constexpr auto ConvStrideH = Number<ConvStrideH_>{};
        constexpr auto ConvStrideW = Number<ConvStrideW_>{};

        constexpr auto ConvDilationH = Number<ConvDilationH_>{};
        constexpr auto ConvDilationW = Number<ConvDilationW_>{};

        constexpr auto Hop = Number<(Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock>{};
        constexpr auto Wop = Number<(Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock>{};

        constexpr auto OutRightPadH = Hop - Ho;
        constexpr auto OutRightPadW = Wop - Wo;

        const auto InLeftPadH_ = in_left_pads[I0];
        const auto InLeftPadW_ = in_left_pads[I1];

        constexpr auto InLeftPadH = Number<InLeftPadH_>{};
        constexpr auto InLeftPadW = Number<InLeftPadW_>{};

        static_assert(InLeftPadH == 0 and InLeftPadW == 0, "");

        constexpr auto in_right_pads = InRightPads{};

        const auto InRightPadH_ = in_right_pads[I0] + OutRightPadH * ConvStrideH;
        const auto InRightPadW_ = in_right_pads[I1] + OutRightPadW * ConvStrideW;

        constexpr auto InRightPadH = Number<InRightPadH_>{};
        constexpr auto InRightPadW = Number<InRightPadW_>{};

        static_assert(OutRightPadW == 0 and OutRightPadH == 0, "");

        // weight tensor
        const auto wei_e_k_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, C)),
            make_tuple(make_pass_through_transform(K), make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        static_assert(wei_e_k_global_desc.IsKnownAtCompileTime(),
                      "wrong! wei_e_k_global_desc need to known at compile-time");

        // input tensor
        const auto in_e_n_ho_wo_global_desc = transform_dynamic_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(make_pass_through_transform(C),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Ho),
                       make_pass_through_transform(Wo)),
            make_tuple(Sequence<1>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        static_assert(in_e_n_ho_wo_global_desc.IsKnownAtCompileTime(),
                      "wrong! in_e_n_ho_wo_global_desc need to known at compile-time");

        // output tensor
        const auto out_k_n_hop_wop_global_desc = transform_dynamic_tensor_descriptor(
            make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, K0, Ho, Wo, K1)),
            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Ho),
                       make_pass_through_transform(Wo)),
            make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        static_assert(out_k_n_hop_wop_global_desc.IsKnownAtCompileTime(),
                      "wrong! out_k_n_hop_wop_global_desc need to known at compile-time");

        const auto E = C;

        std::cerr << "Hop = " << Hop << " Wop = " << Wop << std::endl;

        if(!((K % KPerBlock) == 0 && (Hop % HoPerBlock) == 0 && (Wop % WoPerBlock) == 0 &&
             (E % EPerBlock) == 0))
        {
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        // hack to control index calculation when iterating over a_k_m_global tensor
        constexpr auto a_e_k_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        constexpr auto a_e_k_global_move_slice_window_iterator_hack = Sequence<0, 0, 0>{};

        constexpr auto b_e_n_ho_wo_global_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}));

        constexpr auto b_e_n_ho_wo_global_move_slice_window_iterator_hack =
            Sequence<0, 0, 0, 0, 0>{};

        // hack to control index calculation when iterating over c_m0_m1_n0_n1_global tensor
        // hack for NKHW format
        constexpr auto c_k_n_ho_wo_global_tensor_iterator_hacks =
            make_tuple(make_tuple(Sequence<0, 1, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 2, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}));

        // GEMM
        using gridwise_gemm = GridwiseStaticGemm_km_kn_mn_v3<
            BlockSize,
            FloatAB,
            FloatAcc,
            FloatC,
            InMemoryDataOperation::Set,
            decltype(wei_e_k_global_desc),
            decltype(in_e_n_ho_wo_global_desc),
            decltype(out_k_n_hop_wop_global_desc),
            KPerBlock,
            HoPerBlock,
            WoPerBlock,
            EPerBlock,
            KPerThread,
            HoPerThread,
            WoPerThread,
            EPerThread,
            ABlockTransferThreadSliceLengths_E_K,
            ABlockTransferThreadClusterLengths_E_K,
            Sequence<1, 0>,
            Sequence<1, 0>,
            0,
            ABlockTransferSrcScalarPerVector_E,
            ABlockTransferDstScalarPerVector_K,
            false, // don't move back src coordinate after threadwise copy
            Sequence<0, 2, 3, 1>,
            3,
            BThreadTransferSrcScalarPerVector_W,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<0, 2, 3, 1>,
            0,
            CThreadTransferDstScalarPerVector_W,
            decltype(a_e_k_global_iterator_hacks),
            decltype(b_e_n_ho_wo_global_iterator_hacks),
            decltype(c_k_n_ho_wo_global_tensor_iterator_hacks),
            decltype(a_e_k_global_move_slice_window_iterator_hack),
            decltype(b_e_n_ho_wo_global_move_slice_window_iterator_hack)>;

        const auto GridSize = (K / KPerBlock) * (Hop / HoPerBlock) * (Wop / WoPerBlock) * N;

        constexpr bool has_main_k_block_loop = (E + EPerBlock) / (2 * EPerBlock) > 1;

        constexpr bool has_double_tail_k_block_loop = (E / EPerBlock) % 2 == 0;

        index_t nrepeat = 100;

        std::cout << "conv_v5r1_NCHWc" << K1 << "_n" << N << "c" << C << "h" << Hi << "w" << Wi
                  << "-k" << K << "c" << C << "y" << Y << "x" << X << "-u" << conv_strides[I0]
                  << "v" << conv_strides[I1] << "l" << conv_dilations[I0] << "j"
                  << conv_dilations[I1] << "q" << in_left_pads[I0] << "p" << in_right_pads[I0]
                  << std::endl;

        std::cout << "GridSize = " << GridSize << " BlockSize = " << BlockSize << std::endl;

        for(index_t i = 0; i < 5; ++i)
        {
            std::cout << "Start running " << nrepeat << " times..." << std::endl;

            KernelTimer timer;
            timer.Start();
            std::cout << "has_main_k_block_loop: " << has_main_k_block_loop
                      << " has_double_tail_k_block_loop: " << has_double_tail_k_block_loop
                      << std::endl;

            for(index_t j = 0; j < nrepeat; ++j)
            {
                if constexpr(has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, true>{});
                }
                else if constexpr(has_main_k_block_loop && !has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, true>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, true>{},
                                  integral_constant<bool, false>{});
                }
                else if constexpr(!has_main_k_block_loop && has_double_tail_k_block_loop)
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, true>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, true>{});
                }
                else
                {
                    const auto kernel = run_gridwise_operation<gridwise_gemm,
                                                               const FloatAB*,
                                                               const FloatAB*,
                                                               FloatC*,
                                                               integral_constant<bool, false>,
                                                               integral_constant<bool, false>>;

                    launch_kernel(kernel,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  p_wei_global,
                                  p_in_global,
                                  p_out_global,
                                  integral_constant<bool, false>{},
                                  integral_constant<bool, false>{});
                }
            }

            timer.End();

            float ave_time = timer.GetElapsedTime() / nrepeat;

            float perf = (float)calculate_convolution_flops(in_n_c_hi_wi_global_desc,
                                                            wei_k_c_y_x_global_desc,
                                                            out_n_k0_ho_wo_k1_global_desc) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }
};
} // namespace ck
#endif
