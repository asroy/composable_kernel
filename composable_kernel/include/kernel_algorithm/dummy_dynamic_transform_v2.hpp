#ifndef CK_DUMMY_DYNAMIC_TRANSFORM_V2_HPP
#define CK_DUMMY_DYNAMIC_TRANSFORM_V2_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor_v2.hpp"
#include "dynamic_tensor_descriptor_helper_v2.hpp"

namespace ck {

template <typename... Wei, typename... In, typename... Out>
__host__ __device__ constexpr auto map_convolution_into_gemm_fwd_v4r4(
    const DynamicTensorDescriptor_v2<Wei...>& wei_k_c_y_x_global_desc,
    const DynamicTensorDescriptor_v2<In...>& in_n_c_hi_wi_global_desc,
    const DynamicTensorDescriptor_v2<Out...>& out_n_k_ho_wo_global_desc,
    const MultiIndex<2> conv_strides,
    const MultiIndex<2> conv_dilations,
    const MultiIndex<2> in_left_pads,
    const MultiIndex<2> in_right_pads)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const index_t N = in_n_c_hi_wi_global_desc.GetLength(I0);
    const index_t C = in_n_c_hi_wi_global_desc.GetLength(I1);
    const index_t K = out_n_k_ho_wo_global_desc.GetLength(I1);

    const index_t Y = wei_k_c_y_x_global_desc.GetLength(I2);
    const index_t X = wei_k_c_y_x_global_desc.GetLength(I3);

    const index_t Hi = in_n_c_hi_wi_global_desc.GetLength(I2);
    const index_t Wi = in_n_c_hi_wi_global_desc.GetLength(I3);

    const index_t Ho = out_n_k_ho_wo_global_desc.GetLength(I2);
    const index_t Wo = out_n_k_ho_wo_global_desc.GetLength(I3);

    const index_t ConvStrideH = conv_strides[I0];
    const index_t ConvStrideW = conv_strides[I1];

    const index_t ConvDilationH = conv_dilations[I0];
    const index_t ConvDilationW = conv_dilations[I1];

    const index_t InLeftPadH  = in_left_pads[I0];
    const index_t InLeftPadW  = in_left_pads[I1];
    const index_t InRightPadH = in_right_pads[I0];
    const index_t InRightPadW = in_right_pads[I1];

    // input tensor
    const auto in_n_c_hip_wip_global_desc = transform_dynamic_tensor_descriptor_v2(
        transform_dynamic_tensor_descriptor_v2(
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

    const auto in_n_c_y_ho_x_wo_global_desc = transform_dynamic_tensor_descriptor_v2(
        in_n_c_hip_wip_global_desc,
        make_tuple(
            DynamicPassThrough{N},
            DynamicPassThrough{C},
            DynamicEmbed<2>{make_multi_index(Y, Ho), make_multi_index(ConvDilationH, ConvStrideH)},
            DynamicEmbed<2>{make_multi_index(X, Wo), make_multi_index(ConvDilationW, ConvStrideW)}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

    const auto in_gemmktotal_gemmn_global_desc = transform_dynamic_tensor_descriptor_v2(
        in_n_c_y_ho_x_wo_global_desc,
        make_tuple(DynamicMerge<3>{make_multi_index(C, Y, X)},
                   DynamicMerge<3>{make_multi_index(N, Ho, Wo)}),
        make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    const index_t gemmktotal = in_gemmktotal_gemmn_global_desc.GetLength(I0);
    const index_t gemmn      = in_gemmktotal_gemmn_global_desc.GetLength(I1);

    constexpr index_t GemmKPack = 8;
    const index_t gemmk         = gemmktotal / GemmKPack;

    const auto in_gemmk_gemmn_gemmkpack_global_desc = transform_dynamic_tensor_descriptor_v2(
        in_gemmktotal_gemmn_global_desc,
        make_tuple(DynamicUnMerge<2>{make_multi_index(gemmk, GemmKPack)},
                   DynamicPassThrough{gemmn}),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    static_assert(decltype(in_gemmk_gemmn_gemmkpack_global_desc)::GetNumOfDimension() == 3,
                  "wrong!");

    return make_tuple(in_gemmk_gemmn_gemmkpack_global_desc);
}

#if 0
template <typename... Wei, typename... In, typename... Out>
__host__ __device__ constexpr auto map_convolution_into_gemm_bwd_v4r1(
    const DynamicTensorDescriptor_v2<Wei...>& wei_k_c_y_x_global_desc,
    const DynamicTensorDescriptor_v2<In...>& in_n_c_hi_wi_global_desc,
    const DynamicTensorDescriptor_v2<Out...>& out_n_k_ho_wo_global_desc,
    const MultiIndex<2> conv_strides,
    const MultiIndex<2> conv_dilations,
    const MultiIndex<2> in_left_pads,
    const MultiIndex<2> in_right_pads)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const index_t N = in_n_c_hi_wi_global_desc.GetLength(I0);
    const index_t C = in_n_c_hi_wi_global_desc.GetLength(I1);
    const index_t K = out_n_k_ho_wo_global_desc.GetLength(I1);

    const index_t Y = wei_k_c_y_x_global_desc.GetLength(I2);
    const index_t X = wei_k_c_y_x_global_desc.GetLength(I3);

    const index_t Hi = in_n_c_hi_wi_global_desc.GetLength(I2);
    const index_t Wi = in_n_c_hi_wi_global_desc.GetLength(I3);

    const index_t Ho = out_n_k_ho_wo_global_desc.GetLength(I2);
    const index_t Wo = out_n_k_ho_wo_global_desc.GetLength(I3);

    const index_t ConvStrideH = conv_strides[I0];
    const index_t ConvStrideW = conv_strides[I1];

    const index_t ConvDilationH = conv_dilations[I0];
    const index_t ConvDilationW = conv_dilations[I1];

    const index_t InLeftPadH  = in_left_pads[I0];
    const index_t InLeftPadW  = in_left_pads[I1];
    const index_t InRightPadH = in_right_pads[I0];
    const index_t InRightPadW = in_right_pads[I1];

#if !CK_EXPERIMENTAL_IMPLICIT_GEMM_BACKWARD_DATA_V4R1_OUTPUT_SKIP_OUT_OF_BOUND_CHECK
        constexpr bool out_skip_out_of_bound_check = false;
#else
        constexpr bool out_skip_out_of_bound_check = true;
#endif

        constexpr auto out_n_k_ydot_htilda_xdot_wtilda_global_desc = transform_tensor_descriptor_v2(
            out_n_k_ho_wo_global_desc,
            make_tuple(PassThrough{N},
                       PassThrough{K},
                       Embed<2>{make_multi_index(YDot, HTilda), make_multi_index(-ConvDilationH / GcdStrideDilationH, 1)},
                       Embed<2>{make_multi_index(XDot, WTilda), make_multi_index(-ConvDilationW / GcdStrideDilationW, 1)}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto out_n_k_ydot_htildaslice_xdot_wtildaslice_global_desc =
            transform_tensor_descriptor_v2(
                out_n_k_ydot_htilda_xdot_wtilda_global_desc,
                make_tuple(PassThrough{N},
                           PassThrough{K},
                           PassThrough{YDot},
                           PassThrough{XDot},
                           Slice<Sequence<HTilda, WTilda>,
                                 Sequence<iHTildaLeft, iWTildaLeft>,
                                 Sequence<iHTildaRight, iWTildaRight>>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}, Sequence<3, 5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}, Sequence<3, 5>{}));

        constexpr auto out_n_k_ydotslice_htildaslice_xdotslice_wtildaslice_global_desc =
            transform_tensor_descriptor(
                out_n_k_ydot_htildaslice_xdot_wtildaslice_global_desc,
                make_tuple(
                    PassThrough<N>{},
                    PassThrough<K>{},
                    PassThrough<HTildaSlice>{},
                    PassThrough<WTildaSlice>{},
                    Slice<Sequence<YDot, XDot>, Sequence<0, 0>, Sequence<YDotSlice, XDotSlice>>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<5>{}, Sequence<2, 4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<5>{}, Sequence<2, 4>{}));

        constexpr auto out_gemmk_gemmn_global_desc = transform_tensor_descriptor(
            out_n_k_ydotslice_htildaslice_xdotslice_wtildaslice_global_desc,
            make_tuple(Merge<Sequence<K, YDotSlice, XDotSlice>>{},
                       Merge<Sequence<N, HTildaSlice, WTildaSlice>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

    return make_tuple(in_gemmk_gemmn_gemmkpack_global_desc);
}
#endif

template <index_t BlockSize>
struct DummyDynamicTransform_v2_1
{
    template <typename WeiDesc, typename InDesc, typename OutDesc>
    __device__ void Run_1(index_t* const __restrict__ p_wei_global,
                          float* const __restrict__ p_in_global,
                          float* const __restrict__ p_out_global,
                          const WeiDesc wei_k_c_y_x_global_desc,
                          const InDesc in_n_c_hi_wi_global_desc,
                          const OutDesc out_n_k_ho_wo_global_desc,
                          const MultiIndex<2> conv_strides,
                          const MultiIndex<2> conv_dilations,
                          const MultiIndex<2> in_left_pads,
                          const MultiIndex<2> in_right_pads) const
    {
        const auto transformed_tensor_descs =
            map_convolution_into_gemm(move(wei_k_c_y_x_global_desc),
                                      move(in_n_c_hi_wi_global_desc),
                                      move(out_n_k_ho_wo_global_desc),
                                      conv_strides,
                                      conv_dilations,
                                      in_left_pads,
                                      in_right_pads);

        const auto in_gemmk_gemmn_global_desc = transformed_tensor_descs.At(Number<0>{});

        MultiIndex<2> idx;

        // initialize idx
        static_for<0, 2, 1>{}([&](auto i) { idx(i) = p_wei_global[get_thread_local_1d_id() + i]; });

        const index_t niter = p_wei_global[10];

        auto in_gemmk_gemmn_coord =
            make_dynamic_tensor_coordinate_v2(in_gemmk_gemmn_global_desc, idx);

        const auto in_gemmk_gemmn_coord_step = make_dynamic_tensor_coordinate_step_v2(
            in_gemmk_gemmn_global_desc, make_multi_index(1, 0));

        for(index_t iter = 0; iter < niter; ++iter)
        {
            move_dynamic_tensor_coordinate_v2(
                in_gemmk_gemmn_global_desc, in_gemmk_gemmn_coord, in_gemmk_gemmn_coord_step);

            // write
            float value = 1;

            transfer_data<float,
                          1,
                          AddressSpace::Vgpr,
                          AddressSpace::Global,
                          InMemoryDataOperation::Set,
                          1,
                          1>(&value,
                             0,
                             true,
                             1,
                             p_out_global,
                             in_gemmk_gemmn_coord.GetOffset(),
#if 1
                             coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                 in_gemmk_gemmn_global_desc, in_gemmk_gemmn_coord),
#else
                             true,
#endif
                             in_gemmk_gemmn_global_desc.GetElementSpaceSize());
        }
    }

    template <typename WeiDesc, typename InDesc, typename OutDesc>
    __device__ void Run_2(index_t* const __restrict__ p_wei_global,
                          float* const __restrict__ p_in_global,
                          float* const __restrict__ p_out_global,
                          const WeiDesc wei_k_c_y_x_global_desc,
                          const InDesc in_n_c_hi_wi_global_desc,
                          const OutDesc out_n_k_ho_wo_global_desc,
                          const MultiIndex<2> conv_strides,
                          const MultiIndex<2> conv_dilations,
                          const MultiIndex<2> in_left_pads,
                          const MultiIndex<2> in_right_pads) const
    {
        constexpr auto i0 = Number<0>{};
        constexpr auto i1 = Number<1>{};
        constexpr auto i2 = Number<2>{};
        constexpr auto i3 = Number<3>{};

        const index_t N = in_n_c_hi_wi_global_desc.GetLength(i0);
        const index_t C = in_n_c_hi_wi_global_desc.GetLength(i1);
        const index_t K = out_n_k_ho_wo_global_desc.GetLength(i1);

        const index_t Y = wei_k_c_y_x_global_desc.GetLength(i2);
        const index_t X = wei_k_c_y_x_global_desc.GetLength(i3);

        const index_t Hi = in_n_c_hi_wi_global_desc.GetLength(i2);
        const index_t Wi = in_n_c_hi_wi_global_desc.GetLength(i3);

        const index_t Ho = out_n_k_ho_wo_global_desc.GetLength(i2);
        const index_t Wo = out_n_k_ho_wo_global_desc.GetLength(i3);

        const index_t ConvStrideH = conv_strides[i0];
        const index_t ConvStrideW = conv_strides[i1];

        const index_t ConvDilationH = conv_dilations[i0];
        const index_t ConvDilationW = conv_dilations[i1];

        const index_t InLeftPadH  = in_left_pads[i0];
        const index_t InLeftPadW  = in_left_pads[i1];
        const index_t InRightPadH = in_right_pads[i0];
        const index_t InRightPadW = in_right_pads[i1];

#if 0
        const auto in_n_c_hip_wip_global_desc = transform_dynamic_tensor_descriptor_v2(
            move(in_n_c_hi_wi_global_desc),
            make_tuple(DynamicPassThrough{N},
                       DynamicPassThrough{C},
                       DynamicPassThrough{Hi},
                       DynamicPassThrough{Wi}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
#elif 0
        const auto in_n_c_hip_wip_global_desc = transform_dynamic_tensor_descriptor_v2(
            move(in_n_c_hi_wi_global_desc),
            make_tuple(DynamicPassThrough{N},
                       DynamicPassThrough{C},
                       DynamicLeftPad{Hi, InLeftPadH},
                       DynamicLeftPad{Wi, InLeftPadW}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
#else
        const auto in_n_c_hip_wip_global_desc = transform_dynamic_tensor_descriptor_v2(
            transform_dynamic_tensor_descriptor_v2(
                move(in_n_c_hi_wi_global_desc),
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
#endif

        MultiIndex<4> idx;

        // initialize idx
        static_for<0, 4, 1>{}([&](auto i) { idx(i) = p_wei_global[get_thread_local_1d_id() + i]; });

#if 1
        const index_t niter = p_wei_global[10];

        auto in_coord = make_dynamic_tensor_coordinate_v2(in_n_c_hip_wip_global_desc, idx);

        const auto in_coord_step = make_dynamic_tensor_coordinate_step_v2(
            in_n_c_hip_wip_global_desc, make_multi_index(1, 0, 0, 0));

        for(index_t iter = 0; iter < niter; ++iter)
        {
            move_dynamic_tensor_coordinate_v2(in_n_c_hip_wip_global_desc, in_coord, in_coord_step);

            // write
            float value = 1;

            transfer_data<float,
                          1,
                          AddressSpace::Vgpr,
                          AddressSpace::Global,
                          InMemoryDataOperation::Set,
                          1,
                          1>(&value,
                             0,
                             true,
                             1,
                             p_out_global,
                             in_coord.GetOffset(),
                             coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                 in_n_c_hip_wip_global_desc, in_coord),
                             in_n_c_hip_wip_global_desc.GetElementSpaceSize());
        }
#else
        // write
        // auto in_coord = make_dynamic_tensor_coordinate_v2(in_n_c_hi_wi_global_desc, idx);

        p_out_global[in_n_c_hip_wip_global_desc.CalculateOffset(idx)] = 1;
#endif
    }

    template <typename WeiDesc, typename InDesc, typename OutDesc>
    __device__ void Run(index_t* const __restrict__ p_wei_global,
                        float* const __restrict__ p_in_global,
                        float* const __restrict__ p_out_global,
                        const WeiDesc wei_k_c_y_x_global_desc,
                        const InDesc in_n_c_hi_wi_global_desc,
                        const OutDesc out_n_k_ho_wo_global_desc,
                        const MultiIndex<2> conv_strides,
                        const MultiIndex<2> conv_dilations,
                        const MultiIndex<2> in_left_pads,
                        const MultiIndex<2> in_right_pads) const
    {
        Run_2(p_wei_global,
              p_in_global,
              p_out_global,
              wei_k_c_y_x_global_desc,
              in_n_c_hi_wi_global_desc,
              out_n_k_ho_wo_global_desc,
              conv_strides,
              conv_dilations,
              in_left_pads,
              in_right_pads);
    }
};

template <index_t BlockSize>
struct DummyDynamicTransform_v2_fwd_v4r4
{
    template <typename TransformInDesc>
    __device__ void Run(index_t* const __restrict__ p_wei_global,
                        float* const __restrict__ p_in_global,
                        float* const __restrict__ p_out_global,
                        const TransformInDesc in_gemmk_gemmn_gemmkpack_global_desc) const
    {
        MultiIndex<3> idx;

        // initialize idx
        static_for<0, 3, 1>{}([&](auto i) { idx(i) = p_wei_global[get_thread_local_1d_id() + i]; });

        const index_t niter = p_wei_global[10];

        auto in_gemmk_gemmn_gemmkpack_coord =
            make_dynamic_tensor_coordinate_v2(in_gemmk_gemmn_gemmkpack_global_desc, idx);

        const auto in_gemmk_gemmn_gemmkpack_coord_step_0_0_1 =
            make_dynamic_tensor_coordinate_step_v2(in_gemmk_gemmn_gemmkpack_global_desc,
                                                   make_multi_index(0, 0, 1));

        const auto in_gemmk_gemmn_gemmkpack_coord_step_0_1_0 =
            make_dynamic_tensor_coordinate_step_v2(in_gemmk_gemmn_gemmkpack_global_desc,
                                                   make_multi_index(0, 1, 0));

        const auto in_gemmk_gemmn_gemmkpack_coord_step_1_0_0 =
            make_dynamic_tensor_coordinate_step_v2(in_gemmk_gemmn_gemmkpack_global_desc,
                                                   make_multi_index(1, 0, 0));

        // move (0, 0, 1)
        for(index_t iter = 0; iter < niter; ++iter)
        {
            move_dynamic_tensor_coordinate_v2(in_gemmk_gemmn_gemmkpack_global_desc,
                                              in_gemmk_gemmn_gemmkpack_coord,
                                              in_gemmk_gemmn_gemmkpack_coord_step_0_0_1);

            // write
            float value = 1;

            transfer_data<float,
                          1,
                          AddressSpace::Vgpr,
                          AddressSpace::Global,
                          InMemoryDataOperation::Set,
                          1,
                          1>(
                &value,
                0,
                true,
                1,
                p_out_global,
                in_gemmk_gemmn_gemmkpack_coord.GetOffset(),
#if 1
                coordinate_has_valid_offset_assuming_visible_index_is_valid(
                    in_gemmk_gemmn_gemmkpack_global_desc, in_gemmk_gemmn_gemmkpack_coord),
#else
                true,
#endif
                in_gemmk_gemmn_gemmkpack_global_desc.GetElementSpaceSize());
        }

        // move (0, 1, 0)
        for(index_t iter = 0; iter < niter; ++iter)
        {
            move_dynamic_tensor_coordinate_v2(in_gemmk_gemmn_gemmkpack_global_desc,
                                              in_gemmk_gemmn_gemmkpack_coord,
                                              in_gemmk_gemmn_gemmkpack_coord_step_0_1_0);

            // write
            float value = 1;

            transfer_data<float,
                          1,
                          AddressSpace::Vgpr,
                          AddressSpace::Global,
                          InMemoryDataOperation::Set,
                          1,
                          1>(
                &value,
                0,
                true,
                1,
                p_out_global,
                in_gemmk_gemmn_gemmkpack_coord.GetOffset(),
#if 1
                coordinate_has_valid_offset_assuming_visible_index_is_valid(
                    in_gemmk_gemmn_gemmkpack_global_desc, in_gemmk_gemmn_gemmkpack_coord),
#else
                true,
#endif
                in_gemmk_gemmn_gemmkpack_global_desc.GetElementSpaceSize());
        }

        // move (1, 0, 0)
        for(index_t iter = 0; iter < niter; ++iter)
        {
            move_dynamic_tensor_coordinate_v2(in_gemmk_gemmn_gemmkpack_global_desc,
                                              in_gemmk_gemmn_gemmkpack_coord,
                                              in_gemmk_gemmn_gemmkpack_coord_step_1_0_0);

            // write
            float value = 1;

            transfer_data<float,
                          1,
                          AddressSpace::Vgpr,
                          AddressSpace::Global,
                          InMemoryDataOperation::Set,
                          1,
                          1>(
                &value,
                0,
                true,
                1,
                p_out_global,
                in_gemmk_gemmn_gemmkpack_coord.GetOffset(),
#if 1
                coordinate_has_valid_offset_assuming_visible_index_is_valid(
                    in_gemmk_gemmn_gemmkpack_global_desc, in_gemmk_gemmn_gemmkpack_coord),
#else
                true,
#endif
                in_gemmk_gemmn_gemmkpack_global_desc.GetElementSpaceSize());
        }
    }
};

} // namespace ck
#endif
