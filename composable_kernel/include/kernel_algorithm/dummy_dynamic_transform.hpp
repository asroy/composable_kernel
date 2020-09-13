#ifndef CK_DUMMY_DYNAMIC_TRANSFORM_HPP
#define CK_DUMMY_DYNAMIC_TRANSFORM_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_v2.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "dynamic_tensor_descriptor_helper_v2.hpp"
#include "dynamic_tensor_coordinate.hpp"

namespace ck {
template <typename WeiDesc, typename InDesc, typename OutDesc>
__host__ __device__ constexpr auto
map_convolution_into_gemm(const WeiDesc& wei_k_c_y_x_global_desc,
                          const InDesc& in_n_c_hi_wi_global_desc,
                          const OutDesc& out_n_k_ho_wo_global_desc,
                          const Array<index_t, 2> conv_strides,
                          const Array<index_t, 2> conv_dilations,
                          const Array<index_t, 2> in_left_pads,
                          const Array<index_t, 2> in_right_pads)
{
    const index_t N = in_n_c_hi_wi_global_desc.GetLength(0);
    const index_t C = in_n_c_hi_wi_global_desc.GetLength(1);
    const index_t K = out_n_k_ho_wo_global_desc.GetLength(1);

    const index_t Y = wei_k_c_y_x_global_desc.GetLength(2);
    const index_t X = wei_k_c_y_x_global_desc.GetLength(3);

    const index_t Hi = in_n_c_hi_wi_global_desc.GetLength(2);
    const index_t Wi = in_n_c_hi_wi_global_desc.GetLength(3);

    const index_t Ho = out_n_k_ho_wo_global_desc.GetLength(2);
    const index_t Wo = out_n_k_ho_wo_global_desc.GetLength(3);

    const index_t ConvStrideH = conv_strides[0];
    const index_t ConvStrideW = conv_strides[1];

    const index_t ConvDilationH = conv_dilations[0];
    const index_t ConvDilationW = conv_dilations[1];

    const index_t InLeftPadH  = in_left_pads[0];
    const index_t InLeftPadW  = in_left_pads[1];
    const index_t InRightPadH = in_right_pads[0];
    const index_t InRightPadW = in_right_pads[1];

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

    const index_t Hip = in_n_c_hip_wip_global_desc.GetLength(2);
    const index_t Wip = in_n_c_hip_wip_global_desc.GetLength(3);

    const auto in_n_c_y_ho_x_wo_global_desc = transform_dynamic_tensor_descriptor(
        in_n_c_hip_wip_global_desc,
        make_tuple(DynamicPassThrough{N},
                   DynamicPassThrough{C},
                   DynamicEmbed<2>{{Y, Ho}, {ConvDilationH, ConvStrideH, 0}},
                   DynamicEmbed<2>{{X, Wo}, {ConvDilationW, ConvStrideW, 0}}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

    const auto in_gemmk_gemmn_global_desc = transform_dynamic_tensor_descriptor(
        in_n_c_y_ho_x_wo_global_desc,
        make_tuple(DynamicMerge<3>{{C, Y, X}}, DynamicMerge<3>{{N, Ho, Wo}}),
        make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    return make_tuple(in_gemmk_gemmn_global_desc);
}

template <typename WeiDesc, typename InDesc, typename OutDesc>
__host__ __device__ constexpr auto
map_convolution_into_gemm_v2(const WeiDesc& wei_k_c_y_x_global_desc,
                             const InDesc& in_n_c_hi_wi_global_desc,
                             const OutDesc& out_n_k_ho_wo_global_desc,
                             const Array<index_t, 2> conv_strides,
                             const Array<index_t, 2> conv_dilations,
                             const Array<index_t, 2> in_left_pads,
                             const Array<index_t, 2> in_right_pads)
{
    const index_t N = in_n_c_hi_wi_global_desc.GetLength(0);
    const index_t C = in_n_c_hi_wi_global_desc.GetLength(1);
    const index_t K = out_n_k_ho_wo_global_desc.GetLength(1);

    const index_t Y = wei_k_c_y_x_global_desc.GetLength(2);
    const index_t X = wei_k_c_y_x_global_desc.GetLength(3);

    const index_t Hi = in_n_c_hi_wi_global_desc.GetLength(2);
    const index_t Wi = in_n_c_hi_wi_global_desc.GetLength(3);

    const index_t Ho = out_n_k_ho_wo_global_desc.GetLength(2);
    const index_t Wo = out_n_k_ho_wo_global_desc.GetLength(3);

    const index_t ConvStrideH = conv_strides[0];
    const index_t ConvStrideW = conv_strides[1];

    const index_t ConvDilationH = conv_dilations[0];
    const index_t ConvDilationW = conv_dilations[1];

    const index_t InLeftPadH  = in_left_pads[0];
    const index_t InLeftPadW  = in_left_pads[1];
    const index_t InRightPadH = in_right_pads[0];
    const index_t InRightPadW = in_right_pads[1];

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

    const index_t Hip = in_n_c_hip_wip_global_desc.GetLength(2);
    const index_t Wip = in_n_c_hip_wip_global_desc.GetLength(3);

    const auto in_n_c_y_ho_x_wo_global_desc = transform_dynamic_tensor_descriptor_v2(
        in_n_c_hip_wip_global_desc,
        make_tuple(DynamicPassThrough{N},
                   DynamicPassThrough{C},
                   DynamicEmbed<2>{{Y, Ho}, {ConvDilationH, ConvStrideH, 0}},
                   DynamicEmbed<2>{{X, Wo}, {ConvDilationW, ConvStrideW, 0}}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

    const auto in_gemmk_gemmn_global_desc = transform_dynamic_tensor_descriptor_v2(
        in_n_c_y_ho_x_wo_global_desc,
        make_tuple(DynamicMerge<3>{{C, Y, X}}, DynamicMerge<3>{{N, Ho, Wo}}),
        make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    return make_tuple(in_gemmk_gemmn_global_desc);
}

template <index_t BlockSize>
struct DummyDynamicTransform_1
{
    template <typename WeiDesc, typename InDesc, typename OutDesc>
    __device__ void Run_0(index_t* const __restrict__ p_wei_global,
                          float* const __restrict__ p_in_global,
                          float* const __restrict__ p_out_global,
                          const WeiDesc wei_k_c_y_x_global_desc,
                          const InDesc in_n_c_hi_wi_global_desc,
                          const OutDesc out_n_k_ho_wo_global_desc,
                          const Array<index_t, 2> conv_strides,
                          const Array<index_t, 2> conv_dilations,
                          const Array<index_t, 2> in_left_pads,
                          const Array<index_t, 2> in_right_pads) const
    {
#if 1
        const index_t N = in_n_c_hi_wi_global_desc.GetLength(0);
        const index_t C = in_n_c_hi_wi_global_desc.GetLength(1);
        const index_t K = out_n_k_ho_wo_global_desc.GetLength(1);

        const index_t Y = wei_k_c_y_x_global_desc.GetLength(2);
        const index_t X = wei_k_c_y_x_global_desc.GetLength(3);

        const index_t Hi = in_n_c_hi_wi_global_desc.GetLength(2);
        const index_t Wi = in_n_c_hi_wi_global_desc.GetLength(3);

        const index_t Ho = out_n_k_ho_wo_global_desc.GetLength(2);
        const index_t Wo = out_n_k_ho_wo_global_desc.GetLength(3);

        const index_t ConvStrideH = conv_strides[0];
        const index_t ConvStrideW = conv_strides[1];

        const index_t ConvDilationH = conv_dilations[0];
        const index_t ConvDilationW = conv_dilations[1];

        const index_t InLeftPadH  = in_left_pads[0];
        const index_t InLeftPadW  = in_left_pads[1];
        const index_t InRightPadH = in_right_pads[0];
        const index_t InRightPadW = in_right_pads[1];
#else
        const index_t N = in_n_c_hi_wi_global_desc.GetLength(0);
        const index_t C = in_n_c_hi_wi_global_desc.GetLength(1);

        const index_t Y = 3;
        const index_t X = 3;

        const index_t Hi = in_n_c_hi_wi_global_desc.GetLength(2);
        const index_t Wi = in_n_c_hi_wi_global_desc.GetLength(3);

        const index_t ConvStrideH = conv_strides[0];
        const index_t ConvStrideW = conv_strides[1];

        const index_t ConvDilationH = conv_dilations[0];
        const index_t ConvDilationW = conv_dilations[1];

        const index_t InLeftPadH  = in_left_pads[0];
        const index_t InLeftPadW  = in_left_pads[1];
        const index_t InRightPadH = in_right_pads[0];
        const index_t InRightPadW = in_right_pads[1];
#endif

        // define transform
        // pass through
        auto f_lower_idx_diff_passthrough = [](index_t& idx_low_diff, const index_t& idx_up_diff) {
            idx_low_diff = idx_up_diff;
        };

        // pad
        auto f_lower_idx_diff_pad = [](index_t& idx_low_diff, const index_t& idx_up_diff) {
            idx_low_diff = idx_up_diff;
        };

        // embed
        auto f_lower_idx_diff_embed = [](index_t& idx_low_diff,
                                         const index_t& idx_up_diff_0,
                                         const index_t& idx_up_diff_1,
                                         const index_t coeff0,
                                         const index_t coeff1) {
            idx_low_diff = coeff0 * idx_up_diff_0 + coeff1 * idx_up_diff_1;
        };

        // unmerge
        auto f_lower_idx_diff_unmerge = [](index_t& idx_low_diff,
                                           const index_t& idx_up_diff_0,
                                           const index_t& idx_up_diff_1,
                                           const index_t up_length_1) {
            idx_low_diff = up_length_1 * idx_up_diff_0 + idx_up_diff_1;
        };

        // merge
        auto f_lower_idx_diff_merge_v1 = [](index_t& idx_low_diff_0,
                                            index_t& idx_low_diff_1,
                                            index_t& idx_low_diff_2,
                                            const index_t& idx_up_diff,
                                            const index_t& idx_low_old_0,
                                            const index_t& idx_low_old_1,
                                            const index_t& idx_low_old_2,
                                            const index_t& idx_low_diff_const_0,
                                            const index_t& idx_low_diff_const_1,
                                            const index_t& idx_low_diff_const_2,
                                            const index_t& idx_low_bound_0,
                                            const index_t& idx_low_bound_1,
                                            const index_t& idx_low_bound_2) {
            auto f_carry_arithmetic = [](index_t& idx_low_diff,
                                         index_t& carry,
                                         const index_t& idx_low_old,
                                         const index_t& idx_low_diff_const,
                                         const index_t& idx_low_bound) {
                index_t idx_low_tmp = idx_low_old + carry + idx_low_diff_const;

#if 1 // positive
                bool do_carry = idx_low_tmp >= idx_low_bound;

                index_t idx_low_new = do_carry ? idx_low_tmp - idx_low_bound : idx_low_tmp;

                carry = do_carry ? 1 : 0;
#else // negative
                bool do_borrow = idx_low_tmp < 0;

                index_t idx_low_new = do_borrow ? idx_low_tmp + idx_low_bound : idx_low_tmp;

                carry          = do_borrow ? -1 : 0;
#endif

                idx_low_diff = idx_low_new - idx_low_old;
            };

            index_t carry = 0;

            f_carry_arithmetic(
                idx_low_diff_2, carry, idx_low_old_2, idx_low_diff_const_2, idx_low_bound_2);
            f_carry_arithmetic(
                idx_low_diff_1, carry, idx_low_old_1, idx_low_diff_const_1, idx_low_bound_1);

            idx_low_diff_0 = idx_low_diff_const_0 + carry;
        };

        auto f_lower_idx_diff_merge_v2 = [](index_t& idx_low_diff_0,
                                            index_t& idx_low_diff_1,
                                            index_t& idx_low_diff_2,
                                            const index_t& idx_up_diff,
                                            const index_t& idx_low_old_0,
                                            const index_t& idx_low_old_1,
                                            const index_t& idx_low_old_2,
                                            const index_t& idx_low_diff_const_0,
                                            const index_t& idx_low_diff_const_1,
                                            const index_t& idx_low_diff_const_2,
                                            const index_t& idx_low_bound_0,
                                            const index_t& idx_low_bound_1,
                                            const index_t& idx_low_bound_2) {
            auto f_carry_arithmetic = [](index_t& idx_low_diff,
                                         index_t& carry,
                                         const index_t& idx_low_old,
                                         const index_t& idx_low_diff_const,
                                         const index_t& idx_low_bound) {
                index_t idx_low_tmp                            = idx_low_old + carry;
                index_t idx_low_bound_minus_idx_low_diff_const = idx_low_bound - idx_low_diff_const;

#if 1 // positive
                bool do_carry = idx_low_tmp >= idx_low_bound_minus_idx_low_diff_const;

                idx_low_diff =
                    do_carry ? -idx_low_bound_minus_idx_low_diff_const : idx_low_diff_const;

                idx_low_diff += carry;

                carry = do_carry ? 1 : 0;
#else // negative
                bool do_borrow = idx_low_tmp < -idx_low_diff_const;

                idx_low_diff = do_borrow ? idx_low_diff_const + idx_low_bound : idx_low_diff_const;

                idx_low_diff -= carry;

                carry          = do_borrow ? 1 : carry;
#endif
            };

            index_t carry = 0;

            f_carry_arithmetic(
                idx_low_diff_2, carry, idx_low_old_2, idx_low_diff_const_2, idx_low_bound_2);
            f_carry_arithmetic(
                idx_low_diff_1, carry, idx_low_old_1, idx_low_diff_const_1, idx_low_bound_1);

            idx_low_diff_0 = idx_low_diff_const_0 + carry;
        };

        auto f_lower_idx_diff_merge_v3 = [](index_t& idx_low_diff_0,
                                            index_t& idx_low_diff_1,
                                            index_t& idx_low_diff_2,
                                            const index_t& idx_up_diff,
                                            const index_t& idx_low_old_0,
                                            const index_t& idx_low_old_1,
                                            const index_t& idx_low_old_2,
                                            const index_t& idx_low_diff_const_0,
                                            const index_t& idx_low_diff_const_1,
                                            const index_t& idx_low_diff_const_2,
                                            const index_t& idx_low_bound_0,
                                            const index_t& idx_low_bound_1,
                                            const index_t& idx_low_bound_2) {
            auto f_carry_arithmetic = [](index_t& idx_low_diff,
                                         index_t& negative_carry,
                                         const index_t& idx_low_old,
                                         const index_t& idx_low_diff_const,
                                         const index_t& idx_low_bound) {
                index_t neg_idx_low_tmp = negative_carry - idx_low_old;

                index_t idx_low_diff_const_minus_idx_low_bound = idx_low_diff_const - idx_low_bound;

#if 1
                bool do_carry = neg_idx_low_tmp <= idx_low_diff_const_minus_idx_low_bound;

                idx_low_diff =
                    do_carry ? idx_low_diff_const_minus_idx_low_bound : idx_low_diff_const;

                idx_low_diff -= negative_carry;

                negative_carry = do_carry ? -1 : 0;
#else
                bool do_borrow = neg_idx_low_tmp > idx_low_diff_const;

                idx_low_diff = do_borrow ? idx_low_diff_const + idx_low_bound : idx_low_diff_const;

                idx_low_diff -= negative_carry;

                negative_carry = do_borrow ? 1 : negative_carry;
#endif
            };

            index_t negative_carry = 0;

            f_carry_arithmetic(idx_low_diff_2,
                               negative_carry,
                               idx_low_old_2,
                               idx_low_diff_const_2,
                               idx_low_bound_2);
            f_carry_arithmetic(idx_low_diff_1,
                               negative_carry,
                               idx_low_old_1,
                               idx_low_diff_const_1,
                               idx_low_bound_1);

            idx_low_diff_0 = idx_low_diff_const_0 - negative_carry;
        };

        index_t idx[20];
        index_t idx_diff[20];
        index_t const_tmp[6];

        // populate const
        const index_t GemmKPack = p_wei_global[0];

#if 1
        for(index_t i = 0; i < 6; ++i)
        {
            const_tmp[i] = p_wei_global[i + 1];
        }
#else
        const_tmp[0] = 0;
        const_tmp[1] = 2;
        const_tmp[2] = 2;
#endif

        // initialize idx
        for(index_t i = 0; i < 20; ++i)
        {
            idx[i] = p_wei_global[get_thread_local_1d_id() + i];
        }

        // offset
        idx[0] = idx[1] * in_n_c_hi_wi_global_desc.GetStride(0) +
                 idx[2] * in_n_c_hi_wi_global_desc.GetStride(1) +
                 idx[3] * in_n_c_hi_wi_global_desc.GetStride(2) +
                 idx[4] * in_n_c_hi_wi_global_desc.GetStride(3);

// start lowering diff
#pragma unroll 1
        for(index_t i = 0; i < 100; ++i)
        {
            for(index_t i = 0; i < 20; ++i)
            {
                idx_diff[i] = 0;
            }

            idx_diff[17] = 8;

            // stage 4
            // Unmerge(GemmKTotal) => GemmK, GemmKPack
            f_lower_idx_diff_unmerge(idx_diff[15], idx_diff[17], idx_diff[18], GemmKPack);

            // PassThrough GemmN => GemmN
            f_lower_idx_diff_passthrough(idx_diff[16], idx_diff[19]);

// stage 3
// Merge(C, Y, X) => GemmKTotal
#if 0
            f_lower_idx_diff_merge_v2(idx_diff[10],
                                      idx_diff[11],
                                      idx_diff[13],
                                      idx_diff[15],
                                      idx[10],
                                      idx[11],
                                      idx[13],
                                      const_tmp[0],
                                      const_tmp[1],
                                      const_tmp[2],
                                      C,
                                      Y,
                                      X);
#elif 0
            index_t tmp               = idx_diff[15];
            const index_t const_tmp_0 = tmp / (Y * X);
            tmp -= const_tmp_0 * (Y * X);
            const index_t const_tmp_1 = tmp / X;
            const index_t const_tmp_2 = tmp - const_tmp_1 * X;

            f_lower_idx_diff_merge_v2(idx_diff[10],
                                      idx_diff[11],
                                      idx_diff[13],
                                      idx_diff[15],
                                      idx[10],
                                      idx[11],
                                      idx[13],
                                      const_tmp_0,
                                      const_tmp_1,
                                      const_tmp_2,
                                      C,
                                      Y,
                                      X);
#elif 1
            index_t tmp               = idx_diff[15];
            const index_t const_tmp_0 = __llvm_amdgcn_readfirstlane_i32(tmp / (Y * X));
            tmp -= const_tmp_0 * (Y * X);
            const index_t const_tmp_1 = __llvm_amdgcn_readfirstlane_i32(tmp / X);
            const index_t const_tmp_2 = __llvm_amdgcn_readfirstlane_i32(tmp - const_tmp_1 * X);

            f_lower_idx_diff_merge_v2(idx_diff[10],
                                      idx_diff[11],
                                      idx_diff[13],
                                      idx_diff[15],
                                      idx[10],
                                      idx[11],
                                      idx[13],
                                      const_tmp_0,
                                      const_tmp_1,
                                      const_tmp_2,
                                      C,
                                      Y,
                                      X);
#endif

            // stage 2
            // PassThrough(N) => N
            f_lower_idx_diff_passthrough(idx_diff[5], idx_diff[9]);

            // PassThrough(C) => C
            f_lower_idx_diff_passthrough(idx_diff[6], idx_diff[10]);

            // Embed(Hip) => Y, Ho
            f_lower_idx_diff_embed(
                idx_diff[7], idx_diff[11], idx_diff[12], ConvDilationH, ConvStrideH);

            // Embed(Wip) => X, Wo
            f_lower_idx_diff_embed(
                idx_diff[8], idx_diff[13], idx_diff[14], ConvDilationW, ConvStrideW);

            // stage 1
            // PassThrough(N) => N
            f_lower_idx_diff_passthrough(idx_diff[1], idx_diff[5]);

            // PassThrough(C) => C
            f_lower_idx_diff_passthrough(idx_diff[2], idx_diff[6]);

            // Pad(Hi) => Hip
            f_lower_idx_diff_pad(idx_diff[3], idx_diff[7]);

            // Pad(Wi) => Wip
            f_lower_idx_diff_pad(idx_diff[4], idx_diff[8]);

            // stage 0
            // offset_diff
            idx_diff[0] = idx_diff[1] * in_n_c_hi_wi_global_desc.GetStride(0) +
                          idx_diff[2] * in_n_c_hi_wi_global_desc.GetStride(1) +
                          idx_diff[3] * in_n_c_hi_wi_global_desc.GetStride(2) +
                          idx_diff[4] * in_n_c_hi_wi_global_desc.GetStride(3);

#if 0
            // update idx
            for(index_t i = 0; i < 20; ++ i)
            {
                idx[i] += idx_diff[i];
            }

            // padding check
            bool is_in_bound = idx[3] >= 0 && idx[3] < Hi && idx[4] >= 0 && idx[4] < Wi;
#elif 0 // no pad
        // offset
            idx[0] += idx_diff[0];

            // C, Y, X
            idx[10] += idx_diff[10];
            idx[11] += idx_diff[11];
            idx[13] += idx_diff[13];

            // padding check
            bool is_in_bound = true;
#else   // pad
        // offset
            idx[0] += idx_diff[0];

            // C, Y, X
            idx[10] += idx_diff[10];
            idx[11] += idx_diff[11];
            idx[13] += idx_diff[13];

            // Hi, Wi
            idx[3] += idx_diff[3];
            idx[4] += idx_diff[4];

            // padding check
            bool is_in_bound = idx[3] >= 0 && idx[3] < Hi && idx[4] >= 0 && idx[4] < Wi;
#endif

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
                             idx[0],
                             is_in_bound,
                             out_n_k_ho_wo_global_desc.GetElementSpace());
        }
    }

    template <typename WeiDesc, typename InDesc, typename OutDesc>
    __device__ void Run_1(index_t* const __restrict__ p_wei_global,
                          float* const __restrict__ p_in_global,
                          float* const __restrict__ p_out_global,
                          const WeiDesc wei_k_c_y_x_global_desc,
                          const InDesc in_n_c_hi_wi_global_desc,
                          const OutDesc out_n_k_ho_wo_global_desc,
                          const Array<index_t, 2> conv_strides,
                          const Array<index_t, 2> conv_dilations,
                          const Array<index_t, 2> in_left_pads,
                          const Array<index_t, 2> in_right_pads) const
    {
        const auto transformed_tensor_descs = map_convolution_into_gemm(wei_k_c_y_x_global_desc,
                                                                        in_n_c_hi_wi_global_desc,
                                                                        out_n_k_ho_wo_global_desc,
                                                                        conv_strides,
                                                                        conv_dilations,
                                                                        in_left_pads,
                                                                        in_right_pads);

        const auto in_gemmk_gemmn_global_desc = transformed_tensor_descs.At(Number<0>{});

        MultiIndex<2> idx;

        // initialize idx
        for(index_t i = 0; i < 2; ++i)
        {
            idx(i) = p_wei_global[get_thread_local_1d_id() + i];
        }

        const index_t niter = p_wei_global[10];

        auto in_gemmk_gemmn_coord = make_dynamic_tensor_coordinate(in_gemmk_gemmn_global_desc, idx);

        for(index_t iter = 0; iter < niter; ++iter)
        {
            constexpr auto gemmk1_gemmn0 = MultiIndex<2>{1, 0};

            in_gemmk_gemmn_coord += gemmk1_gemmn0;

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
#if 0
                             in_gemmk_gemmn_coord.IsOffsetValidAssumingUpperIndexIsValid(),
#else
                             true,
#endif
                             in_gemmk_gemmn_global_desc.GetElementSpace());
        }
    }

    template <typename WeiDesc, typename InDesc, typename OutDesc>
    __device__ void Run_2(index_t* const __restrict__ p_wei_global,
                          float* const __restrict__ p_in_global,
                          float* const __restrict__ p_out_global,
                          const WeiDesc wei_k_c_y_x_global_desc,
                          const InDesc in_n_c_hi_wi_global_desc,
                          const OutDesc out_n_k_ho_wo_global_desc,
                          const Array<index_t, 2> conv_strides,
                          const Array<index_t, 2> conv_dilations,
                          const Array<index_t, 2> in_left_pads,
                          const Array<index_t, 2> in_right_pads) const
    {
        const auto transformed_tensor_descs =
            map_convolution_into_gemm_v2(wei_k_c_y_x_global_desc,
                                         in_n_c_hi_wi_global_desc,
                                         out_n_k_ho_wo_global_desc,
                                         conv_strides,
                                         conv_dilations,
                                         in_left_pads,
                                         in_right_pads);

        const auto in_gemmk_gemmn_global_desc = transformed_tensor_descs.At(Number<0>{});

        MultiIndex<2> idx;

        // initialize idx
        for(index_t i = 0; i < 2; ++i)
        {
            idx(i) = p_wei_global[get_thread_local_1d_id() + i];
        }

        const index_t niter = p_wei_global[10];

        auto in_gemmk_gemmn_coord =
            make_dynamic_tensor_coordinate_v2(in_gemmk_gemmn_global_desc, idx);

        const auto in_gemmk_gemmn_coord_step = make_dynamic_tensor_coordinate_step_v2(
            in_gemmk_gemmn_global_desc, MultiIndex<2>{{1, 0}});

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
                             coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                 in_gemmk_gemmn_global_desc, in_gemmk_gemmn_coord),
                             in_gemmk_gemmn_global_desc.GetElementSpaceSize());
        }
    }

    template <typename WeiDesc, typename InDesc, typename OutDesc>
    __device__ void Run(index_t* const __restrict__ p_wei_global,
                        float* const __restrict__ p_in_global,
                        float* const __restrict__ p_out_global,
                        const WeiDesc wei_k_c_y_x_global_desc,
                        const InDesc in_n_c_hi_wi_global_desc,
                        const OutDesc out_n_k_ho_wo_global_desc,
                        const Array<index_t, 2> conv_strides,
                        const Array<index_t, 2> conv_dilations,
                        const Array<index_t, 2> in_left_pads,
                        const Array<index_t, 2> in_right_pads) const
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
struct DummyDynamicTransform_2
{
    template <typename WeiDesc, typename InDesc, typename OutDesc>
    __device__ void Run(index_t* const __restrict__ p_wei_global,
                        float* const __restrict__ p_in_global,
                        float* const __restrict__ p_out_global,
                        const WeiDesc wei_k_c_y_x_global_desc,
                        const InDesc in_n_c_hi_wi_global_desc,
                        const OutDesc out_n_k_ho_wo_global_desc,
                        const Array<index_t, 2> conv_strides,
                        const Array<index_t, 2> conv_dilations,
                        const Array<index_t, 2> in_left_pads,
                        const Array<index_t, 2> in_right_pads) const
    {
        const index_t N = in_n_c_hi_wi_global_desc.GetLength(0);
        const index_t C = in_n_c_hi_wi_global_desc.GetLength(1);
        const index_t K = out_n_k_ho_wo_global_desc.GetLength(1);

        const index_t Y = wei_k_c_y_x_global_desc.GetLength(2);
        const index_t X = wei_k_c_y_x_global_desc.GetLength(3);

        const index_t Hi = in_n_c_hi_wi_global_desc.GetLength(2);
        const index_t Wi = in_n_c_hi_wi_global_desc.GetLength(3);

        const index_t Ho = out_n_k_ho_wo_global_desc.GetLength(2);
        const index_t Wo = out_n_k_ho_wo_global_desc.GetLength(3);

        const index_t ConvStrideH = conv_strides[0];
        const index_t ConvStrideW = conv_strides[1];

        const index_t ConvDilationH = conv_dilations[0];
        const index_t ConvDilationW = conv_dilations[1];

        const index_t InLeftPadH  = in_left_pads[0];
        const index_t InLeftPadW  = in_left_pads[1];
        const index_t InRightPadH = in_right_pads[0];
        const index_t InRightPadW = in_right_pads[1];

        const auto in_n_c_hip_wip_global_desc = transform_dynamic_tensor_descriptor_v2(
            in_n_c_hi_wi_global_desc,
            make_tuple(DynamicPassThrough{N},
                       DynamicPassThrough{C},
                       DynamicLeftPad{Hi, InLeftPadH},
                       DynamicLeftPad{Wi, InLeftPadW}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        MultiIndex<4> idx;

        // initialize idx
        for(index_t i = 0; i < 4; ++i)
        {
            idx(i) = p_wei_global[get_thread_local_1d_id() + i];
        }

        const index_t niter = p_wei_global[10];

        auto in_coord = make_dynamic_tensor_coordinate_v2(in_n_c_hip_wip_global_desc, idx);

        const auto in_coord_step = make_dynamic_tensor_coordinate_step_v2(
            in_n_c_hip_wip_global_desc, MultiIndex<4>{{1, 0, 0, 0}});

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
    }
};

template <index_t BlockSize>
struct DummyDynamicTransform_3
{
    template <typename WeiDesc, typename InDesc, typename OutDesc, typename TransformInDesc>
    __device__ void Run(index_t* const __restrict__ p_wei_global,
                        float* const __restrict__ p_in_global,
                        float* const __restrict__ p_out_global,
                        const WeiDesc wei_k_c_y_x_global_desc,
                        const InDesc in_n_c_hi_wi_global_desc,
                        const OutDesc out_n_k_ho_wo_global_desc,
                        const TransformInDesc in_gemmk_gemmn_global_desc,
                        const Array<index_t, 2> conv_strides,
                        const Array<index_t, 2> conv_dilations,
                        const Array<index_t, 2> in_left_pads,
                        const Array<index_t, 2> in_right_pads) const
    {
        MultiIndex<2> idx;

        // initialize idx
        for(index_t i = 0; i < 2; ++i)
        {
            idx(i) = p_wei_global[get_thread_local_1d_id() + i];
        }

        const index_t niter = p_wei_global[10];

        auto in_gemmk_gemmn_coord =
            make_dynamic_tensor_coordinate_v2(in_gemmk_gemmn_global_desc, idx);

        const auto in_gemmk_gemmn_coord_step = make_dynamic_tensor_coordinate_step_v2(
            in_gemmk_gemmn_global_desc, MultiIndex<2>{{1, 0}});

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
                             coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                 in_gemmk_gemmn_global_desc, in_gemmk_gemmn_coord),
                             in_gemmk_gemmn_global_desc.GetElementSpaceSize());
        }
    }
};

} // namespace ck
#endif
