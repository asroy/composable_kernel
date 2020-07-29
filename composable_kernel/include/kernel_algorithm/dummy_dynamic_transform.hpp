#ifndef CK_DUMMY_DYNAMIC_TRANSFORM_HPP
#define CK_DUMMY_DYNAMIC_TRANSFORM_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

template <index_t BlockSize>
struct DummyDynamicTransform
{
    __device__ void Run_(index_t* const __restrict__ p_wei_global,
                         index_t* const __restrict__ p_in_global,
                         float* const __restrict__ p_out_global,
                         const DynamicNativeTensorDescriptor<4> wei_k_c_y_x_global_desc,
                         const DynamicNativeTensorDescriptor<4> in_n_c_hi_wi_global_desc,
                         const DynamicNativeTensorDescriptor<4> out_n_k_ho_wo_global_desc,
                         const Array<index_t, 2> conv_strides,
                         const Array<index_t, 2> conv_dilations,
                         const Array<index_t, 2> in_left_pads,
                         const Array<index_t, 2> in_right_pads,
                         index_t k_block_num,
                         index_t c_block_num,
                         index_t y_block_num,
                         index_t x_block_num) const
    {
        const index_t N  = in_n_c_hi_wi_global_desc.GetLength(0);
        const index_t C  = in_n_c_hi_wi_global_desc.GetLength(1);
        const index_t Hi = in_n_c_hi_wi_global_desc.GetLength(2);
        const index_t Wi = in_n_c_hi_wi_global_desc.GetLength(3);

        const index_t K  = out_n_k_ho_wo_global_desc.GetLength(1);
        const index_t Ho = out_n_k_ho_wo_global_desc.GetLength(2);
        const index_t Wo = out_n_k_ho_wo_global_desc.GetLength(3);

        const index_t Y = wei_k_c_y_x_global_desc.GetLength(2);
        const index_t X = wei_k_c_y_x_global_desc.GetLength(3);

        const index_t ConvStrideH = conv_strides[0];
        const index_t ConvStrideW = conv_strides[1];

        const index_t ConvDilationH = conv_dilations[0];
        const index_t ConvDilationW = conv_dilations[1];

        p_wei_global[0] = wei_k_c_y_x_global_desc.GetElementSize();
        p_wei_global[1] = wei_k_c_y_x_global_desc.GetElementSpace();

        const index_t k_block_num_stride = c_block_num * y_block_num * x_block_num;
        const index_t c_block_num_stride = y_block_num * x_block_num;
        const index_t y_block_num_stride = x_block_num;

        index_t tmp = get_block_1d_id();
#if 0
        const index_t k_block = tmp / k_block_num_stride;
        tmp -= k_block * k_block_num_stride;
        const index_t c_block = tmp / c_block_num_stride;
        tmp -= c_block * c_block_num_stride;
        const index_t y_block = tmp / y_block_num_stride;
        tmp -= y_block * y_block_num_stride;
        const index_t x_block = tmp;
#else
        const index_t k_block = __llvm_amdgcn_readfirstlane_i32(tmp / k_block_num_stride);
        tmp -= k_block * k_block_num_stride;
        const index_t c_block = __llvm_amdgcn_readfirstlane_i32(tmp / c_block_num_stride);
        tmp -= c_block * c_block_num_stride;
        const index_t y_block = __llvm_amdgcn_readfirstlane_i32(tmp / y_block_num_stride);
        tmp -= y_block * y_block_num_stride;
        const index_t x_block = __llvm_amdgcn_readfirstlane_i32(tmp);
#endif
        const index_t k_thread = p_in_global[get_thread_local_1d_id()];
        const index_t c_thread = p_in_global[get_thread_local_1d_id() + 1];
        const index_t y_thread = p_in_global[get_thread_local_1d_id() + 2];
        const index_t x_thread = p_in_global[get_thread_local_1d_id() + 3];

        p_wei_global[3] = wei_k_c_y_x_global_desc.CalculateOffset(
            {k_block + k_thread, c_block + c_thread, y_block + y_thread, x_block + x_thread});
    }

    __device__ void Run(index_t* const __restrict__ p_wei_global,
                        index_t* const __restrict__ p_in_global,
                        float* const __restrict__ p_out_global,
                        const DynamicNativeTensorDescriptor<4> wei_k_c_y_x_global_desc,
                        const DynamicNativeTensorDescriptor<4> in_n_c_hi_wi_global_desc,
                        const DynamicNativeTensorDescriptor<4> out_n_k_ho_wo_global_desc,
                        const Array<index_t, 2> conv_strides,
                        const Array<index_t, 2> conv_dilations,
                        const Array<index_t, 2> in_left_pads,
                        const Array<index_t, 2> in_right_pads,
                        index_t,
                        index_t,
                        index_t,
                        index_t) const
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
        const index_t N       = in_n_c_hi_wi_global_desc.GetLength(0);
        const index_t C       = in_n_c_hi_wi_global_desc.GetLength(1);

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

#if 0 // positive
                bool do_carry = idx_low_tmp >= idx_low_bound;

                index_t idx_low_new = do_carry ? idx_low_tmp - idx_low_bound : idx_low_tmp;

                carry = do_carry ? 1 : 0;
#else // negative
                bool do_borrow    = idx_low_tmp < 0;

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
        const_tmp[0]           = 0;
        const_tmp[1]           = 2;
        const_tmp[2]           = 2;
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

#if 0
            // Merge(N, Ho, Wo) => GemmN
            f_lower_idx_diff_merge(idx_diff[9],
                                   idx_diff[12],
                                   idx_diff[14],
                                   idx_diff[16],
                                   idx[9],
                                   idx[12],
                                   idx[14],
                                   const_tmp[3],
                                   const_tmp[4],
                                   const_tmp[5],
                                   N,
                                   Ho,
                                   Wo);
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
};

} // namespace ck
#endif
