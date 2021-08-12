/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CK_DYNAMIC_GRIDWISE_GENERIC_REDUCTION_FIRST_CALL_HPP
#define CK_DYNAMIC_GRIDWISE_GENERIC_REDUCTION_FIRST_CALL_HPP

#include "data_type.hpp"

#include "dynamic_gridwise_generic_2d_reduction_direct_threadwise.hpp"
#include "dynamic_gridwise_generic_2d_reduction_direct_warpwise.hpp"
#include "dynamic_gridwise_generic_2d_reduction_blockwise.hpp"
#include "dynamic_gridwise_generic_2d_reduction_multiblock.hpp"

namespace ck {

template <index_t BlockSize,
          typename srcDataType, // the type with which the data of the source tensor are stored
          typename dstDataType, // the type with which the data of the destintion tensor are stored
          typename compType,    // the type used by the reduce binary operator
          index_t reduceImpl_I,
          index_t op_I,          // the enumerate value representing the operation used in Reduction
          index_t nanPropaOpt_I, // the enumerate value representing the NanPropagation Option
          index_t reduceIndicesOpt_I, // the enumerate value representing the Reduce Indices Option
          index_t GredThreadBufferLength,
          index_t GredAccessesPerThreadInBlock,
          index_t GredAccessesPerThreadInWarp>
struct Gridwise2dReduction
{
    static constexpr auto reduceImpl       = static_cast<ReductionMethod_t>(reduceImpl_I);
    static constexpr auto op               = static_cast<ReduceTensorOp_t>(op_I);
    static constexpr auto nanPropaOpt      = static_cast<NanPropagation_t>(nanPropaOpt_I);
    static constexpr auto reduceIndicesOpt = static_cast<ReduceTensorIndices_t>(reduceIndicesOpt_I);

    static constexpr bool indexable = reduce_binary_operator<compType, op>::indexable;
    static constexpr bool need_indices =
        indexable && (reduceIndicesOpt != ReduceTensorIndices_t::NO_INDICES);

    __device__ Gridwise2dReduction(int origReduceLen_, int BlkGroupSize_)
    {
        origReduceLen = origReduceLen_;
        BlkGroupSize  = BlkGroupSize_;
    };

    template <ReductionMethod_t impl>
    struct Gridwise2dReduction_impl_wrapper;

    template <>
    struct Gridwise2dReduction_impl_wrapper<ReductionMethod_t::DirectThreadWise>
    {
        template <typename src2dDescType, typename dst1dDescType>
        __device__ static void RunMethod(const src2dDescType& src2dDesc,
                                         const dst1dDescType& dst1dDesc,
                                         int origReduceLen,
                                         int BlkGroupSize,
                                         srcDataType alpha,
                                         const srcDataType* const __restrict__ p_src_global,
                                         dstDataType beta,
                                         dstDataType* const __restrict__ p_dst_global,
                                         srcDataType* const __restrict__ ws_buf1_global,
                                         int* const __restrict__ ws_buf2_global,
                                         int* const __restrict__ indices_global)
        {
            (void)ws_buf1_global; // unused
            (void)BlkGroupSize;

            using gridwise_reduce =
                GridwiseReduction_xy_to_x_direct_threadwise<BlockSize,
                                                            srcDataType,
                                                            dstDataType,
                                                            compType,
                                                            src2dDescType,
                                                            dst1dDescType,
                                                            op,
                                                            nanPropaOpt,
                                                            reduceIndicesOpt,
                                                            true,
                                                            true,
                                                            GredThreadBufferLength>;
            constexpr int RunId = need_indices ? 2 : 1;
            gridwise_reduce::template Run<RunId>(
                src2dDesc,
                dst1dDesc,
                origReduceLen,
                alpha,
                p_src_global,
                beta,
                p_dst_global,
                const_cast<const int* const __restrict__>(ws_buf2_global),
                indices_global); // ws_buf2_global will be read at the second-time
        };
    };

    template <>
    struct Gridwise2dReduction_impl_wrapper<ReductionMethod_t::DirectWarpWise>
    {
        template <typename src2dDescType, typename dst1dDescType>
        __device__ static void RunMethod(const src2dDescType& src2dDesc,
                                         const dst1dDescType& dst1dDesc,
                                         int origReduceLen,
                                         int BlkGroupSize,
                                         srcDataType alpha,
                                         const srcDataType* const __restrict__ p_src_global,
                                         dstDataType beta,
                                         dstDataType* const __restrict__ p_dst_global,
                                         srcDataType* const __restrict__ ws_buf1_global,
                                         int* const __restrict__ ws_buf2_global,
                                         int* const __restrict__ indices_global)
        {
            (void)ws_buf1_global; // unused
            (void)BlkGroupSize;

            using gridwise_reduce =
                GridwiseReduction_xy_to_x_direct_warpwise<BlockSize,
                                                          srcDataType,
                                                          dstDataType,
                                                          compType,
                                                          src2dDescType,
                                                          dst1dDescType,
                                                          op,
                                                          nanPropaOpt,
                                                          reduceIndicesOpt,
                                                          true,
                                                          true,
                                                          GredAccessesPerThreadInWarp>;
            constexpr int RunId = need_indices ? 2 : 1;
            gridwise_reduce::template Run<RunId>(
                src2dDesc,
                dst1dDesc,
                origReduceLen,
                alpha,
                p_src_global,
                beta,
                p_dst_global,
                const_cast<const int* const __restrict__>(ws_buf2_global),
                indices_global); // ws_buf2_global will be read at the second-time
        };
    };

    template <>
    struct Gridwise2dReduction_impl_wrapper<ReductionMethod_t::BlockWise>
    {
        template <typename src2dDescType, typename dst1dDescType>
        __device__ static void RunMethod(const src2dDescType& src2dDesc,
                                         const dst1dDescType& dst1dDesc,
                                         int origReduceLen,
                                         int BlkGroupSize,
                                         srcDataType alpha,
                                         const srcDataType* const __restrict__ p_src_global,
                                         dstDataType beta,
                                         dstDataType* const __restrict__ p_dst_global,
                                         srcDataType* const __restrict__ ws_buf1_global,
                                         int* const __restrict__ ws_buf2_global,
                                         int* const __restrict__ indices_global)
        {
            (void)ws_buf1_global; // unused
            (void)BlkGroupSize;   // unused

            using gridwise_reduce =
                GridwiseReduction_xy_to_x_blockwise<BlockSize,
                                                    srcDataType,
                                                    dstDataType,
                                                    compType,
                                                    src2dDescType,
                                                    dst1dDescType,
                                                    op,
                                                    nanPropaOpt,
                                                    reduceIndicesOpt,
                                                    true,
                                                    true,
                                                    GredAccessesPerThreadInBlock>;
            constexpr int RunId = need_indices ? 2 : 1;
            gridwise_reduce::template Run<RunId>(
                src2dDesc,
                dst1dDesc,
                origReduceLen,
                alpha,
                p_src_global,
                beta,
                p_dst_global,
                const_cast<const int* const __restrict__>(ws_buf2_global),
                indices_global); // ws_buf2_global will be read at the second-time
        };
    };

    template <>
    struct Gridwise2dReduction_impl_wrapper<ReductionMethod_t::MultiBlock>
    {
        template <typename src2dDescType, typename dst1dDescType>
        __device__ static void RunMethod(const src2dDescType& src2dDesc,
                                         const dst1dDescType& dst1dDesc,
                                         int origReduceLen,
                                         int BlkGroupSize,
                                         srcDataType alpha,
                                         const srcDataType* const __restrict__ p_src_global,
                                         dstDataType beta,
                                         dstDataType* const __restrict__ p_dst_global,
                                         srcDataType* const __restrict__ ws_buf1_global,
                                         int* const __restrict__ ws_buf2_global,
                                         int* const __restrict__ indices_global)
        {
            (void)p_dst_global;   // unused
            (void)indices_global; // unused

            using gridwise_reduce =
                GridwiseReduction_xy_to_x_multiblock<BlockSize,
                                                     srcDataType,
                                                     dstDataType,
                                                     compType,
                                                     src2dDescType,
                                                     dst1dDescType,
                                                     op,
                                                     nanPropaOpt,
                                                     reduceIndicesOpt,
                                                     GredAccessesPerThreadInBlock>;
            constexpr int RunId = need_indices ? 2 : 1;
            gridwise_reduce::template Run<RunId>(
                src2dDesc,
                dst1dDesc,
                origReduceLen,
                BlkGroupSize,
                alpha,
                p_src_global,
                beta,
                ws_buf1_global,
                ws_buf2_global); // ws_buf1_global instead of p_dst_global,
                                 // ws_buf2_global instead of indices_global
        };
    };

    template <typename src2dDescType, typename dst1dDescType>
    __device__ void Run(const src2dDescType& src2dDesc,
                        const dst1dDescType& dst1dDesc,
                        float alpha,
                        const void* const __restrict__ p_src_global,
                        float beta,
                        void* const __restrict__ p_dst_global,
                        void* const __restrict__ ws_buf1_global,
                        long ws_buf2_bytes_offset,
                        void* const __restrict__ indices_global) const
    {
        void* const ws_buf2_global =
            ws_buf2_bytes_offset > 0
                ? static_cast<void*>(static_cast<char*>(ws_buf1_global) + ws_buf2_bytes_offset)
                : nullptr;

        using gridwise_2d_reduce_impl = Gridwise2dReduction_impl_wrapper<reduceImpl>;

        gridwise_2d_reduce_impl::RunMethod(
            src2dDesc,
            dst1dDesc,
            this->origReduceLen,
            this->BlkGroupSize,
            type_convert<srcDataType>{}(alpha),
            static_cast<const srcDataType* const __restrict__>(p_src_global),
            type_convert<dstDataType>{}(beta),
            static_cast<dstDataType* const __restrict__>(p_dst_global),
            static_cast<dstDataType* const __restrict__>(ws_buf1_global),
            static_cast<int* const __restrict__>(ws_buf2_global),
            static_cast<int* const __restrict__>(indices_global));
    };

    int origReduceLen;
    int BlkGroupSize;
};

} // namespace ck
#endif
