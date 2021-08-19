/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#include "config.hpp"
#include "number.hpp"
#include "sequence.hpp"
#include "tensor_descriptor_helper.hpp"
#include "reduction_common.hpp"
#include "gridwise_generic_reduction_second_call_method_chooser.hpp"
#include "gridwise_generic_reduction_wrapper_common.hpp"

using namespace ck;

using srcDataType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_SRC_DATATYPE)>::type;
using dstDataType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_DST_DATATYPE)>::type;
using compType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_REDUCE_COMPTYPE)>::type;

constexpr index_t BlockSize = CK_PARAM_BLOCKSIZE; // tunable

constexpr index_t srcDims = CK_PARAM_IN_DIMS;
constexpr index_t dstDims = CK_PARAM_OUT_DIMS;

using toReduceDims  = Sequence<CK_PARAM_TOREDUCE_DIMS>;
using invariantDims = Sequence<CK_PARAM_INVARIANT_DIMS>; // this could be empty

constexpr ReductionMethod_t reduceImpl = static_cast<ReductionMethod_t>(CK_PARAM_REDUCE_IMPL);

constexpr ReduceTensorOp_t op          = get_reduce_op<CK_PARAM_REDUCE_OP>::op;
constexpr NanPropagation_t nanPropaOpt = CK_PARAM_NAN_PROPAGATE == 0
                                             ? NanPropagation_t::NOT_PROPAGATE_NAN
                                             : NanPropagation_t::PROPAGATE_NAN;
constexpr ReduceTensorIndices_t reduceIndicesOpt = CK_PARAM_REDUCE_INDICES == 0
                                                       ? ReduceTensorIndices_t::NO_INDICES
                                                       : ReduceTensorIndices_t::FLATTENED_INDICES;

constexpr bool src2d_need_padding = static_cast<bool>(CK_PARAM_SRC2D_PADDING);
constexpr bool dst1d_need_padding = static_cast<bool>(CK_PARAM_DST1D_PADDING);

////////////////////////////////////////////////////////////////////////////////////////
using specDims = typename sequence_merge<invariantDims, toReduceDims>::type;

static_assert(is_valid_sequence_map<specDims>::value && specDims::Size() == srcDims,
              "Wrong invariant and/or toReduce dimensions!");

// The number of invariant dimensions can be zero if all dimension are to be reduced
static_assert(invariantDims::Size() > 0 || dstDims == 1,
              "If all source dimensions are reduced, the dest should have only one dimension !!");

constexpr bool reduceAllDims = (invariantDims::Size() == 0) ? true : false;

extern "C" __global__ void gridwise_generic_reduce_2_prepare(int GridSize,
                                                             int BlkGroupSize,
                                                             int outLength0,
                                                             int outLength1,
                                                             int outLength2,
                                                             int outLength3,
                                                             int outLength4,
                                                             int outLength5,
                                                             int outStride0,
                                                             int outStride1,
                                                             int outStride2,
                                                             int outStride3,
                                                             int outStride4,
                                                             int outStride5,
                                                             void* p_src2dDesc,
                                                             void* p_dst1dDesc)
{
    const int dstLengths[6] = {
        outLength0, outLength1, outLength2, outLength3, outLength4, outLength5};
    const int dstStrides[6] = {
        outStride0, outStride1, outStride2, outStride3, outStride4, outStride5};

    const auto tupleDstLengths = make_tuple_from_array(dstLengths, Number<dstDims>{});
    const auto tupleDstStrides = make_tuple_from_array(dstStrides, Number<dstDims>{});

    const auto dstDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

    const auto one_dim_dstDesc = transform_tensor_descriptor(
        dstDesc,
        make_tuple(make_merge_transform(tupleDstLengths)),
        make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
        make_tuple(Sequence<0>{}));

    const index_t invariantLen = one_dim_dstDesc.GetLength(Number<0>{});
    const index_t toReduceLen  = BlkGroupSize;

    const auto workspace_2d_desc =
        make_naive_tensor_descriptor_packed(make_tuple(invariantLen, toReduceLen));

    gridwise_generic_reduce_pad_and_store<reduceImpl, src2d_need_padding, dst1d_need_padding>::
        RunMethod(GridSize, 0, workspace_2d_desc, one_dim_dstDesc, p_src2dDesc, p_dst1dDesc);
};

template <index_t srcDims, index_t dstDims, typename invariantDims, typename toReduceDims>
struct get_ref_desc_types
{
    static constexpr auto ref_tupleDstLengths =
        make_tuple_from_seq(typename uniform_sequence_gen<dstDims, 8>::type{});
    static constexpr auto ref_dstDesc =
        make_naive_tensor_descriptor(ref_tupleDstLengths, ref_tupleDstLengths);

    static constexpr auto ref_dst1dDesc = transform_tensor_descriptor(
        ref_dstDesc,
        make_tuple(make_merge_transform(ref_tupleDstLengths)),
        make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
        make_tuple(Sequence<0>{}));

    static constexpr index_t ref_invariantLen = ref_dst1dDesc.GetLength(Number<0>{});
    static constexpr index_t ref_toReduceLen  = 8;

    static constexpr auto ref_src2dDesc =
        make_naive_tensor_descriptor_packed(make_tuple(ref_invariantLen, ref_toReduceLen));

    using refType_src2dDesc = decltype(ref_src2dDesc);
    using refType_dst1dDesc = decltype(ref_dst1dDesc);

    // used by the DirectThreadWise and DirectWarpWise method
    using refType_src2dDesc_padded_12 =
        decltype(transform_tensor_descriptor(ref_src2dDesc,
                                             make_tuple(make_pad_transform(ref_invariantLen, 0, 2),
                                                        make_pad_transform(ref_toReduceLen, 0, 2)),
                                             make_tuple(Sequence<0>{}, Sequence<1>{}),
                                             make_tuple(Sequence<0>{}, Sequence<1>{})));

    // used by the BlockWise and MultiBlock method
    using refType_src2dDesc_padded_34 = decltype(
        transform_tensor_descriptor(ref_src2dDesc,
                                    make_tuple(make_pass_through_transform(ref_invariantLen),
                                               make_pad_transform(ref_toReduceLen, 0, 2)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1>{})));

    using refType_dst1dDesc_padded =
        decltype(transform_tensor_descriptor(ref_dst1dDesc,
                                             make_tuple(make_pad_transform(ref_invariantLen, 0, 2)),
                                             make_tuple(Sequence<0>{}),
                                             make_tuple(Sequence<0>{})));
};

using refType_src2dDesc =
    typename get_ref_desc_types<srcDims, dstDims, invariantDims, toReduceDims>::refType_src2dDesc;
using refType_dst1dDesc =
    typename get_ref_desc_types<srcDims, dstDims, invariantDims, toReduceDims>::refType_dst1dDesc;
using refType_src2dDesc_padded_12 =
    typename get_ref_desc_types<srcDims, dstDims, invariantDims, toReduceDims>::
        refType_src2dDesc_padded_12;
using refType_src2dDesc_padded_34 =
    typename get_ref_desc_types<srcDims, dstDims, invariantDims, toReduceDims>::
        refType_src2dDesc_padded_34;
using refType_dst1dDesc_padded =
    typename get_ref_desc_types<srcDims, dstDims, invariantDims, toReduceDims>::
        refType_dst1dDesc_padded;

template <ReductionMethod_t impl>
static __device__ auto get_reduction_src2d_descriptor(const void* p_src2dDesc);

template <>
__device__ auto
get_reduction_src2d_descriptor<ReductionMethod_t::DirectThreadWise>(const void* p_src2dDesc)
{
    if constexpr(src2d_need_padding)
        return (*reinterpret_cast<const refType_src2dDesc_padded_12*>(p_src2dDesc));
    else
        return (*reinterpret_cast<const refType_src2dDesc*>(p_src2dDesc));
};

template <>
__device__ auto
get_reduction_src2d_descriptor<ReductionMethod_t::DirectWarpWise>(const void* p_src2dDesc)
{
    if constexpr(src2d_need_padding)
        return (*reinterpret_cast<const refType_src2dDesc_padded_12*>(p_src2dDesc));
    else
        return (*reinterpret_cast<const refType_src2dDesc*>(p_src2dDesc));
};

template <>
__device__ auto
get_reduction_src2d_descriptor<ReductionMethod_t::BlockWise>(const void* p_src2dDesc)
{
    if constexpr(src2d_need_padding)
        return (*reinterpret_cast<const refType_src2dDesc_padded_34*>(p_src2dDesc));
    else
        return (*reinterpret_cast<const refType_src2dDesc*>(p_src2dDesc));
};

template <>
__device__ auto
get_reduction_src2d_descriptor<ReductionMethod_t::MultiBlock>(const void* p_src2dDesc)
{
    if constexpr(src2d_need_padding)
        return (*reinterpret_cast<const refType_src2dDesc_padded_34*>(p_src2dDesc));
    else
        return (*reinterpret_cast<const refType_src2dDesc*>(p_src2dDesc));
};

static __device__ auto get_reduction_dst1d_descriptor(const void* p_dst1dDesc)
{
    if constexpr(dst1d_need_padding)
        return (*reinterpret_cast<const refType_dst1dDesc_padded*>(p_dst1dDesc));
    else
        return (*reinterpret_cast<const refType_dst1dDesc*>(p_dst1dDesc));
};

extern "C" __global__ void gridwise_generic_reduce_2(int origReduceLen,
                                                     const void* p_src2dDesc,
                                                     const void* p_dst1dDesc,
                                                     float alpha,
                                                     const void* __restrict__ p_src_global,
                                                     float beta,
                                                     void* __restrict__ p_dst_global,
                                                     void* __restrict__ ws_buf1_global,
                                                     size_t ws_buf2_bytes_offset,
                                                     void* __restrict__ indices_global)
{
    constexpr index_t GredThreadBufferLength = CK_PARAM_THREAD_BUFFER_LENGTH; // tunable
    constexpr index_t GredAccessesPerThreadInBlock =
        CK_PARAM_ACCESSES_PER_THREAD_INBLOCK;                                            // tunable
    constexpr index_t GredAccessesPerThreadInWarp = CK_PARAM_ACCESSES_PER_THREAD_INWARP; // tunable

    const auto src2dDesc = get_reduction_src2d_descriptor<reduceImpl>((const void*)p_src2dDesc);
    const auto dst1dDesc = get_reduction_dst1d_descriptor((const void*)p_dst1dDesc);

    const auto gridwise_2d_reduce = Gridwise2dReduction<BlockSize,
                                                        srcDataType,
                                                        dstDataType,
                                                        compType,
                                                        static_cast<index_t>(reduceImpl),
                                                        static_cast<index_t>(op),
                                                        static_cast<index_t>(nanPropaOpt),
                                                        static_cast<index_t>(reduceIndicesOpt),
                                                        GredThreadBufferLength,
                                                        GredAccessesPerThreadInBlock,
                                                        GredAccessesPerThreadInWarp>(origReduceLen);

    gridwise_2d_reduce.Run(src2dDesc,
                           dst1dDesc,
                           alpha,
                           const_cast<const void* const __restrict__>(p_src_global),
                           beta,
                           const_cast<void* const __restrict__>(p_dst_global),
                           const_cast<void* const __restrict__>(ws_buf1_global),
                           ws_buf2_bytes_offset,
                           const_cast<void* const __restrict__>(indices_global));
};
