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
#ifndef CK_GRIDWISE_GENERIC_REDUCTION_WRAPPER_COMMON
#define CK_GRIDWISE_GENERIC_REDUCTION_WRAPPER_COMMON

#include "config.hpp"
#include "number.hpp"
#include "sequence.hpp"
#include "tensor_descriptor_helper.hpp"
#include "reduction_common.hpp"

namespace ck {

template <char tid>
struct get_type_from_type_id
{
    using type = float;
};

template <>
struct get_type_from_type_id<'H'>
{
    using type = half_t;
};

template <>
struct get_type_from_type_id<'F'>
{
    using type = float;
};

template <>
struct get_type_from_type_id<'D'>
{
    using type = double;
};

template <index_t persistentID>
struct get_reduce_op // any other ID
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::ADD;
};

template <>
struct get_reduce_op<656868> // 'A' * 10000 + 'D' * 100 + 'D'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::ADD;
};

template <>
struct get_reduce_op<778576> // 'M' * 10000 + 'U' * 100 + 'L'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MUL;
};

template <>
struct get_reduce_op<777378> // 'M' * 10000 + 'I' * 100 + 'N'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MIN;
};

template <>
struct get_reduce_op<776588> // 'M' * 10000 + 'A' * 100 + 'X'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MAX;
};

template <>
struct get_reduce_op<657788> // 'A' * 10000 + 'M' * 100 + 'X'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::AMAX;
};

template <>
struct get_reduce_op<658671> // 'A' * 10000 + 'V' * 100 + 'G'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::AVG;
};

template <>
struct get_reduce_op<788201> // 'N' * 10000 + 'R' * 100 + '1'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::NORM1;
};

template <>
struct get_reduce_op<788202> // 'N' * 10000 + 'R' * 100 + '2'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::NORM2;
};

template <index_t... Ns>
__device__ static auto make_tuple_from_array_and_index_seq(const int* lengths, Sequence<Ns...>)
{
    return make_tuple(static_cast<index_t>(lengths[Ns])...);
};

template <index_t arraySize>
__device__ static auto make_tuple_from_array(const int* lengths, Number<arraySize>)
{
    static_assert(arraySize >= 1 && arraySize <= 6, "The tensor should have 1 to 6 dimensions");

    constexpr auto index_seq = typename arithmetic_sequence_gen<0, arraySize, 1>::type{};

    return make_tuple_from_array_and_index_seq(lengths, index_seq);
};

template <index_t... Ids>
__device__ static auto make_passthrough_tuple_from_array_and_index_seq(const int* lengths,
                                                                       Sequence<Ids...>)
{
    return make_tuple(make_pass_through_transform(static_cast<index_t>(lengths[Ids]))...);
};

template <index_t... Ns>
__device__ static constexpr auto make_tuple_from_seq(Sequence<Ns...>)
{
    return make_tuple(Ns...);
};

template <index_t... Ns>
__device__ static constexpr auto make_dimensions_tuple(Sequence<Ns...>)
{
    return make_tuple(Sequence<Ns>{}...);
};

template <index_t... Ns>
__device__ static constexpr auto make_passthrough_tuple_from_seq(Sequence<Ns...>)
{
    return make_tuple(make_pass_through_transform(Ns)...);
};

template <ReductionMethod_t impl, bool src_need_padding, bool dst_need_padding>
struct gridwise_generic_reduce_pad_and_store;

template <bool src_need_padding, bool dst_need_padding>
struct gridwise_generic_reduce_pad_and_store<ReductionMethod_t::DirectThreadWise,
                                             src_need_padding,
                                             dst_need_padding>
{
    static constexpr index_t GredThreadBufferLength = CK_PARAM_THREAD_BUFFER_LENGTH; // tunable
    static constexpr index_t BlockSize              = CK_PARAM_BLOCKSIZE;            // tunable

    template <typename src2dDescType, typename dst1dDescType>
    __device__ static inline void RunMethod(int GridSize,
                                            int BlkGroupSize,
                                            const src2dDescType& src2dDesc,
                                            const dst1dDescType& dst1dDesc,
                                            void* p_src2dDesc,
                                            void* p_dst1dDesc)
    {
        (void)BlkGroupSize;

        const auto invariantLen = src2dDesc.GetLength(Number<0>{});
        const auto toReduceLen  = src2dDesc.GetLength(Number<1>{});

        constexpr auto copySliceLen = GredThreadBufferLength;

        if constexpr(src_need_padding)
        {
            const auto srcPad1 = GridSize * BlockSize - invariantLen;
            const auto srcPad2 =
                ((toReduceLen + copySliceLen - 1) / copySliceLen) * copySliceLen - toReduceLen;
            auto src2dDesc_2 =
                transform_tensor_descriptor(src2dDesc,
                                            make_tuple(make_pad_transform(invariantLen, 0, srcPad1),
                                                       make_pad_transform(toReduceLen, 0, srcPad2)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));
            if(hipThreadIdx_x == 0)
                *static_cast<decltype(src2dDesc_2)*>(p_src2dDesc) = src2dDesc_2;
        }
        else
        {
            if(hipThreadIdx_x == 0)
                *static_cast<src2dDescType*>(p_src2dDesc) = src2dDesc;
        }

        if constexpr(dst_need_padding)
        {
            const auto dstPad = GridSize * BlockSize - invariantLen;
            auto dst1dDesc_2 =
                transform_tensor_descriptor(dst1dDesc,
                                            make_tuple(make_pad_transform(invariantLen, 0, dstPad)),
                                            make_tuple(Sequence<0>{}),
                                            make_tuple(Sequence<0>{}));
            if(hipThreadIdx_x == 0)
                *static_cast<decltype(dst1dDesc_2)*>(p_dst1dDesc) = dst1dDesc_2;
        }
        else
        {
            if(hipThreadIdx_x == 0)
                *static_cast<dst1dDescType*>(p_dst1dDesc) = dst1dDesc;
        }
    };
};

template <bool src_need_padding, bool dst_need_padding>
struct gridwise_generic_reduce_pad_and_store<ReductionMethod_t::DirectWarpWise,
                                             src_need_padding,
                                             dst_need_padding>
{
    static constexpr index_t GredAccessesPerThreadInWarp =
        CK_PARAM_ACCESSES_PER_THREAD_INWARP;                 // tunable
    static constexpr index_t BlockSize = CK_PARAM_BLOCKSIZE; // tunable

    template <typename src2dDescType, typename dst1dDescType>
    __device__ static inline void RunMethod(int GridSize,
                                            int BlkGroupSize,
                                            const src2dDescType& src2dDesc,
                                            const dst1dDescType& dst1dDesc,
                                            void* p_src2dDesc,
                                            void* p_dst1dDesc)
    {
        (void)BlkGroupSize;

        const auto invariantLen = src2dDesc.GetLength(Number<0>{});
        const auto toReduceLen  = src2dDesc.GetLength(Number<1>{});

        constexpr auto copySliceLen = warpSize * GredAccessesPerThreadInWarp;

        if constexpr(src_need_padding)
        {
            const auto srcPad1 = GridSize * BlockSize / warpSize - invariantLen;
            const auto srcPad2 =
                ((toReduceLen + copySliceLen - 1) / copySliceLen) * copySliceLen - toReduceLen;

            auto src2dDesc_2 =
                transform_tensor_descriptor(src2dDesc,
                                            make_tuple(make_pad_transform(invariantLen, 0, srcPad1),
                                                       make_pad_transform(toReduceLen, 0, srcPad2)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));
            if(hipThreadIdx_x == 0)
                *static_cast<decltype(src2dDesc_2)*>(p_src2dDesc) = src2dDesc_2;
        }
        else
        {
            if(hipThreadIdx_x == 0)
                *static_cast<src2dDescType*>(p_src2dDesc) = src2dDesc;
        }

        if constexpr(dst_need_padding)
        {
            const auto dstPad = GridSize * BlockSize / warpSize - invariantLen;
            auto dst1dDesc_2 =
                transform_tensor_descriptor(dst1dDesc,
                                            make_tuple(make_pad_transform(invariantLen, 0, dstPad)),
                                            make_tuple(Sequence<0>{}),
                                            make_tuple(Sequence<0>{}));
            if(hipThreadIdx_x == 0)
                *static_cast<decltype(dst1dDesc_2)*>(p_dst1dDesc) = dst1dDesc_2;
        }
        else
        {
            if(hipThreadIdx_x == 0)
                *static_cast<dst1dDescType*>(p_dst1dDesc) = dst1dDesc;
        }
    };
};

template <bool src_need_padding, bool dst_need_padding>
struct gridwise_generic_reduce_pad_and_store<ReductionMethod_t::BlockWise,
                                             src_need_padding,
                                             dst_need_padding>
{
    static constexpr index_t GredAccessesPerThreadInBlock =
        CK_PARAM_ACCESSES_PER_THREAD_INBLOCK;                // tunable
    static constexpr index_t BlockSize = CK_PARAM_BLOCKSIZE; // tunable

    template <typename src2dDescType, typename dst1dDescType>
    __device__ static inline void RunMethod(int GridSize,
                                            int BlkGroupSize,
                                            const src2dDescType& src2dDesc,
                                            const dst1dDescType& dst1dDesc,
                                            void* p_src2dDesc,
                                            void* p_dst1dDesc)
    {
        (void)GridSize;
        (void)BlkGroupSize;

        const auto invariantLen = src2dDesc.GetLength(Number<0>{});
        const auto toReduceLen  = src2dDesc.GetLength(Number<1>{});

        constexpr auto copySliceLen = BlockSize * GredAccessesPerThreadInBlock;

        if constexpr(src_need_padding)
        {
            const auto srcPad =
                ((toReduceLen + copySliceLen - 1) / copySliceLen) * copySliceLen - toReduceLen;

            auto src2dDesc_2 =
                transform_tensor_descriptor(src2dDesc,
                                            make_tuple(make_pass_through_transform(invariantLen),
                                                       make_pad_transform(toReduceLen, 0, srcPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));
            if(hipThreadIdx_x == 0)
                *static_cast<decltype(src2dDesc_2)*>(p_src2dDesc) = src2dDesc_2;
        }
        else
        {
            if(hipThreadIdx_x == 0)
                *static_cast<src2dDescType*>(p_src2dDesc) = src2dDesc;
        }

        if(hipThreadIdx_x == 0)
            *static_cast<dst1dDescType*>(p_dst1dDesc) = dst1dDesc;
    };
};

template <bool src_need_padding, bool dst_need_padding>
struct gridwise_generic_reduce_pad_and_store<ReductionMethod_t::MultiBlock,
                                             src_need_padding,
                                             dst_need_padding>
{
    static constexpr index_t GredAccessesPerThreadInBlock =
        CK_PARAM_ACCESSES_PER_THREAD_INBLOCK;                // tunable
    static constexpr index_t BlockSize = CK_PARAM_BLOCKSIZE; // tunable

    template <typename src2dDescType, typename dst1dDescType>
    __device__ static inline void RunMethod(int GridSize,
                                            int BlkGroupSize,
                                            const src2dDescType& src2dDesc,
                                            const dst1dDescType& dst1dDesc,
                                            void* p_src2dDesc,
                                            void* p_dst1dDesc)
    {
        (void)GridSize;

        const auto invariantLen = src2dDesc.GetLength(Number<0>{});
        const auto toReduceLen  = src2dDesc.GetLength(Number<1>{});

        constexpr auto copySliceLen = BlockSize * GredAccessesPerThreadInBlock;
        const index_t reduceSizePerBlock =
            (((toReduceLen + BlkGroupSize - 1) / BlkGroupSize + copySliceLen - 1) / copySliceLen) *
            copySliceLen;

        if constexpr(src_need_padding)
        {
            const auto srcPad = reduceSizePerBlock * BlkGroupSize - toReduceLen;

            auto src2dDesc_2 =
                transform_tensor_descriptor(src2dDesc,
                                            make_tuple(make_pass_through_transform(invariantLen),
                                                       make_pad_transform(toReduceLen, 0, srcPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));
            if(hipThreadIdx_x == 0)
                *static_cast<decltype(src2dDesc_2)*>(p_src2dDesc) = src2dDesc_2;
        }
        else
        {
            if(hipThreadIdx_x == 0)
                *static_cast<src2dDescType*>(p_src2dDesc) = src2dDesc;
        }

        if(hipThreadIdx_x == 0)
            *static_cast<dst1dDescType*>(p_dst1dDesc) = dst1dDesc;
    };
};

}; // end of namespace ck

#endif
