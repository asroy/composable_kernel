#ifndef CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "constant_tensor_descriptor.hpp"
#include "constant_merged_tensor_descriptor.hpp"

#ifndef CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1 0
#endif

namespace ck {

// user need to make sure alignment requirement is satisfied when setting DataPerAccesss > 1
template <class Float,
          class SrcDesc,
          class DstDesc,
          class SliceLengths,
          class DimAccessOrder,
          index_t DataPerAccess>
__device__ void threadwise_generic_tensor_slice_copy_v1(
    SrcDesc,
    const Float* __restrict__ p_src,
    Array<index_t, SrcDesc::GetNumOfDimension()> src_multi_id_begin,
    DstDesc,
    Float* __restrict__ p_dst,
    Array<index_t, DstDesc::GetNumOfDimension()> dst_multi_id_begin,
    SliceLengths,
    DimAccessOrder,
    Number<DataPerAccess>)
{
    constexpr index_t nDim = SrcDesc::GetNumOfDimension();

    static_assert(nDim == SrcDesc::GetNumOfDimension() && nDim == DstDesc::GetNumOfDimension() &&
                      nDim == SliceLengths::GetSize() && nDim == DimAccessOrder::GetSize(),
                  "wrong! # of dimensions not the same");

    static_assert(is_valid_sequence_map<DimAccessOrder>::value, "wrong! map is not valid");

    // TODO: do more sanity-check here, something like:
    // constexpr auto src_strides_in_access_order =
    //     SrcDesc::ReorderGivenNew2Old(DimAccessOrder{}).GetStride(Number<nDim-1>{});

    // constexpr auto dst_strides_in_access_order =
    //     SrcDesc::ReorderGivenNew2Old(DimAccessOrder{}).GetStride(Number<nDim-1>{});

    // // check src/dst stride on the lowest access dimension
    // static_assert((DataPerAccess == 1 || src_strides_in_access_order.Back() == 1) &&
    //                   (DataPerAccess == 1 || dst_strides_in_access_order.Back() == 1),
    //               "wrong! src/dst stride on the lowest access dimension needs to be 1 for "
    //               "vectorized read/write");

    constexpr auto slice_lengths_in_access_order =
        SliceLengths::ReorderGivenNew2Old(DimAccessOrder{});

    // check slice length on the lowest access dimension
    static_assert(slice_lengths_in_access_order.Back() % DataPerAccess == 0,
                  "wrong! slice length on the lowest access dimension should be evenly divided by "
                  "DataPerAccess");

    constexpr index_t num_access_on_lowest_access_dimension =
        slice_lengths_in_access_order.Back() / DataPerAccess;

    constexpr auto access_lengths = slice_lengths_in_access_order.Modify(
        Number<nDim - 1>{}, Number<num_access_on_lowest_access_dimension>{});

#if 1
    if(get_block_1d_id() == 0 && get_thread_local_1d_id() == 0)
    {
        print_Sequence("access_lengths: ", access_lengths);
    }
#endif

    using vector_t = typename vector_type<Float, DataPerAccess>::MemoryType;

#if 1
    if(get_block_1d_id() == 0 && get_thread_local_1d_id() == 0)
    {
        printf("src:");
        for(index_t i = 0; i < SliceLengths{}[0]; ++i)
        {
            for(index_t j = 0; j < SliceLengths{}[1]; ++j)
            {
                index_t offset = SrcDesc::GetOffsetFromMultiIndex(i, j);

                printf("%d %d %d %f, ", i, j, offset, p_src[offset]);
            }
        }
        printf("\n");

        printf("dst:");
        for(index_t i = 0; i < SliceLengths{}[0]; ++i)
        {
            for(index_t j = 0; j < SliceLengths{}[1]; ++j)
            {
                index_t offset = DstDesc::GetOffsetFromMultiIndex(i, j);

                printf("%d %d %d %f, ", i, j, offset, p_dst[offset]);
            }
        }
        printf("\n");

        printf("\n");
    }
#endif

#if CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1
    static_ford<decltype(access_lengths)>{}([&](auto access_multi_id) {
        constexpr index_t itmp = access_multi_id.Back() * DataPerAccess;

        constexpr auto data_multi_id_in_access_order =
            access_multi_id.Modify(Number<nDim - 1>{}, Number<itmp>{});

        constexpr auto data_multi_id = reorder_array_given_old2new(
            sequence2array(data_multi_id_in_access_order), DimAccessOrder{});

        const index_t src_index =
            SrcDesc::GetOffsetFromMultiIndex(src_multi_id_begin + data_multi_id);

        const index_t dst_index =
            DstDesc::GetOffsetFromMultiIndex(dst_multi_id_begin + data_multi_id);

        *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
            *reinterpret_cast<const vector_t*>(&p_src[src_index]);
    });
#else
    ford<decltype(access_lengths)>{}([&](auto access_multi_id) {
        auto data_multi_id_in_access_order      = access_multi_id;
        data_multi_id_in_access_order(nDim - 1) = access_multi_id[nDim - 1] * DataPerAccess;

        const auto data_multi_id =
            reorder_array_given_old2new(data_multi_id_in_access_order, DimAccessOrder{});

        const index_t src_index =
            SrcDesc::GetOffsetFromMultiIndex(src_multi_id_begin + data_multi_id);

        const index_t dst_index =
            DstDesc::GetOffsetFromMultiIndex(dst_multi_id_begin + data_multi_id);

        *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
            *reinterpret_cast<const vector_t*>(&p_src[src_index]);
#if 1
        if(get_block_1d_id() == 0 && get_thread_local_1d_id() == 0)
        {
            printf("src_index %d, dst_index %d\n", src_index, dst_index);

            printf("src:");
            for(index_t i = 0; i < SliceLengths{}[0]; ++i)
            {
                for(index_t j = 0; j < SliceLengths{}[1]; ++j)
                {
                    index_t offset = SrcDesc::GetOffsetFromMultiIndex(i, j);

                    printf("%d %d %d %f, ", i, j, offset, p_src[offset]);
                }
            }
            printf("\n");

            printf("dst:");
            for(index_t i = 0; i < SliceLengths{}[0]; ++i)
            {
                for(index_t j = 0; j < SliceLengths{}[1]; ++j)
                {
                    index_t offset = DstDesc::GetOffsetFromMultiIndex(i, j);

                    printf("%d %d %d %f, ", i, j, offset, p_dst[offset]);
                }
            }
            printf("\n");

            printf("\n");
        }
#endif
    });
#endif

#if 1
    if(get_block_1d_id() == 0 && get_thread_local_1d_id() == 0)
    {
        printf("src:");
        for(index_t i = 0; i < SliceLengths{}[0]; ++i)
        {
            for(index_t j = 0; j < SliceLengths{}[1]; ++j)
            {
                index_t offset = SrcDesc::GetOffsetFromMultiIndex(i, j);

                printf("%d %d %d %f, ", i, j, offset, p_src[offset]);
            }
        }
        printf("\n");

        printf("dst:");
        for(index_t i = 0; i < SliceLengths{}[0]; ++i)
        {
            for(index_t j = 0; j < SliceLengths{}[1]; ++j)
            {
                index_t offset = DstDesc::GetOffsetFromMultiIndex(i, j);

                printf("%d %d %d %f, ", i, j, offset, p_dst[offset]);
            }
        }
        printf("\n");

        printf("\n");
    }
#endif
}

// user need to make sure alignment requirement is satisfied when setting SrcDataPerAccesss > 1 or
// DstDataPerAccess > 1
template <class SrcTensor,         // src tensor view
          class DstTensor,         // dst tensor view
          class SrcDimAccessOrder, // Sequence
          class DstDimAccessOrder, // Sequence
          index_t SrcDimVectorAccess,
          index_t DstDimVectorAccess,
          index_t SrcDataPerAccess,
          index_t DstDataPerAccess>
struct ThreadwiseTensorCopy_v2
{
    static constexpr index_t nDim = SrcTensor::GetNumOfDimension();
    using DataType                = typename SrcTensor::DataType;

    __device__ ThreadwiseTensorCopy_v2(SrcTensor src_tensor, DstTensor dst_tensor)
        : mBufferTensor(mpBufferData)
    {
        static_assert(is_same<typename SrcTensor::DataType, typename DstTensor::DataType>,
                      "wrong! src and dst should have the same data type");

        static_assert(
            nDim == SrcTensor::GetNumOfDimension() && nDim == DstTensor::GetNumOfDimension() &&
                nDim == SrcDimAccessOrder::GetSize() && nDim == DstDimAccessOrder::GetSize(),
            "wrong! # of dimensions should be the same");

        static_assert(is_same<decltype(SrcTensor::GetLengths()), decltype(DstTensor::GetLengths())>,
                      "wrong! src and dst should have same lengths on all dimension");

        static_assert(is_valid_sequence_map<SrcDimAccessOrder>::value &&
                          is_valid_sequence_map<DstDimAccessOrder>::value,
                      "wrong! src or dst dimension-access-order is not valid map");

        static_assert(SrcTensor::IsVectorAccessAllowed(Number<SrcDimVectorAccess>{},
                                                       Number<SrcDataOerAccess>{}) &&
                          DstTensor::IsVectorAccessAllowed(Number<DstDimVectorAccess>{},
                                                           Number<DstDataPerAccess>{}),
                      "wrong! src or dst vector access is not allowed");
    }

    __device__ static constexpr auto GetSrcAccessLengths()
    {
        return SliceLengths::Modify(Number<SrcDimVectorAccess>{},
                                    SliceLengths{}[SrcDimVectorAccess] / SrcDataPerAccess)
            .ReorderGiveNew2Old(SrcDimAddccessOrder{});
    }

    __device__ static constexpr auto GetDstAccessLengths() {}

    // read data from src into buffer
    __device__ void RunRead(const Float* __restrict__ p_src)
    {
        using vector_t = typename vector_type<TData, SrcDataPerAccess>::MemoryType;

        static_ford<decltype(GetSrcAccessLengths())>([&](auto access_id) {
            constexpr auto data_id = access_id

                src_tensor[]
        });
    }

    // write data from buffer into dst
    __device__ void RunWrite(Float* __restrict__ p_dst) const {}

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst)
    {
        RunRead(p_src);
        RunWrite(p_dst);
    }

    private:
    __device__ static constexpr auto GetBufferTensorDescriptor()
    {
        return make_ConstantTensorDescriptor_packed(SrcTensor::GetLengths());
    }

    DataType mpBufferData[SrcTensor::GetElementSize()];
    TensorView<TData, decltype(GetBufferTensorDescriptor())> mBufferTensor;
};

} // namespace ck
#endif
