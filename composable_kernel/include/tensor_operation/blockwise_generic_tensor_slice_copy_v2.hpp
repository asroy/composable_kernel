#ifndef CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_V2_HPP
#define CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_V2_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_coordinate.hpp"
#include "cluster_descriptor.hpp"
#include "threadwise_generic_tensor_slice_copy_v2.hpp"

namespace ck {

// This blockwise copy allow vector access of src and dst.
// It allows the vector size to be different on src and dst.
// The dimension of vector access can be different for src and dst.
// The dimension access order can be different for src and dst.
// Will do valid mapping check on src data: Read 0 if src data has a invalid mapping
// Will do valid mapping check on dst data: No write if dst data has a invalid mapping
// BlockSize can be equal or larger than ThreadCluster size, which means some threads may not do
// threadwise copy
template <index_t BlockSize,
          typename BlockSrcDesc,
          typename BlockDstDesc,
          typename BlockSliceLengths,
          typename ThreadSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectoReadDim,
          index_t DstVectorWriteDim,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite,
          AddressSpace SrcAddressSpace          = AddressSpace::Generic,
          AddressSpace ThreadBufferAddressSpace = AddressSpace::Generic,
          AddressSpace DstAddressSpace          = AddressSpace::Generic,
          InMemoryDataOperation DstInMemOp      = InMemoryDataOperation::Set,
          index_t SrcDataStride                 = 1,
          index_t DstDataStride                 = 1>
struct BlockwiseGenericTensorSliceCopy_v5
{
    static constexpr index_t nDim = BlockSrcDesc::GetNumOfDimension();
    using Index                   = MultiIndex<nDim>;

    __device__ constexpr BlockwiseGenericTensorSliceCopy_v5(const Index& src_block_slice_origin,
                                                            const Index& dst_block_slice_origin)
    {
        static_assert(nDim == BlockSrcDesc::GetNumOfDimension() &&
                          nDim == BlockDstDesc::GetNumOfDimension() &&
                          nDim == BlockSliceLengths::Size() && nDim == ThreadSliceLengths::Size() &&
                          nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == SrcDimAccessOrder::Size() && nDim == DstDimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(ThreadSliceLengths{} * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(BlockSize >= mThreadClusterDesc.GetElementSize(),
                      "wrong! BlockSize too small");

        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            const auto thread_cluster_id =
                mThreadClusterDesc.CalculateClusterIndex(get_thread_local_1d_id());

            const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};

            mThreadwiseCopy.SetSrcSliceOrigin(src_block_slice_origin + thread_data_id_begin);
            mThreadwiseCopy.SetDstSliceOrigin(dst_block_slice_origin + thread_data_id_begin);
        }
    }

    __device__ static constexpr index_t GetThreadBufferSize()
    {
        return ThreadBufferDesc::GetElementSpace();
    }

    template <typename BlockSrcData>
    __device__ void RunLoadThreadBuffer(const BlockSrcData* p_block_src)
    {
        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            mThreadwiseCopy.Load(p_block_src);
        }
    }

    template <typename BlockDstData>
    __device__ void RunStoreThreadBuffer(BlockDstData* p_block_dst)
    {
        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            mThreadwiseCopy.Store(p_block_dst);
        }
    }

    template <typename BlockSrcData, typename BlockDstData>
    __device__ void Run(const BlockSrcData* p_block_src, BlockDstData* p_block_dst)
    {
        static_assert(ThreadBufferAddressSpace == AddressSpace::Vgpr,
                      "wrong! This function use vgpr as its thread "
                      "buffer. However, you have set RunLoadThreadBuffer and RunStoreThreadBuffer "
                      "to use ThreadBufferAddressSpace as their thread buffer, which is not vgpr. "
                      "Behavior may be different");

        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            RunLoadThreadBuffer(p_block_src);
            RunStoreThreadBuffer(p_block_dst);
        }
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveSrcSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            mThreadwiseCopy.MoveSrcSliceWindow(step_sizes, positive_direction);
        }
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveDstSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        if(BlockSize == mThreadClusterDesc.GetElementSize() or
           get_thread_local_1d_id() < mThreadClusterDesc.GetElementSize())
        {
            mThreadwiseCopy.MoveDstSliceWindow(step_sizes, positive_direction);
        }
    }

    private:
    using ThreadBufferDesc = decltype(make_native_tensor_descriptor_packed(ThreadSliceLengths{}));

    using ThreadwiseCopy = ThreadwiseGenericTensorSliceCopy_v5<BlockSrcDesc,
                                                               BlockDstDesc,
                                                               ThreadSliceLengths,
                                                               SrcDimAccessOrder,
                                                               SrcVectoReadDim,
                                                               SrcDataPerRead,
                                                               DstDataPerWrite,
                                                               SrcAddressSpace,
                                                               DstAddressSpace,
                                                               InMemoryDataOperation::Set,
                                                               SrcDataStride,
                                                               DstDataStride>;

    static constexpr auto mThreadClusterDesc =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    ThreadwiseCopy mThreadwiseCopy;
};

} // namespace ck

#endif
