#ifndef CK_BLOCKWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP
#define CK_BLOCKWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "cluster_descriptor.hpp"
#include "threadwise_dynamic_tensor_slice_transfer.hpp"

namespace ck {

// this version does not have scratch memory issue, which is good, but I don't know why
template <index_t BlockSize,
          typename BlockSrcData,
          typename BlockDstData,
          typename BlockSrcDesc,
          typename BlockDstDesc,
          typename BlockSliceLengths,
          typename ThreadSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDstDimAccessOrder,
          index_t SrcDstVectoReadDim,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          InMemoryDataOperation DstInMemOp,
          index_t SrcDataStride,
          index_t DstDataStride>
struct BlockwiseDynamicTensorSliceTransfer_v1r1
{
    static constexpr index_t nDim =
        remove_reference_t<remove_cv_t<BlockSrcDesc>>::GetNumOfDimension();

    using Index = MultiIndex<nDim>;

    __device__ constexpr BlockwiseDynamicTensorSliceTransfer_v1r1(
        const BlockSrcDesc& block_src_desc,
        const Index& src_block_slice_origin,
        const BlockDstDesc& block_dst_desc,
        const Index& dst_block_slice_origin)
        : threadwise_transfer_(block_src_desc,
                               make_zero_multi_index<nDim>(),
                               block_dst_desc,
                               make_zero_multi_index<nDim>())

    {
        static_assert(
            nDim == remove_reference_t<remove_cv_t<BlockSrcDesc>>::GetNumOfDimension() &&
                nDim == remove_reference_t<remove_cv_t<BlockDstDesc>>::GetNumOfDimension() &&
                nDim == BlockSliceLengths::Size() && nDim == ThreadSliceLengths::Size() &&
                nDim == ThreadClusterLengths::Size() && nDim == ThreadClusterArrangeOrder::Size() &&
                nDim == SrcDstDimAccessOrder::Size(),
            "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(ThreadSliceLengths{} * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(BlockSize >= thread_cluster_desc_.GetElementSize(),
                      "wrong! BlockSize too small");

        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_id =
                thread_cluster_desc_.CalculateClusterIndex(get_thread_local_1d_id());

            const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};

            threadwise_transfer_.SetSrcSliceOrigin(src_block_slice_origin + thread_data_id_begin);
            threadwise_transfer_.SetDstSliceOrigin(dst_block_slice_origin + thread_data_id_begin);
        }
    }

    __device__ void Run(const BlockSrcData* p_block_src, BlockDstData* p_block_dst)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.Run(p_block_src, p_block_dst);
        }
    }

    __device__ void MoveSrcSliceWindow(const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrcSliceWindow(step);
        }
    }

    __device__ void MoveDstSliceWindow(const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveDstSliceWindow(step);
        }
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    using ThreadwiseTransfer = ThreadwiseDynamicTensorSliceTransfer_v1r1<BlockSrcDesc,
                                                                         BlockDstDesc,
                                                                         ThreadSliceLengths,
                                                                         SrcDstDimAccessOrder,
                                                                         SrcDstVectoReadDim,
                                                                         SrcDataPerRead,
                                                                         DstDataPerWrite,
                                                                         SrcAddressSpace,
                                                                         DstAddressSpace,
                                                                         DstInMemOp,
                                                                         SrcDataStride,
                                                                         DstDataStride>;

    ThreadwiseTransfer threadwise_transfer_;
};

// this version has scratch memory issue, due to:
// 1. ThreadwiseDynamicTensorSliceTransfer_v1r1 keeps reference to tensor descriptor
// 2. threadwise_dynamic_tensor_slice_transfer_v1r1 constructs new tensor coordinate
template <index_t BlockSize,
          typename BlockSrcData,
          typename BlockDstData,
          typename BlockSrcDesc,
          typename BlockDstDesc,
          typename BlockSliceLengths,
          typename ThreadSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorReadDim,
          index_t DstVectorWriteDim,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          InMemoryDataOperation DstInMemOp,
          index_t SrcDataStride,
          index_t DstDataStride>
struct BlockwiseDynamicTensorSliceTransfer_v2r1
{
    static constexpr index_t nDim =
        remove_reference_t<remove_cv_t<BlockSrcDesc>>::GetNumOfDimension();

    using Index = MultiIndex<nDim>;

    __device__ constexpr BlockwiseDynamicTensorSliceTransfer_v2r1(
        const BlockSrcDesc& block_src_desc,
        const Index& src_block_slice_origin,
        const BlockDstDesc& block_dst_desc,
        const Index& dst_block_slice_origin)
        : threadwise_read_(block_src_desc,
                           make_zero_multi_index<nDim>(),
                           thread_buffer_desc_,
                           make_zero_multi_index<nDim>()),
          threadwise_write_(thread_buffer_desc_,
                            make_zero_multi_index<nDim>(),
                            block_dst_desc,
                            make_zero_multi_index<nDim>())

    {
        static_assert(
            nDim == remove_reference_t<remove_cv_t<BlockSrcDesc>>::GetNumOfDimension() &&
                nDim == remove_reference_t<remove_cv_t<BlockDstDesc>>::GetNumOfDimension() &&
                nDim == BlockSliceLengths::Size() && nDim == ThreadSliceLengths::Size() &&
                nDim == ThreadClusterLengths::Size() && nDim == ThreadClusterArrangeOrder::Size() &&
                nDim == SrcDimAccessOrder::Size() && nDim == DstDimAccessOrder::Size(),
            "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(ThreadSliceLengths{} * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(BlockSize >= thread_cluster_desc_.GetElementSize(),
                      "wrong! BlockSize too small");

        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_id =
                thread_cluster_desc_.CalculateClusterIndex(get_thread_local_1d_id());

            const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};

            threadwise_read_.SetSrcSliceOrigin(src_block_slice_origin + thread_data_id_begin);
            threadwise_read_.SetDstSliceOrigin(make_zero_multi_index<nDim>());

            threadwise_write_.SetSrcSliceOrigin(make_zero_multi_index<nDim>());
            threadwise_write_.SetDstSliceOrigin(dst_block_slice_origin + thread_data_id_begin);
        }
    }

    __device__ void RunRead(const BlockSrcData* p_block_src)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_read_.Run(p_block_src, p_thread_buffer_);
        }
    }

    __device__ void RunWrite(BlockDstData* p_block_dst)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_write_.Run(p_thread_buffer_, p_block_dst);
        }
    }

    __device__ void Run(const BlockSrcData* p_block_src, BlockDstData* p_block_dst)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_read_.Run(p_block_src, p_thread_buffer_);

            // if there is type conversion, it's done during write
            threadwise_write_.Run(p_thread_buffer_, p_block_dst);
        }
    }

    __device__ void MoveSrcSliceWindow(const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_read_.MoveSrcSliceWindow(step);
        }
    }

    __device__ void MoveDstSliceWindow(const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_write_.MoveDstSliceWindow(step);
        }
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    static constexpr auto thread_buffer_desc_ =
        make_dynamic_naive_tensor_descriptor_packed<nDim>(to_multi_index(ThreadSliceLengths{}));

    using ThreadwiseRead = ThreadwiseDynamicTensorSliceTransfer_v1r1<BlockSrcDesc,
                                                                     decltype(thread_buffer_desc_),
                                                                     ThreadSliceLengths,
                                                                     SrcDimAccessOrder,
                                                                     SrcVectorReadDim,
                                                                     SrcDataPerRead,
                                                                     1,
                                                                     SrcAddressSpace,
                                                                     AddressSpace::Vgpr,
                                                                     InMemoryDataOperation::Set,
                                                                     SrcDataStride,
                                                                     1>;

    using ThreadwiseWrite = ThreadwiseDynamicTensorSliceTransfer_v1r1<decltype(thread_buffer_desc_),
                                                                      BlockDstDesc,
                                                                      ThreadSliceLengths,
                                                                      DstDimAccessOrder,
                                                                      DstVectorWriteDim,
                                                                      1,
                                                                      DstDataPerWrite,
                                                                      AddressSpace::Vgpr,
                                                                      DstAddressSpace,
                                                                      DstInMemOp,
                                                                      1,
                                                                      DstDataStride>;

    ThreadwiseRead threadwise_read_;
    ThreadwiseWrite threadwise_write_;

    static constexpr index_t thread_buffer_element_size_ =
        thread_buffer_desc_.GetElementSpaceSize();

    BlockSrcData p_thread_buffer_[thread_buffer_element_size_];
};

// this version does not have scratch memory issue, due to:
// 1. ThreadwiseDynamicTensorSliceTransfer_v1r2 does not keep reference to tensor descriptor
// 2. threadwise_dynamic_tensor_slice_transfer_v1r2 does not construct new tensor coordinate
template <index_t BlockSize,
          typename BlockSrcData,
          typename BlockDstData,
          typename BlockSrcDesc,
          typename BlockDstDesc,
          typename BlockSliceLengths,
          typename ThreadSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorReadDim,
          index_t DstVectorWriteDim,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          InMemoryDataOperation DstInMemOp,
          index_t SrcDataStride,
          index_t DstDataStride>
struct BlockwiseDynamicTensorSliceTransfer_v2r2
{
    static constexpr index_t nDim =
        remove_reference_t<remove_cv_t<BlockSrcDesc>>::GetNumOfDimension();

    using Index = MultiIndex<nDim>;

    __device__ constexpr BlockwiseDynamicTensorSliceTransfer_v2r2(
        const BlockSrcDesc& block_src_desc,
        const Index& src_block_slice_origin,
        const BlockDstDesc& block_dst_desc,
        const Index& dst_block_slice_origin)
        : threadwise_read_(block_src_desc,
                           make_zero_multi_index<nDim>(),
                           thread_buffer_desc_,
                           make_zero_multi_index<nDim>()),
          threadwise_write_(thread_buffer_desc_,
                            make_zero_multi_index<nDim>(),
                            block_dst_desc,
                            make_zero_multi_index<nDim>())

    {
        static_assert(
            nDim == remove_reference_t<remove_cv_t<BlockSrcDesc>>::GetNumOfDimension() &&
                nDim == remove_reference_t<remove_cv_t<BlockDstDesc>>::GetNumOfDimension() &&
                nDim == BlockSliceLengths::Size() && nDim == ThreadSliceLengths::Size() &&
                nDim == ThreadClusterLengths::Size() && nDim == ThreadClusterArrangeOrder::Size() &&
                nDim == SrcDimAccessOrder::Size() && nDim == DstDimAccessOrder::Size(),
            "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(ThreadSliceLengths{} * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(BlockSize >= thread_cluster_desc_.GetElementSize(),
                      "wrong! BlockSize too small");

        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_id =
                thread_cluster_desc_.CalculateClusterIndex(get_thread_local_1d_id());

            const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};

            threadwise_read_.SetSrcSliceOrigin(block_src_desc,
                                               src_block_slice_origin + thread_data_id_begin);
            threadwise_read_.SetDstSliceOrigin(thread_buffer_desc_, make_zero_multi_index<nDim>());

            threadwise_write_.SetSrcSliceOrigin(thread_buffer_desc_, make_zero_multi_index<nDim>());
            threadwise_write_.SetDstSliceOrigin(block_dst_desc,
                                                dst_block_slice_origin + thread_data_id_begin);
        }
    }

    __device__ void RunRead(const BlockSrcDesc& block_src_desc, const BlockSrcData* p_block_src)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_read_.Run(
                block_src_desc, p_block_src, thread_buffer_desc_, p_thread_buffer_);
        }
    }

    __device__ void RunWrite(const BlockDstDesc& block_dst_desc, BlockDstData* p_block_dst)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_write_.Run(
                thread_buffer_desc_, p_thread_buffer_, block_dst_desc, p_block_dst);
        }
    }

    __device__ void Run(const BlockSrcDesc& block_src_desc,
                        const BlockSrcData* p_block_src,
                        const BlockDstDesc& block_dst_desc,
                        BlockDstData* p_block_dst)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_read_.Run(
                block_src_desc, p_block_src, thread_buffer_desc_, p_thread_buffer_);

            // if there is type conversion, it's done during write
            threadwise_write_.Run(
                thread_buffer_desc_, p_thread_buffer_, block_dst_desc, p_block_dst);
        }
    }

    __device__ void MoveSrcSliceWindow(const BlockSrcDesc& block_src_desc, const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_read_.MoveSrcSliceWindow(block_src_desc, step);
        }
    }

    __device__ void MoveDstSliceWindow(const BlockDstDesc& block_dst_desc, const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_write_.MoveDstSliceWindow(block_dst_desc, step);
        }
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    static constexpr auto thread_buffer_desc_ =
        make_dynamic_naive_tensor_descriptor_packed<nDim>(to_multi_index(ThreadSliceLengths{}));

    using ThreadwiseRead = ThreadwiseDynamicTensorSliceTransfer_v1r2<BlockSrcDesc,
                                                                     decltype(thread_buffer_desc_),
                                                                     ThreadSliceLengths,
                                                                     SrcDimAccessOrder,
                                                                     SrcVectorReadDim,
                                                                     SrcDataPerRead,
                                                                     1,
                                                                     SrcAddressSpace,
                                                                     AddressSpace::Vgpr,
                                                                     InMemoryDataOperation::Set,
                                                                     SrcDataStride,
                                                                     1>;

    using ThreadwiseWrite = ThreadwiseDynamicTensorSliceTransfer_v1r2<decltype(thread_buffer_desc_),
                                                                      BlockDstDesc,
                                                                      ThreadSliceLengths,
                                                                      DstDimAccessOrder,
                                                                      DstVectorWriteDim,
                                                                      1,
                                                                      DstDataPerWrite,
                                                                      AddressSpace::Vgpr,
                                                                      DstAddressSpace,
                                                                      DstInMemOp,
                                                                      1,
                                                                      DstDataStride>;

    ThreadwiseRead threadwise_read_;
    ThreadwiseWrite threadwise_write_;

    static constexpr index_t thread_buffer_element_size_ =
        thread_buffer_desc_.GetElementSpaceSize();

    BlockSrcData p_thread_buffer_[thread_buffer_element_size_];
};

} // namespace ck
#endif
