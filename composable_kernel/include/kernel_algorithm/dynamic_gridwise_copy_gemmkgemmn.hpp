#ifndef CK_DYNAMIC_GRIDWISE_COPY_GEMMKGEMMN_HPP
#define CK_DYNAMIC_GRIDWISE_COPY_GEMMKGEMMN_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "blockwise_dynamic_tensor_slice_transfer.hpp"
#include "threadwise_dynamic_tensor_slice_transfer.hpp"

namespace ck {

template <index_t BlockSize,
          index_t GemmKPerBlock,
          index_t GemmNPerBlock,
          typename BlockCopySubLengths_GemmK_GemmN,
          typename BlockCopyClusterLengths_GemmK_GemmN,
          typename BlockCopyThreadClusterArrangeOrder,
          typename BlockCopySrcAccessOrder,
          typename BlockCopyDstAccessOrder,
          index_t BlockCopyDataPerAccess_GemmN>
struct DynamicGridwiseCopy_gemmkgemmn
{
#if 1
    template <typename... Src, typename... Dst>
    __device__ void Run(const float* const __restrict__ p_src_global,
                        float* const __restrict__ p_dst_global,
                        const DynamicTensorDescriptor<Src...>& src_gemmk_gemmn_global_desc,
                        const DynamicTensorDescriptor<Dst...>& dst_gemmk_gemmn_global_desc) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        const index_t GemmK = src_gemmk_gemmn_global_desc.GetLength(I0);
        const index_t GemmN = src_gemmk_gemmn_global_desc.GetLength(I1);

        // divide block work by GemmN
        const index_t GemmNBlockWork = GemmN / GemmNPerBlock;

        const index_t block_work_id = get_block_1d_id();

        const index_t gemmn_block_data_on_global = block_work_id * GemmNPerBlock;

        // blockwise atomic accumulation
        auto blockwise_copy =
#if 0
            BlockwiseDynamicTensorSliceTransfer_v1<BlockSize,
                                                   float,
                                                   float,
                                                   decltype(src_gemmk_gemmn_global_desc),
                                                   decltype(dst_gemmk_gemmn_global_desc),
                                                   Sequence<GemmKPerBlock, GemmNPerBlock>,
                                                   BlockCopySubLengths_GemmK_GemmN,
                                                   BlockCopyClusterLengths_GemmK_GemmN,
                                                   BlockCopyThreadClusterArrangeOrder,
                                                   BlockCopySrcAccessOrder,
                                                   1,
                                                   BlockCopyDataPerAccess_GemmN,
                                                   BlockCopyDataPerAccess_GemmN,
                                                   AddressSpace::Global,
                                                   AddressSpace::Global,
                                                   InMemoryDataOperation::Set,
                                                   1,
                                                   1>(
#else
            BlockwiseDynamicTensorSliceTransfer_v2<BlockSize,
                                                   float,
                                                   float,
                                                   decltype(src_gemmk_gemmn_global_desc),
                                                   decltype(dst_gemmk_gemmn_global_desc),
                                                   Sequence<GemmKPerBlock, GemmNPerBlock>,
                                                   BlockCopySubLengths_GemmK_GemmN,
                                                   BlockCopyClusterLengths_GemmK_GemmN,
                                                   BlockCopyThreadClusterArrangeOrder,
                                                   BlockCopySrcAccessOrder,
                                                   BlockCopyDstAccessOrder,
                                                   1,
                                                   1,
                                                   BlockCopyDataPerAccess_GemmN,
                                                   BlockCopyDataPerAccess_GemmN,
                                                   AddressSpace::Global,
                                                   AddressSpace::Global,
                                                   InMemoryDataOperation::Set,
                                                   1,
                                                   1>(
#endif
                src_gemmk_gemmn_global_desc,
                make_multi_index(0, gemmn_block_data_on_global),
                dst_gemmk_gemmn_global_desc,
                make_multi_index(0, gemmn_block_data_on_global));

        for(index_t gemmk = 0; gemmk < GemmK; gemmk += GemmKPerBlock)
        {
            blockwise_copy.Run(p_src_global, p_dst_global);

            blockwise_copy.MoveSrcSliceWindow(make_multi_index(GemmKPerBlock, 0));
            blockwise_copy.MoveDstSliceWindow(make_multi_index(GemmKPerBlock, 0));
        }
    }
#else
    template <typename... Src, typename... Dst>
    __device__ void Run(const float* const __restrict__ p_src_global,
                        float* const __restrict__ p_dst_global,
                        const DynamicTensorDescriptor<Src...>& src_gemmk_gemmn_global_desc,
                        const DynamicTensorDescriptor<Dst...>& dst_gemmk_gemmn_global_desc) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        const index_t GemmK = src_gemmk_gemmn_global_desc.GetLength(I0);
        const index_t GemmN = src_gemmk_gemmn_global_desc.GetLength(I1);

        // divide block work by GemmN
        const index_t GemmNBlockWork = GemmN / GemmNPerBlock;

        const index_t block_work_id = get_block_1d_id();

        // divide thread work by GemmK, GemmN
        static constexpr auto thread_cluster_desc = make_cluster_descriptor(
            BlockCopyClusterLengths_GemmK_GemmN{}, BlockCopyThreadClusterArrangeOrder{});

        const auto thread_work_id =
            thread_cluster_desc.CalculateClusterIndex(get_thread_local_1d_id());

        // gemmk, gemmn
        constexpr index_t GemmKPerThread = BlockCopySubLengths_GemmK_GemmN::At(I0);
        constexpr index_t GemmNPerThread = BlockCopySubLengths_GemmK_GemmN::At(I1);

        const index_t gemmk_thread_data_on_global =
            thread_work_id[I0] * BlockCopySubLengths_GemmK_GemmN::At(I0);

        const index_t gemmn_thread_data_on_global =
            block_work_id * GemmNPerBlock +
            thread_work_id[I1] * BlockCopySubLengths_GemmK_GemmN::At(I1);

        auto src_coord = make_dynamic_tensor_coordinate(
            src_gemmk_gemmn_global_desc,
            make_multi_index(gemmk_thread_data_on_global, gemmn_thread_data_on_global));

        auto dst_coord = make_dynamic_tensor_coordinate(
            dst_gemmk_gemmn_global_desc,
            make_multi_index(gemmk_thread_data_on_global, gemmn_thread_data_on_global));

        threadwise_dynamic_tensor_slice_transfer_v1<float,
                                                    float,
                                                    decltype(src_gemmk_gemmn_global_desc),
                                                    decltype(dst_gemmk_gemmn_global_desc),
                                                    BlockCopySubLengths_GemmK_GemmN,
                                                    BlockCopyThreadClusterArrangeOrder,
                                                    1,
                                                    1,
                                                    1,
                                                    AddressSpace::Global,
                                                    AddressSpace::Global,
                                                    InMemoryDataOperation::Set,
                                                    1,
                                                    1>(src_gemmk_gemmn_global_desc,
                                                       src_coord,
                                                       p_src_global,
                                                       dst_gemmk_gemmn_global_desc,
                                                       dst_coord,
                                                       p_dst_global);
    }
#endif
};

} // namespace ck
#endif
