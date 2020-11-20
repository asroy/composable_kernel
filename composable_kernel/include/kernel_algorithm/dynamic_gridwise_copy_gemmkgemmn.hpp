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
    template <typename... Src, typename... Dst>
    __device__ void Run_r1(const float* const __restrict__ p_src_global,
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
            BlockwiseDynamicTensorSliceTransfer_v1r1<BlockSize,
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
                                                     1>
#elif 1
            BlockwiseDynamicTensorSliceTransfer_v2r1<BlockSize,
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
                                                     1>
#endif
            (src_gemmk_gemmn_global_desc,
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

    template <typename... Src, typename... Dst>
    __device__ void Run_r2(const float* const __restrict__ p_src_global,
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
            BlockwiseDynamicTensorSliceTransfer_v2r2<BlockSize,
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
                src_gemmk_gemmn_global_desc,
                make_multi_index(0, gemmn_block_data_on_global),
                dst_gemmk_gemmn_global_desc,
                make_multi_index(0, gemmn_block_data_on_global));

        for(index_t gemmk = 0; gemmk < GemmK; gemmk += GemmKPerBlock)
        {
            blockwise_copy.Run(src_gemmk_gemmn_global_desc,
                               p_src_global,
                               dst_gemmk_gemmn_global_desc,
                               p_dst_global);

            blockwise_copy.MoveSrcSliceWindow(src_gemmk_gemmn_global_desc,
                                              make_multi_index(GemmKPerBlock, 0));
            blockwise_copy.MoveDstSliceWindow(dst_gemmk_gemmn_global_desc,
                                              make_multi_index(GemmKPerBlock, 0));
        }
    }

    template <typename... Src, typename... Dst>
    __device__ void Run(const float* const __restrict__ p_src_global,
                        float* const __restrict__ p_dst_global,
                        const DynamicTensorDescriptor<Src...>& src_gemmk_gemmn_global_desc,
                        const DynamicTensorDescriptor<Dst...>& dst_gemmk_gemmn_global_desc) const
    {
        Run_r2(
            p_src_global, p_dst_global, src_gemmk_gemmn_global_desc, dst_gemmk_gemmn_global_desc);
    }
};

} // namespace ck
#endif
