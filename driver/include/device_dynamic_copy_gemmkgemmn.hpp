#pragma once
#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "dynamic_gridwise_copy_gemmkgemmn.hpp"

template <typename T, typename SrcDesc, typename DstDesc>
void device_dynamic_copy(SrcDesc,
                         const Tensor<T>& src_gemmk_gemmn,
                         DstDesc,
                         Tensor<T>& dst_gemmk_gemmn)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    std::size_t data_sz = sizeof(T);
    DeviceMem src_gemmk_gemmn_device_buf(data_sz * src_gemmk_gemmn.mDesc.GetElementSpace());
    DeviceMem dst_gemmk_gemmn_device_buf(data_sz * dst_gemmk_gemmn.mDesc.GetElementSpace());

    src_gemmk_gemmn_device_buf.ToDevice(src_gemmk_gemmn.mData.data());

    const auto src_gemmk_gemmn_desc = make_dynamic_native_tensor_descriptor<2>(
        to_multi_index(SrcDesc::GetLengths()), to_multi_index(SrcDesc::GetStrides()));

    const auto dst_gemmk_gemmn_desc = make_dynamic_native_tensor_descriptor<2>(
        to_multi_index(DstDesc::GetLengths()), to_multi_index(DstDesc::GetStrides()));

    index_t GemmK = src_gemmk_gemmn_desc.GetLength(I0);
    index_t GemmN = src_gemmk_gemmn_desc.GetLength(I1);

#if 1
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmKPerBlock = 8;
    constexpr index_t GemmNPerBlock = 128;

    using BlockCopySubLengths_GemmK_GemmN     = Sequence<1, 8>;
    using BlockCopyClusterLengths_GemmK_GemmN = Sequence<8, 16>;
    using BlockCopyThreadClusterArrangeOrder  = Sequence<0, 1>; // [GemmK, GemmN]
    using BlockCopySrcAccessOrder             = Sequence<0, 1>; // [GemmK, GemmN]
    using BlockCopyDstAccessOrder             = Sequence<0, 1>; // [GemmK, GemmN]

    constexpr index_t BlockCopyDataPerAccess_GemmN = 1;
#endif

    const index_t GridSize = GemmN / GemmNPerBlock;

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    constexpr auto gridwise_copy =
        DynamicGridwiseCopy_gemmkgemmn<BlockSize,
                                       GemmKPerBlock,
                                       GemmNPerBlock,
                                       BlockCopySubLengths_GemmK_GemmN,
                                       BlockCopyClusterLengths_GemmK_GemmN,
                                       BlockCopyThreadClusterArrangeOrder,
                                       BlockCopySrcAccessOrder,
                                       BlockCopyDstAccessOrder,
                                       BlockCopyDataPerAccess_GemmN>{};

    std::cout << "Start running " << std::endl;

    launch_kernel(run_gridwise_operation<decltype(gridwise_copy),
                                         const T* const __restrict__,
                                         T* const __restrict__,
                                         decltype(src_gemmk_gemmn_desc),
                                         decltype(dst_gemmk_gemmn_desc)>,
                  dim3(GridSize),
                  dim3(BlockSize),
                  0,
                  0,
                  const_cast<const T* const __restrict__>(
                      static_cast<T*>(src_gemmk_gemmn_device_buf.GetDeviceBuffer())),
                  const_cast<T* const __restrict__>(
                      static_cast<T*>(dst_gemmk_gemmn_device_buf.GetDeviceBuffer())),
                  src_gemmk_gemmn_desc,
                  dst_gemmk_gemmn_desc);

    dst_gemmk_gemmn_device_buf.FromDevice(dst_gemmk_gemmn.mData.data());
}
