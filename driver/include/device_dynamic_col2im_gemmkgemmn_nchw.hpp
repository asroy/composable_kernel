#pragma once
#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "dynamic_gridwise_col2im_gemmkgemmn_nchw.hpp"

template <typename T,
          typename ColDesc,
          typename ImgDesc,
          typename FilterSizes,
          typename OutputSizes,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_dynamic_col2im_gemmkgemmn_nchw(ColDesc,
                                           const Tensor<T>& col_gemmk_gemmn,
                                           ImgDesc,
                                           Tensor<T>& img_n_c_hi_wi,
                                           FilterSizes,
                                           OutputSizes,
                                           ConvStrides,
                                           ConvDilations,
                                           InLeftPads,
                                           InRightPads,
                                           std::size_t nrepeat)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    std::size_t data_sz = sizeof(T);
    DeviceMem col_gemmk_gemmn_device_buf(data_sz * col_gemmk_gemmn.mDesc.GetElementSpace());
    DeviceMem img_n_c_hi_wi_device_buf(data_sz * img_n_c_hi_wi.mDesc.GetElementSpace());

    col_gemmk_gemmn_device_buf.ToDevice(col_gemmk_gemmn.mData.data());
    img_n_c_hi_wi_device_buf.ToDevice(img_n_c_hi_wi.mData.data());

    const auto col_gemmk_gemmn_desc = make_dynamic_native_tensor_descriptor<2>(
        to_multi_index(ColDesc::GetLengths()), to_multi_index(ColDesc::GetStrides()));

    const auto img_n_c_hi_wi_desc = make_dynamic_native_tensor_descriptor<4>(
        to_multi_index(ImgDesc::GetLengths()), to_multi_index(ImgDesc::GetStrides()));

    const auto filter_sizes   = to_multi_index(FilterSizes{});
    const auto out_sizes      = to_multi_index(OutputSizes{});
    const auto conv_strides   = to_multi_index(ConvStrides{});
    const auto conv_dilations = to_multi_index(ConvDilations{});
    const auto in_left_pads   = to_multi_index(InLeftPads{});
    const auto in_right_pads  = to_multi_index(InRightPads{});

    const auto img_gemmk_gemmn_desc = map_img_into_col(img_n_c_hi_wi_desc,
                                                       out_sizes,
                                                       filter_sizes,
                                                       conv_strides,
                                                       conv_dilations,
                                                       in_left_pads,
                                                       in_right_pads);

    const index_t GemmN = col_gemmk_gemmn_desc.GetLength(I1);

#if 1
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmKPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;

    using BlockCopySubLengths_GemmK_GemmN     = Sequence<8, 8>;
    using BlockCopyClusterLengths_GemmK_GemmN = Sequence<16, 16>;
    using BlockCopyThreadClusterArrangeOrder  = Sequence<0, 1>; // [GemmK, GemmN]
    using BlockCopySrcAccessOrder             = Sequence<0, 1>; // [GemmK, GemmN]
    using BlockCopyDstAccessOrder             = Sequence<0, 1>; // [GemmK, GemmN]

    constexpr index_t BlockCopyDataPerAccess_GemmN = 1;
#endif

    const index_t GridSize = GemmN / GemmNPerBlock;

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    constexpr auto gridwise_col2im =
        DynamicGridwiseCol2Im_gemmkgemmn_nchw<BlockSize,
                                              GemmKPerBlock,
                                              GemmNPerBlock,
                                              BlockCopySubLengths_GemmK_GemmN,
                                              BlockCopyClusterLengths_GemmK_GemmN,
                                              BlockCopyThreadClusterArrangeOrder,
                                              BlockCopySrcAccessOrder,
                                              BlockCopyDstAccessOrder,
                                              BlockCopyDataPerAccess_GemmN>{};

    for(index_t i = 0; i < 1; ++i)
    {
        std::cout << "Start running " << nrepeat << " times..." << std::endl;

        KernelTimer timer;
        timer.Start();

        for(index_t j = 0; j < nrepeat; ++j)
        {
            launch_kernel(run_gridwise_operation<decltype(gridwise_col2im),
                                                 const T* const __restrict__,
                                                 T* const __restrict__,
                                                 decltype(col_gemmk_gemmn_desc),
                                                 decltype(img_gemmk_gemmn_desc)>,
                          dim3(GridSize),
                          dim3(BlockSize),
                          0,
                          0,
                          const_cast<const T* const __restrict__>(
                              static_cast<T*>(col_gemmk_gemmn_device_buf.GetDeviceBuffer())),
                          const_cast<T* const __restrict__>(
                              static_cast<T*>(img_n_c_hi_wi_device_buf.GetDeviceBuffer())),
                          col_gemmk_gemmn_desc,
                          img_gemmk_gemmn_desc);
        }

        timer.End();

        float ave_time = timer.GetElapsedTime() / nrepeat;

        std::cout << "Average time : " << ave_time << " ms" << std::endl;
    }

    img_n_c_hi_wi_device_buf.FromDevice(img_n_c_hi_wi.mData.data());
}
