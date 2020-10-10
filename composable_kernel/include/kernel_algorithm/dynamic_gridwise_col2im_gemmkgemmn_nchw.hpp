#ifndef CK_DYNAMIC_GRIDWISE_COL2IM_GEMMKGEMMN_NCHW_HPP
#define CK_DYNAMIC_GRIDWISE_COL2IM_GEMMKGEMMN_NCHW_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "blockwise_dynamic_tensor_slice_transfer.hpp"

namespace ck {

template <typename... In>
__host__ __device__ constexpr auto
map_img_into_col(const DynamicTensorDescriptor<In...>& in_n_c_hi_wi_global_desc,
                 const MultiIndex<2> out_sizes,
                 const MultiIndex<2> filter_sizes,
                 const MultiIndex<2> conv_strides,
                 const MultiIndex<2> conv_dilations,
                 const MultiIndex<2> in_left_pads,
                 const MultiIndex<2> in_right_pads)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const index_t N  = in_n_c_hi_wi_global_desc.GetLength(I0);
    const index_t C  = in_n_c_hi_wi_global_desc.GetLength(I1);
    const index_t Hi = in_n_c_hi_wi_global_desc.GetLength(I2);
    const index_t Wi = in_n_c_hi_wi_global_desc.GetLength(I3);

    const index_t Ho = out_sizes[I0];
    const index_t Wo = out_sizes[I1];

    const index_t Y = filter_sizes[I0];
    const index_t X = filter_sizes[I1];

    const index_t ConvStrideH = conv_strides[I0];
    const index_t ConvStrideW = conv_strides[I1];

    const index_t ConvDilationH = conv_dilations[I0];
    const index_t ConvDilationW = conv_dilations[I1];

    const index_t InLeftPadH = in_left_pads[I0];
    const index_t InLeftPadW = in_left_pads[I1];

    const index_t InRightPadH = in_right_pads[I0];
    const index_t InRightPadW = in_right_pads[I1];

    const auto in_n_c_hip_wip_global_desc = transform_dynamic_tensor_descriptor(
        transform_dynamic_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(DynamicPassThrough{N},
                       DynamicPassThrough{C},
                       DynamicLeftPad{Hi, InLeftPadH},
                       DynamicLeftPad{Wi, InLeftPadW}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{})),
        make_tuple(DynamicPassThrough{N},
                   DynamicPassThrough{C},
                   DynamicRightPad{Hi + InLeftPadH, InRightPadH},
                   DynamicRightPad{Wi + InLeftPadW, InRightPadW}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    const index_t Hip = in_n_c_hip_wip_global_desc.GetLength(I2);
    const index_t Wip = in_n_c_hip_wip_global_desc.GetLength(I3);

    const auto in_n_c_y_ho_x_wo_global_desc = transform_dynamic_tensor_descriptor(
        in_n_c_hip_wip_global_desc,
        make_tuple(
            DynamicPassThrough{N},
            DynamicPassThrough{C},
            DynamicEmbed<2>{make_multi_index(Y, Ho), make_multi_index(ConvDilationH, ConvStrideH)},
            DynamicEmbed<2>{make_multi_index(X, Wo), make_multi_index(ConvDilationW, ConvStrideW)}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

    const auto in_gemmk_gemmn_global_desc = transform_dynamic_tensor_descriptor(
        in_n_c_y_ho_x_wo_global_desc,
        make_tuple(DynamicMerge<3>{make_multi_index(C, Y, X)},
                   DynamicMerge<3>{make_multi_index(N, Ho, Wo)}),
        make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    return in_gemmk_gemmn_global_desc;
}

template <index_t BlockSize,
          index_t GemmKPerBlock,
          index_t GemmNPerBlock,
          typename BlockCopySubLengths_GemmK_GemmN,
          typename BlockCopyClusterLengths_GemmK_GemmN,
          typename BlockCopyThreadClusterArrangeOrder,
          typename BlockCopySrcAccessOrder,
          typename BlockCopyDstAccessOrder,
          index_t BlockCopyDataPerAccess_GemmN>
struct DynamicGridwiseCol2Im_gemmkgemmn_nchw
{
    template <typename... Col, typename... Img>
    __device__ void Run(const float* const __restrict__ p_col_global,
                        float* const __restrict__ p_img_global,
                        const DynamicTensorDescriptor<Col...>& col_gemmk_gemmn_global_desc,
                        const DynamicTensorDescriptor<Img...>& img_gemmk_gemmn_global_desc) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        const index_t GemmK = col_gemmk_gemmn_global_desc.GetLength(I0);
        const index_t GemmN = col_gemmk_gemmn_global_desc.GetLength(I1);

        // divide block work by GemmN
        const index_t GemmNBlockWork = GemmN / GemmNPerBlock;

        const index_t block_work_id = get_block_1d_id();

        const index_t gemmn_block_data_on_global = block_work_id * GemmNPerBlock;

        // blockwise atomic accumulation
        auto blockwise_copy =
            BlockwiseDynamicTensorSliceTransfer_v1<BlockSize,
                                                   float,
                                                   float,
                                                   decltype(col_gemmk_gemmn_global_desc),
                                                   decltype(img_gemmk_gemmn_global_desc),
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
                                                   InMemoryDataOperation::AtomicAdd,
                                                   1,
                                                   1>(
                col_gemmk_gemmn_global_desc,
                make_multi_index(0, gemmn_block_data_on_global),
                img_gemmk_gemmn_global_desc,
                make_multi_index(0, gemmn_block_data_on_global));

        auto col_gemmk_gemmn_coord =
            make_dynamic_tensor_coordinate(col_gemmk_gemmn_global_desc, make_multi_index(0, 0));

        auto img_gemmk_gemmn_coord =
            make_dynamic_tensor_coordinate(img_gemmk_gemmn_global_desc, make_multi_index(0, 0));

        const auto col_gemmk_gemmn_coord_step = make_dynamic_tensor_coordinate_step(
            col_gemmk_gemmn_global_desc, make_multi_index(GemmKPerBlock, 0));

        const auto img_gemmk_gemmn_coord_step = make_dynamic_tensor_coordinate_step(
            img_gemmk_gemmn_global_desc, make_multi_index(GemmKPerBlock, 0));

        for(index_t gemmk = 0; gemmk < GemmK - GemmKPerBlock; gemmk += GemmKPerBlock)
        {
            blockwise_copy.Run(p_col_global, p_img_global);

            move_dynamic_tensor_coordinate(
                col_gemmk_gemmn_global_desc, col_gemmk_gemmn_coord, col_gemmk_gemmn_coord_step);

            move_dynamic_tensor_coordinate(
                img_gemmk_gemmn_global_desc, img_gemmk_gemmn_coord, img_gemmk_gemmn_coord_step);
        }
    }
};

} // namespace ck
#endif
