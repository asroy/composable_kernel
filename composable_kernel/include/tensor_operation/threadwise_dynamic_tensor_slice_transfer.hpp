#ifndef CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP
#define CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

// threadwise_dynamic_tensor_slice_transfer_v1r1 has scratch memory issue, due to
// it constructs new tensor coordinate
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SrcDstDimAccessOrder,
          index_t SrcDstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          InMemoryDataOperation DstInMemOp,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector>
__host__ __device__ constexpr void threadwise_dynamic_tensor_slice_transfer_v1r1(
    const SrcDesc& src_desc,
    const DynamicTensorCoordinate_t<SrcDesc>& src_origin_coord,
    const SrcData* p_src,
    const DstDesc& dst_desc,
    const DynamicTensorCoordinate_t<DstDesc>& dst_origin_coord,
    DstData* p_dst)
{
    // comment: construction tensor coordinate here seems to cause scratch memory issue
    auto src_coord = src_origin_coord;
    auto dst_coord = dst_origin_coord;

    // TODO use constexpr for coordinate-step to make sure compiler behave correctly
    const auto src_step_0_p1 =
        make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(0, 1));
    const auto src_step_0_m1 =
        make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(0, -1));
    const auto src_step_p1_0 =
        make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(1, 0));
    const auto src_step_m1_0 =
        make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(-1, 0));

    const auto dst_step_0_p1 =
        make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, 1));
    const auto dst_step_0_m1 =
        make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, -1));
    const auto dst_step_p1_0 =
        make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(1, 0));
    const auto dst_step_m1_0 =
        make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(-1, 0));

    constexpr index_t J0 = SliceLengths{}[0];
    constexpr index_t J1 = SliceLengths{}[1];

    bool forward_dim0 = true;
    bool forward_dim1 = true;

    // hardcoded for 2d loop for now
#pragma unroll
    for(index_t j0 = 0; j0 < J0; ++j0)
    {
#pragma unroll
        for(index_t j1 = 0; j1 < J1; ++j1)
        {
            // do work
            transfer_data<SrcData,
                          1,
                          SrcAddressSpace,
                          DstAddressSpace,
                          DstInMemOp,
                          SrcScalarStrideInVector,
                          DstScalarStrideInVector>(
                p_src,
                src_coord.GetOffset(),
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord),
                src_desc.GetElementSpaceSize(),
                p_dst,
                dst_coord.GetOffset(),
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord),
                dst_desc.GetElementSpaceSize());

            // move dim1 iterator
            if(j1 < J1 - 1)
            {
                if(forward_dim1)
                {
                    move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_0_p1);
                    move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_0_p1);
                }
                else
                {
                    move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_0_m1);
                    move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_0_m1);
                }
            }
        }

        // switch dim1 iteration direction
        forward_dim1 = !forward_dim1;

        // move dim0 iterator
        if(j0 < J0 - 1)
        {
            if(forward_dim0)
            {
                move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_p1_0);
                move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_p1_0);
            }
            else
            {
                move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_m1_0);
                move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_m1_0);
            }
        }
    }
}

// threadwise_dynamic_tensor_slice_transfer_v1r2 does not have scratch memory issue, due to
// it does not construct new tensor coordinate
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SrcDstDimAccessOrder,
          index_t SrcDstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          InMemoryDataOperation DstInMemOp,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector>
__host__ __device__ constexpr void
threadwise_dynamic_tensor_slice_transfer_v1r2(const SrcDesc& src_desc,
                                              DynamicTensorCoordinate_t<SrcDesc>& src_coord,
                                              const SrcData* p_src,
                                              const DstDesc& dst_desc,
                                              DynamicTensorCoordinate_t<DstDesc>& dst_coord,
                                              DstData* p_dst)
{
    static_assert(remove_reference_t<SrcDesc>::GetNumOfDimension() ==
                      remove_reference_t<DstDesc>::GetNumOfDimension(),
                  "inconsistent # of dimension");

    if constexpr(remove_reference_t<SrcDesc>::GetNumOfDimension() == 2)
    {
        // TODO use constexpr for coordinate-step to make sure compiler behave correctly
        const auto src_step_0_p1 =
            make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(0, 1));
        const auto src_step_0_m1 =
            make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(0, -1));

        const auto src_step_p1_0 =
            make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(1, 0));
        const auto src_step_m1_0 =
            make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(-1, 0));

        const auto dst_step_0_p1 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, 1));
        const auto dst_step_0_m1 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, -1));

        const auto dst_step_p1_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(1, 0));
        const auto dst_step_m1_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(-1, 0));

        constexpr index_t Len0 = SliceLengths{}[0];
        constexpr index_t Len1 = SliceLengths{}[1];

        bool forward_dim0 = true;
        bool forward_dim1 = true;

#pragma unroll
        for(index_t j0 = 0; j0 < Len0; ++j0)
        {
#pragma unroll
            for(index_t j1 = 0; j1 < Len1; ++j1)
            {
                // do work
                transfer_data<SrcData,
                              1,
                              SrcAddressSpace,
                              DstAddressSpace,
                              DstInMemOp,
                              SrcScalarStrideInVector,
                              DstScalarStrideInVector>(
                    p_src,
                    src_coord.GetOffset(),
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc,
                                                                                src_coord),
                    src_desc.GetElementSpaceSize(),
                    p_dst,
                    dst_coord.GetOffset(),
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc,
                                                                                dst_coord),
                    dst_desc.GetElementSpaceSize());

                // move dim1 iterator
                if(j1 < Len1 - 1)
                {
                    if(forward_dim1)
                    {
                        move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_0_p1);
                        move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_0_p1);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_0_m1);
                        move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_0_m1);
                    }
                }
            }

            // switch dim1 iteration direction
            forward_dim1 = !forward_dim1;

            // move dim0 iterator
            if(j0 < Len0 - 1)
            {
                if(forward_dim0)
                {
                    move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_p1_0);
                    move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_p1_0);
                }
                else
                {
                    move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_m1_0);
                    move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_m1_0);
                }
            }
        }

        // move src and dst coordinate back to their origins
        // move src and dst coordinate back to their origins
        constexpr index_t loc0 = Len0 - 1;
        constexpr index_t loc1 = Len0 % 2 == 0 ? 0 : Len1 - 1;

        const auto src_step_back =
            make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(-loc0, -loc1));

        const auto dst_step_back =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(-loc0, -loc1));

        move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_back);
        move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_back);
    }
    else if constexpr(remove_reference_t<SrcDesc>::GetNumOfDimension() == 4)
    {
        // TODO use constexpr for coordinate-step to make sure compiler behave correctly
        const auto src_step_0_0_0_p1 =
            make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(0, 0, 0, 1));
        const auto src_step_0_0_0_m1 =
            make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(0, 0, 0, -1));

        const auto src_step_0_0_p1_0 =
            make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(0, 0, 1, 0));
        const auto src_step_0_0_m1_0 =
            make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(0, 0, -1, 0));

        const auto src_step_0_p1_0_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, 1, 0, 0));
        const auto src_step_0_m1_0_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, -1, 0, 0));

        const auto src_step_p1_0_0_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(1, 0, 0, 0));
        const auto src_step_m1_0_0_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(-1, 0, 0, 0));

        const auto dst_step_0_0_0_p1 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, 0, 0, 1));
        const auto dst_step_0_0_0_m1 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, 0, 0, -1));

        const auto dst_step_0_0_p1_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, 0, 1, 0));
        const auto dst_step_0_0_m1_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, 0, -1, 0));

        const auto dst_step_0_p1_0_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, 1, 0, 0));
        const auto dst_step_0_m1_0_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, -1, 0, 0));

        const auto dst_step_p1_0_0_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(1, 0, 0, 0));
        const auto dst_step_m1_0_0_0 =
            make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(-1, 0, 0, 0));

        constexpr index_t Len0 = SliceLengths{}[0];
        constexpr index_t Len1 = SliceLengths{}[1];
        constexpr index_t Len2 = SliceLengths{}[2];
        constexpr index_t Len3 = SliceLengths{}[3];

        bool forward_dim0 = true;
        bool forward_dim1 = true;
        bool forward_dim2 = true;
        bool forward_dim3 = true;

#pragma unroll
        for(index_t j0 = 0; j0 < Len0; ++j0)
        {
#pragma unroll
            for(index_t j1 = 0; j1 < Len1; ++j1)
            {
#pragma unroll
                for(index_t j2 = 0; j2 < Len2; ++j2)
                {
#pragma unroll
                    for(index_t j3 = 0; j3 < Len3; ++j3)
                    {
                        // do work
                        transfer_data<SrcData,
                                      1,
                                      SrcAddressSpace,
                                      DstAddressSpace,
                                      DstInMemOp,
                                      SrcScalarStrideInVector,
                                      DstScalarStrideInVector>(
                            p_src,
                            src_coord.GetOffset(),
                            coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc,
                                                                                        src_coord),
                            src_desc.GetElementSpaceSize(),
                            p_dst,
                            dst_coord.GetOffset(),
                            coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc,
                                                                                        dst_coord),
                            dst_desc.GetElementSpaceSize());

                        // move dim1 iterator
                        if(j3 < Len3 - 1)
                        {
                            if(forward_dim3)
                            {
                                move_dynamic_tensor_coordinate(
                                    src_desc, src_coord, src_step_0_0_0_p1);
                                move_dynamic_tensor_coordinate(
                                    dst_desc, dst_coord, dst_step_0_0_0_p1);
                            }
                            else
                            {
                                move_dynamic_tensor_coordinate(
                                    src_desc, src_coord, src_step_0_0_0_m1);
                                move_dynamic_tensor_coordinate(
                                    dst_desc, dst_coord, dst_step_0_0_0_m1);
                            }
                        }
                    }

                    // switch dim3 iteration direction
                    forward_dim3 = !forward_dim3;

                    // move dim1 iterator
                    if(j2 < Len2 - 1)
                    {
                        if(forward_dim2)
                        {
                            move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_0_0_p1_0);
                            move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_0_0_p1_0);
                        }
                        else
                        {
                            move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_0_0_m1_0);
                            move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_0_0_m1_0);
                        }
                    }
                }

                // switch dim2 iteration direction
                forward_dim2 = !forward_dim2;

                // move dim1 iterator
                if(j1 < Len1 - 1)
                {
                    if(forward_dim1)
                    {
                        move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_0_p1_0_0);
                        move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_0_p1_0_0);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_0_m1_0_0);
                        move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_0_m1_0_0);
                    }
                }
            }

            // switch dim1 iteration direction
            forward_dim1 = !forward_dim1;

            // move dim0 iterator
            if(j0 < Len0 - 1)
            {
                if(forward_dim0)
                {
                    move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_p1_0_0_0);
                    move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_p1_0_0_0);
                }
                else
                {
                    move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_m1_0_0_0);
                    move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_m1_0_0_0);
                }
            }
        }

        // move src and dst coordinate back to their origins
        constexpr index_t loc0 = Len0 - 1;
        constexpr index_t loc1 = Len0 % 2 == 0 ? 0 : Len1 - 1;
        constexpr index_t loc2 = Len1 % 2 == 0 ? 0 : Len2 - 1;
        constexpr index_t loc3 = Len2 % 2 == 0 ? 0 : Len3 - 1;

        const auto src_step_back = make_dynamic_tensor_coordinate_step(
            src_desc, make_multi_index(-loc0, -loc1, -loc2, -loc3));

        const auto dst_step_back = make_dynamic_tensor_coordinate_step(
            dst_desc, make_multi_index(-loc0, -loc1, -loc2, -loc3));

        move_dynamic_tensor_coordinate(src_desc, src_coord, src_step_back);
        move_dynamic_tensor_coordinate(dst_desc, dst_coord, dst_step_back);
    }
}

// this version has scratch memory issue, due to:
// 1. ThreadwiseDynamicTensorSliceTransfer_v1r1 keeps reference to tensor descriptor
// 2. threadwise_dynamic_tensor_slice_transfer_v1r1 constructs new tensor coordinate
template <typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SrcDstDimAccessOrder,
          index_t SrcDstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          InMemoryDataOperation DstInMemOp,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector>
struct ThreadwiseDynamicTensorSliceTransfer_v1r1
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = decltype(make_dynamic_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_dynamic_tensor_coordinate(DstDesc{}, Index{}));

    using SrcCoordStep = decltype(make_dynamic_tensor_coordinate_step(SrcDesc{}, Index{}));
    using DstCoordStep = decltype(make_dynamic_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseDynamicTensorSliceTransfer_v1r1(const SrcDesc& src_desc,
                                                                   const Index& src_slice_origin,
                                                                   const DstDesc& dst_desc,
                                                                   const Index& dst_slice_origin)
        : src_desc_(src_desc),
          src_slice_origin_(make_dynamic_tensor_coordinate(src_desc, src_slice_origin)),
          dst_desc_(dst_desc),
          dst_slice_origin_(make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin))
    {
    }

    __device__ constexpr ThreadwiseDynamicTensorSliceTransfer_v1r1()
        : ThreadwiseDynamicTensorSliceTransfer_v1r1(
              SrcDesc{}, make_zero_multi_index<nDim>(), DstDesc{}, make_zero_multi_index<nDim>())
    {
    }

    template <typename SrcData, typename DstData>
    __device__ void Run(const SrcData* p_src, DstData* p_dst) const
    {
        threadwise_dynamic_tensor_slice_transfer_v1r1<SrcData,
                                                      DstData,
                                                      SrcDesc,
                                                      DstDesc,
                                                      SliceLengths,
                                                      SrcDstDimAccessOrder,
                                                      SrcDstVectorDim,
                                                      SrcScalarPerVector,
                                                      DstScalarPerVector,
                                                      SrcAddressSpace,
                                                      DstAddressSpace,
                                                      DstInMemOp,
                                                      SrcScalarStrideInVector,
                                                      DstScalarStrideInVector>(
            src_desc_, src_slice_origin_, p_src, dst_desc_, dst_slice_origin_, p_dst);
    }

    __device__ void SetSrcSliceOrigin(const Index& src_slice_origin_idx)
    {
        src_slice_origin_ = make_dynamic_tensor_coordinate(src_desc_, src_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const Index& dst_slice_origin_idx)
    {
        dst_slice_origin_ = make_dynamic_tensor_coordinate(dst_desc_, dst_slice_origin_idx);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const Index& src_slice_origin_step_idx)
    {
        const auto src_slice_origin_step =
            make_dynamic_tensor_coordinate_step(src_desc_, src_slice_origin_step_idx);

        move_dynamic_tensor_coordinate(src_desc_, src_slice_origin_, src_slice_origin_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const Index& dst_slice_origin_step_idx)
    {
        const auto dst_slice_origin_step =
            make_dynamic_tensor_coordinate_step(dst_desc_, dst_slice_origin_step_idx);

        move_dynamic_tensor_coordinate(dst_desc_, dst_slice_origin_, dst_slice_origin_step);
    }

    private:
    const SrcDesc& src_desc_;
    const DstDesc& dst_desc_;

    SrcCoord src_slice_origin_;
    DstCoord dst_slice_origin_;
};

// this version does not have scratch memory issue, due to:
// 1. ThreadwiseDynamicTensorSliceTransfer_v1r2 does not keep reference to tensor descriptor
// 2. threadwise_dynamic_tensor_slice_transfer_v1r2 does not construct new tensor coordinate
template <typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SrcDstDimAccessOrder,
          index_t SrcDstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          InMemoryDataOperation DstInMemOp,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector>
struct ThreadwiseDynamicTensorSliceTransfer_v1r2
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = decltype(make_dynamic_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_dynamic_tensor_coordinate(DstDesc{}, Index{}));

    using SrcCoordStep = decltype(make_dynamic_tensor_coordinate_step(SrcDesc{}, Index{}));
    using DstCoordStep = decltype(make_dynamic_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseDynamicTensorSliceTransfer_v1r2(const SrcDesc& src_desc,
                                                                   const Index& src_slice_origin,
                                                                   const DstDesc& dst_desc,
                                                                   const Index& dst_slice_origin)
        : src_slice_origin_(make_dynamic_tensor_coordinate(src_desc, src_slice_origin)),
          dst_slice_origin_(make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin))
    {
    }

    __device__ constexpr ThreadwiseDynamicTensorSliceTransfer_v1r2()
        : ThreadwiseDynamicTensorSliceTransfer_v1r2(
              SrcDesc{}, make_zero_multi_index<nDim>(), DstDesc{}, make_zero_multi_index<nDim>())
    {
    }

    template <typename SrcData, typename DstData>
    __device__ void
    Run(const SrcDesc& src_desc, const SrcData* p_src, const DstDesc& dst_desc, DstData* p_dst)
    {
        threadwise_dynamic_tensor_slice_transfer_v1r2<SrcData,
                                                      DstData,
                                                      SrcDesc,
                                                      DstDesc,
                                                      SliceLengths,
                                                      SrcDstDimAccessOrder,
                                                      SrcDstVectorDim,
                                                      SrcScalarPerVector,
                                                      DstScalarPerVector,
                                                      SrcAddressSpace,
                                                      DstAddressSpace,
                                                      DstInMemOp,
                                                      SrcScalarStrideInVector,
                                                      DstScalarStrideInVector>(
            src_desc, src_slice_origin_, p_src, dst_desc, dst_slice_origin_, p_dst);
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_slice_origin_ = make_dynamic_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_slice_origin_ = make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // is it OK to do this every time?
        const auto src_slice_origin_step =
            make_dynamic_tensor_coordinate_step(src_desc, src_slice_origin_step_idx);

        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_slice_origin_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // is it OK to do this every time?
        const auto dst_slice_origin_step =
            make_dynamic_tensor_coordinate_step(dst_desc, dst_slice_origin_step_idx);

        move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_slice_origin_step);
    }

    private:
    SrcCoord src_slice_origin_;
    DstCoord dst_slice_origin_;
};

} // namespace ck
#endif
