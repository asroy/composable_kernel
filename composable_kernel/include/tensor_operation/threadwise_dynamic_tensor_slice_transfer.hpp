#ifndef CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP
#define CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

// this version is less likely to have scratch memory issue, due to:
// 1. It does not keep reference to tensor descriptor
// 2. It does not construct new tensor coordinate for this->Run()
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
          index_t DstScalarStrideInVector,
          bool SrcResetCoordinateAfterRun, // control whether to move back src coordinate after each
                                           // Run(),  will be fused with MoveSrcSliceWindow to
                                           // save addr computation
          bool DstResetCoordinateAfterRun> // control whether to move back dst coordinate after each
                                           // RunWrite(),  will be fused with MoveDstSliceWindow to
                                           // save addr computation
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

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_slice_origin_ = make_dynamic_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_slice_origin_ = make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    __device__ void
    Run(const SrcDesc& src_desc, const SrcData* p_src, const DstDesc& dst_desc, DstData* p_dst)
    {
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

#pragma unroll
            for(index_t iter0 = 0; iter0 < Len0; ++iter0)
            {
#pragma unroll
                for(index_t iter1 = 0; iter1 < Len1; ++iter1)
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
                        src_slice_origin_.GetOffset(),
                        coordinate_has_valid_offset_assuming_visible_index_is_valid(
                            src_desc, src_slice_origin_),
                        src_desc.GetElementSpaceSize(),
                        p_dst,
                        dst_slice_origin_.GetOffset(),
                        coordinate_has_valid_offset_assuming_visible_index_is_valid(
                            dst_desc, dst_slice_origin_),
                        dst_desc.GetElementSpaceSize());

                    // move dim1 iterator
                    if(iter1 < Len1 - 1)
                    {
                        bool forward_dim1 = (iter0 % 2 == 0);

                        if(forward_dim1)
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_p1);
                            move_dynamic_tensor_coordinate(
                                dst_desc, dst_slice_origin_, dst_step_0_p1);
                        }
                        else
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_m1);
                            move_dynamic_tensor_coordinate(
                                dst_desc, dst_slice_origin_, dst_step_0_m1);
                        }
                    }
                }

                // move dim0 iterator
                if(iter0 < Len0 - 1)
                {
                    move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_p1_0);
                    move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_step_p1_0);
                }
            }
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

#pragma unroll
            for(index_t iter0 = 0; iter0 < Len0; ++iter0)
            {
#pragma unroll
                for(index_t iter1 = 0; iter1 < Len1; ++iter1)
                {
#pragma unroll
                    for(index_t iter2 = 0; iter2 < Len2; ++iter2)
                    {
#pragma unroll
                        for(index_t iter3 = 0; iter3 < Len3; ++iter3)
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
                                src_slice_origin_.GetOffset(),
                                coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                    src_desc, src_slice_origin_),
                                src_desc.GetElementSpaceSize(),
                                p_dst,
                                dst_slice_origin_.GetOffset(),
                                coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                    dst_desc, dst_slice_origin_),
                                dst_desc.GetElementSpaceSize());

                            // move dim1 iterator
                            if(iter3 < Len3 - 1)
                            {
                                bool forward_dim3 = (iter2 % 2 == 0);

                                if(forward_dim3)
                                {
                                    move_dynamic_tensor_coordinate(
                                        src_desc, src_slice_origin_, src_step_0_0_0_p1);
                                    move_dynamic_tensor_coordinate(
                                        dst_desc, dst_slice_origin_, dst_step_0_0_0_p1);
                                }
                                else
                                {
                                    move_dynamic_tensor_coordinate(
                                        src_desc, src_slice_origin_, src_step_0_0_0_m1);
                                    move_dynamic_tensor_coordinate(
                                        dst_desc, dst_slice_origin_, dst_step_0_0_0_m1);
                                }
                            }
                        }

                        // move dim1 iterator
                        if(iter2 < Len2 - 1)
                        {
                            bool forward_dim2 = (iter1 % 2 == 0);

                            if(forward_dim2)
                            {
                                move_dynamic_tensor_coordinate(
                                    src_desc, src_slice_origin_, src_step_0_0_p1_0);
                                move_dynamic_tensor_coordinate(
                                    dst_desc, dst_slice_origin_, dst_step_0_0_p1_0);
                            }
                            else
                            {
                                move_dynamic_tensor_coordinate(
                                    src_desc, src_slice_origin_, src_step_0_0_m1_0);
                                move_dynamic_tensor_coordinate(
                                    dst_desc, dst_slice_origin_, dst_step_0_0_m1_0);
                            }
                        }
                    }

                    // move dim1 iterator
                    if(iter1 < Len1 - 1)
                    {
                        bool forward_dim1 = (iter0 % 2 == 0);

                        if(forward_dim1)
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_p1_0_0);
                            move_dynamic_tensor_coordinate(
                                dst_desc, dst_slice_origin_, dst_step_0_p1_0_0);
                        }
                        else
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_m1_0_0);
                            move_dynamic_tensor_coordinate(
                                dst_desc, dst_slice_origin_, dst_step_0_m1_0_0);
                        }
                    }
                }

                // move dim0 iterator:
                if(iter0 < Len0 - 1)
                {
                    // move forward in dim0
                    move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_p1_0_0_0);
                    move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_step_p1_0_0_0);
                }
            }
        }

        // move src and dst coordinate back to their origins
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_back_step =
                make_dynamic_tensor_coordinate_step(src_desc, GetCoordinateBackStep());

            move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_back_step);
        }

        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_back_step =
                make_dynamic_tensor_coordinate_step(dst_desc, GetCoordinateBackStep());

            move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_back_step);
        }
    }

    __device__ static constexpr auto GetCoordinateBackStep()
    {
        MultiIndex<nDim> back_step;

        back_step(Number<0>{}) = 1 - SliceLengths{}[0];

        static_for<1, nDim, 1>{}([&](auto i) {
            back_step(i) = (SliceLengths{}[i - Number<1>{}] % 2 == 0) ? 0 : (1 - SliceLengths{}[i]);
        });

        return back_step;
    }

    __device__ void
    Run_hack(const SrcDesc& src_desc, const SrcData* p_src, const DstDesc& dst_desc, DstData* p_dst)
    {
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

#pragma unroll
            for(index_t iter0 = 0; iter0 < Len0; ++iter0)
            {
#pragma unroll
                for(index_t iter1 = 0; iter1 < Len1; ++iter1)
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
                        src_slice_origin_.GetOffset(),
                        coordinate_has_valid_offset_assuming_visible_index_is_valid(
                            src_desc, src_slice_origin_),
                        src_desc.GetElementSpaceSize(),
                        p_dst,
                        dst_slice_origin_.GetOffset(),
                        coordinate_has_valid_offset_assuming_visible_index_is_valid(
                            dst_desc, dst_slice_origin_),
                        dst_desc.GetElementSpaceSize());

                    // move dim1 iterator
                    if(iter1 < Len1 - 1)
                    {
                        bool forward_dim1 = (iter0 % 2 == 0);

                        if(forward_dim1)
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_p1);
                            move_dynamic_tensor_coordinate(
                                dst_desc, dst_slice_origin_, dst_step_0_p1);
                        }
                        else
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_m1);
                            move_dynamic_tensor_coordinate(
                                dst_desc, dst_slice_origin_, dst_step_0_m1);
                        }
                    }
                }

                // move dim0 iterator
                if(iter0 < Len0 - 1)
                {
                    move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_p1_0);
                    move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_step_p1_0);
                }
            }
        }
        else if constexpr(remove_reference_t<SrcDesc>::GetNumOfDimension() == 4)
        {
        // TODO use constexpr for coordinate-step to make sure compiler behave correctly
#if 0
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
#else
            // hack for output tensor
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

            const auto dst_step_0_0_0_p1 = make_dynamic_tensor_coordinate_step_hack(
                dst_desc, make_multi_index(0, 0, 0, 1), Sequence<0, 0, 1, 0, 0>{});
            const auto dst_step_0_0_0_m1 = make_dynamic_tensor_coordinate_step_hack(
                dst_desc, make_multi_index(0, 0, 0, -1), Sequence<0, 0, 2, 0, 0>{});

            const auto dst_step_0_0_p1_0 = make_dynamic_tensor_coordinate_step_hack(
                dst_desc, make_multi_index(0, 0, 1, 0), Sequence<0, 0, 1, 0, 0>{});
            const auto dst_step_0_0_m1_0 = make_dynamic_tensor_coordinate_step_hack(
                dst_desc, make_multi_index(0, 0, -1, 0), Sequence<0, 0, 2, 0, 0>{});

            const auto dst_step_0_p1_0_0 =
                make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, 1, 0, 0));
            const auto dst_step_0_m1_0_0 =
                make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, -1, 0, 0));

            const auto dst_step_p1_0_0_0 =
                make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(1, 0, 0, 0));
            const auto dst_step_m1_0_0_0 =
                make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(-1, 0, 0, 0));
#endif

            constexpr index_t Len0 = SliceLengths{}[0];
            constexpr index_t Len1 = SliceLengths{}[1];
            constexpr index_t Len2 = SliceLengths{}[2];
            constexpr index_t Len3 = SliceLengths{}[3];

#pragma unroll
            for(index_t iter0 = 0; iter0 < Len0; ++iter0)
            {
#pragma unroll
                for(index_t iter1 = 0; iter1 < Len1; ++iter1)
                {
#pragma unroll
                    for(index_t iter2 = 0; iter2 < Len2; ++iter2)
                    {
#pragma unroll
                        for(index_t iter3 = 0; iter3 < Len3; ++iter3)
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
                                src_slice_origin_.GetOffset(),
                                coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                    src_desc, src_slice_origin_),
                                src_desc.GetElementSpaceSize(),
                                p_dst,
                                dst_slice_origin_.GetOffset(),
                                coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                    dst_desc, dst_slice_origin_),
                                dst_desc.GetElementSpaceSize());

                            // move dim1 iterator
                            if(iter3 < Len3 - 1)
                            {
                                bool forward_dim3 = (iter2 % 2 == 0);

                                if(forward_dim3)
                                {
                                    move_dynamic_tensor_coordinate(
                                        src_desc, src_slice_origin_, src_step_0_0_0_p1);
                                    move_dynamic_tensor_coordinate(
                                        dst_desc, dst_slice_origin_, dst_step_0_0_0_p1);
                                }
                                else
                                {
                                    move_dynamic_tensor_coordinate(
                                        src_desc, src_slice_origin_, src_step_0_0_0_m1);
                                    move_dynamic_tensor_coordinate(
                                        dst_desc, dst_slice_origin_, dst_step_0_0_0_m1);
                                }
                            }
                        }

                        // move dim1 iterator
                        if(iter2 < Len2 - 1)
                        {
                            bool forward_dim2 = (iter1 % 2 == 0);

                            if(forward_dim2)
                            {
                                move_dynamic_tensor_coordinate(
                                    src_desc, src_slice_origin_, src_step_0_0_p1_0);
                                move_dynamic_tensor_coordinate(
                                    dst_desc, dst_slice_origin_, dst_step_0_0_p1_0);
                            }
                            else
                            {
                                move_dynamic_tensor_coordinate(
                                    src_desc, src_slice_origin_, src_step_0_0_m1_0);
                                move_dynamic_tensor_coordinate(
                                    dst_desc, dst_slice_origin_, dst_step_0_0_m1_0);
                            }
                        }
                    }

                    // move dim1 iterator
                    if(iter1 < Len1 - 1)
                    {
                        bool forward_dim1 = (iter0 % 2 == 0);

                        if(forward_dim1)
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_p1_0_0);
                            move_dynamic_tensor_coordinate(
                                dst_desc, dst_slice_origin_, dst_step_0_p1_0_0);
                        }
                        else
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_m1_0_0);
                            move_dynamic_tensor_coordinate(
                                dst_desc, dst_slice_origin_, dst_step_0_m1_0_0);
                        }
                    }
                }

                // move dim0 iterator:
                if(iter0 < Len0 - 1)
                {
                    // move forward in dim0
                    move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_p1_0_0_0);
                    move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_step_p1_0_0_0);
                }
            }
        }

        // move src and dst coordinate back to their origins
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_back_step =
                make_dynamic_tensor_coordinate_step(src_desc, GetCoordinateBackStep());

            move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_back_step);
        }

        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_back_step =
                make_dynamic_tensor_coordinate_step(dst_desc, GetCoordinateBackStep());

            move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_back_step);
        }
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // is it OK to construct a new step every time?
        const auto src_slice_origin_step =
            make_dynamic_tensor_coordinate_step(src_desc, src_slice_origin_step_idx);

        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_slice_origin_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // is it OK to construct a new step every time?
        const auto dst_slice_origin_step =
            make_dynamic_tensor_coordinate_step(dst_desc, dst_slice_origin_step_idx);

        move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_slice_origin_step);
    }

    private:
    SrcCoord src_slice_origin_;
    DstCoord dst_slice_origin_;
};

// this version does following things to avoid "alloca" in LLVM-IR, which would cause scratch memory
// and sometimes useless instructions
// 1. It does not keep reference to tensor descriptor
// 2. It does not construct new tensor coordinate for this->Run()
// 3. It does not use pointer for VGPR thread buffer
// 4. It calculate offset for thread buffer directly, instead of moving the coordinate
template <typename SliceLengths,
          InMemoryDataOperation DstInMemOp,
          typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          bool SrcResetCoordinateAfterRun, // control whether to move back src coordinate after each
                                           // RunRead(),  will be fused with MoveSrcSliceWindow to
                                           // save addr computation
          bool DstResetCoordinateAfterRun> // control whether to move back dst coordinate after each
                                           // RunWrite(),  will be fused with MoveDstSliceWindow to
                                           // save addr computation
struct ThreadwiseDynamicTensorSliceTransfer_v3
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = decltype(make_dynamic_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_dynamic_tensor_coordinate(DstDesc{}, Index{}));

    using SrcCoordStep = decltype(make_dynamic_tensor_coordinate_step(SrcDesc{}, Index{}));
    using DstCoordStep = decltype(make_dynamic_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseDynamicTensorSliceTransfer_v3(const SrcDesc& src_desc,
                                                                 const Index& src_slice_origin,
                                                                 const DstDesc& dst_desc,
                                                                 const Index& dst_slice_origin)
        : src_slice_origin_(make_dynamic_tensor_coordinate(src_desc, src_slice_origin)),
          dst_slice_origin_(make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin))
    {
        static_assert(SrcAddressSpace == AddressSpace::Global or
                          SrcAddressSpace == AddressSpace::Lds,
                      "wrong!");
        static_assert(DstAddressSpace == AddressSpace::Global or
                          DstAddressSpace == AddressSpace::Lds,
                      "wrong!");
    }

    __device__ constexpr ThreadwiseDynamicTensorSliceTransfer_v3()
        : ThreadwiseDynamicTensorSliceTransfer_v3(
              SrcDesc{}, make_zero_multi_index<nDim>(), DstDesc{}, make_zero_multi_index<nDim>())
    {
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_slice_origin_ = make_dynamic_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_slice_origin_ = make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    __device__ void RunRead(const SrcDesc& src_desc, const SrcData* p_src)
    {
        static_assert(remove_reference_t<SrcDesc>::GetNumOfDimension() == 2,
                      "wrong! hardcoded for 2D tensor");

        // hardcoded for 2D
        // TODO implemente N-D
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

            constexpr index_t Len0 = SliceLengths{}[0];
            constexpr index_t Len1 = SliceLengths{}[1];

            static_for<0, Len0, 1>{}([&](auto iter0) {
                static_for<0, Len1, 1>{}([&](auto iter1) {
                    // step direction
                    constexpr bool forward_dim1 = (iter0.value % 2 == 0);

                    constexpr index_t i0 = iter0.value;
                    constexpr index_t i1 = forward_dim1 ? iter1.value : Len1 - iter1.value - 1;

                    // do work
                    constexpr index_t buffer_offset =
                        buffer_desc_.CalculateOffset(make_multi_index(i0, i1));

                    // hardcoding for buffer_load
                    // TODO refactor transfer_data() to encapsulate this
                    static_assert(SrcAddressSpace == AddressSpace::Global,
                                  "wrong! hardcoded to use buffer_load, src must be global mem");

                    buffer_(Number<buffer_offset>{}) = amd_buffer_load<SrcData, 1>(
                        p_src,
                        src_slice_origin_.GetOffset(),
                        coordinate_has_valid_offset_assuming_visible_index_is_valid(
                            src_desc, src_slice_origin_),
                        src_desc.GetElementSpaceSize());

                    // move dim1 iterator
                    if constexpr(iter1.value < Len1 - 1)
                    {
                        if constexpr(forward_dim1)
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_p1);
                        }
                        else
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_m1);
                        }
                    }
                });

                // move dim0 iterator
                if constexpr(iter0.value < Len0 - 1)
                {
                    move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_p1_0);
                }
            });
        }

        // move src coordinate back to its slice origin
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_back_step =
                make_dynamic_tensor_coordinate_step(src_desc, GetCoordinateBackStep());

            move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_back_step);
        }
    }

    __device__ void RunWrite(const DstDesc& dst_desc, DstData* p_dst)
    {
        static_assert(remove_reference_t<DstDesc>::GetNumOfDimension() == 2,
                      "wrong! hardcoded for 2D tensor");

        // hardcoded for 2D
        // TODO implement N-D
        if constexpr(remove_reference_t<SrcDesc>::GetNumOfDimension() == 2)
        {
            // TODO use constexpr for coordinate-step to make sure compiler behave correctly
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

            static_for<0, Len0, 1>{}([&](auto iter0) {
                static_for<0, Len1, 1>{}([&](auto iter1) {
                    // step direction
                    constexpr bool forward_dim1 = (iter0.value % 2 == 0);

                    constexpr index_t i0 = iter0;
                    constexpr index_t i1 = forward_dim1 ? iter1.value : Len1 - iter1.value - 1;

                    // do work
                    constexpr index_t buffer_offset =
                        buffer_desc_.CalculateOffset(make_multi_index(i0, i1));

                    // hardcoding for ds_write
                    // TODO refactor transfer_data() to encapsulate this
                    static_assert(DstAddressSpace == AddressSpace::Lds &&
                                      DstInMemOp == InMemoryDataOperation::Set,
                                  "wrong! hardcoded for ds_write");

                    p_dst[dst_slice_origin_.GetOffset()] = buffer_[Number<buffer_offset>{}];

                    // move dim1 iterator
                    if constexpr(iter1.value < Len1 - 1)
                    {
                        if constexpr(forward_dim1)
                        {
                            move_dynamic_tensor_coordinate(
                                dst_desc, dst_slice_origin_, dst_step_0_p1);
                        }
                        else
                        {
                            move_dynamic_tensor_coordinate(
                                dst_desc, dst_slice_origin_, dst_step_0_m1);
                        }
                    }
                });

                // move dim0 iterator
                if constexpr(iter0.value < Len0 - 1)
                {
                    move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_step_p1_0);
                }
            });
        }

        // move dst coordinate back to its slice origin
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_back_step =
                make_dynamic_tensor_coordinate_step(dst_desc, GetCoordinateBackStep());

            move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_back_step);
        }
    }

    __device__ void RunRead_hack(const SrcDesc& src_desc, const SrcData* p_src)
    {
        static_assert(remove_reference_t<SrcDesc>::GetNumOfDimension() == 2,
                      "wrong! hardcoded for 2D tensor");

        // hardcoded for 2D
        // TODO implemente N-D
        if constexpr(remove_reference_t<SrcDesc>::GetNumOfDimension() == 2)
        {
#if 0 // hack
      // TODO use constexpr for coordinate-step to make sure compiler behave correctly
            const auto src_step_0_p1 =
                make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(0, 1));
            const auto src_step_0_m1 =
                make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(0, -1));

            const auto src_step_p1_0 =
                make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(1, 0));
            const auto src_step_m1_0 =
                make_dynamic_tensor_coordinate_step(src_desc, make_multi_index(-1, 0));
#elif 1
            // for padded input tensor
            const auto src_step_0_p1 = make_dynamic_tensor_coordinate_step_hack(
                src_desc, make_multi_index(0, 1), Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1>{});
            const auto src_step_0_m1 = make_dynamic_tensor_coordinate_step_hack(
                src_desc, make_multi_index(0, -1), Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2>{});

            const auto src_step_p1_0 = make_dynamic_tensor_coordinate_step_hack(
                src_desc, make_multi_index(1, 0), Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0>{});
            const auto src_step_m1_0 = make_dynamic_tensor_coordinate_step_hack(
                src_desc, make_multi_index(-1, 0), Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0>{});
#elif 1
            // for non-padded input tensor
            const auto src_step_0_p1 = make_dynamic_tensor_coordinate_step_hack(
                src_desc, make_multi_index(0, 1), Sequence<0, 0, 0, 0, 0, 0, 1>{});
            const auto src_step_0_m1 = make_dynamic_tensor_coordinate_step_hack(
                src_desc, make_multi_index(0, -1), Sequence<0, 0, 0, 0, 0, 0, 2>{});

            const auto src_step_p1_0 = make_dynamic_tensor_coordinate_step_hack(
                src_desc, make_multi_index(1, 0), Sequence<0, 0, 0, 0, 0, 1, 0>{});
            const auto src_step_m1_0 = make_dynamic_tensor_coordinate_step_hack(
                src_desc, make_multi_index(-1, 0), Sequence<0, 0, 0, 0, 0, 2, 0>{});
#endif

            constexpr index_t Len0 = SliceLengths{}[0];
            constexpr index_t Len1 = SliceLengths{}[1];

            static_for<0, Len0, 1>{}([&](auto iter0) {
                static_for<0, Len1, 1>{}([&](auto iter1) {
                    // step direction
                    constexpr bool forward_dim1 = (iter0.value % 2 == 0);

                    constexpr index_t i0 = iter0.value;
                    constexpr index_t i1 = forward_dim1 ? iter1.value : Len1 - iter1.value - 1;

                    // do work
                    constexpr index_t buffer_offset =
                        buffer_desc_.CalculateOffset(make_multi_index(i0, i1));

                    // hardcoding for buffer_load
                    // TODO refactor transfer_data() to encapsulate this
                    static_assert(SrcAddressSpace == AddressSpace::Global,
                                  "wrong! hardcoded to use buffer_load, src must be global mem");

#if 0 // debug
                    buffer_(Number<buffer_offset>{}) = amd_buffer_load<SrcData, 1>(
                        p_src,
                        src_slice_origin_.GetOffset(),
                        coordinate_has_valid_offset_assuming_visible_index_is_valid(
                            src_desc, src_slice_origin_),
                        src_desc.GetElementSpaceSize());
#else
                    SrcData tmp = amd_buffer_load<SrcData, 1>(
                        p_src, src_slice_origin_.GetOffset(), true, src_desc.GetElementSpaceSize());

                    const bool is_valid =
                        coordinate_has_valid_offset_assuming_visible_index_is_valid(
                            src_desc, src_slice_origin_);

                    buffer_(Number<buffer_offset>{}) = is_valid ? tmp : SrcData{0};
#endif

                    // move dim1 iterator
                    if constexpr(iter1.value < Len1 - 1)
                    {
                        if constexpr(forward_dim1)
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_p1);
                        }
                        else
                        {
                            move_dynamic_tensor_coordinate(
                                src_desc, src_slice_origin_, src_step_0_m1);
                        }
                    }
                });

                // move dim0 iterator
                if constexpr(iter0.value < Len0 - 1)
                {
                    move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_p1_0);
                }
            });
        }

        // move src coordinate back to its slice origin
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_back_step =
                make_dynamic_tensor_coordinate_step(src_desc, GetCoordinateBackStep());

            move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_back_step);
        }
    }

    __device__ static constexpr auto GetCoordinateBackStep()
    {
        MultiIndex<nDim> back_step;

        back_step(Number<0>{}) = 1 - SliceLengths{}[0];

        static_for<1, nDim, 1>{}([&](auto i) {
            back_step(i) = (SliceLengths{}[i - Number<1>{}] % 2 == 0) ? 0 : (1 - SliceLengths{}[i]);
        });

        return back_step;
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx = SrcResetCoordinateAfterRun
                                           ? src_slice_origin_step_idx
                                           : src_slice_origin_step_idx + GetCoordinateBackStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_dynamic_tensor_coordinate_step(src_desc, adjusted_step_idx);

        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, adjusted_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by RunWrite(), then need to adjust the step here
        const auto adjusted_step_idx = DstResetCoordinateAfterRun
                                           ? dst_slice_origin_step_idx
                                           : dst_slice_origin_step_idx + GetCoordinateBackStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_dynamic_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, adjusted_step);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow_hack(const SrcDesc& src_desc,
                                            const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx = SrcResetCoordinateAfterRun
                                           ? src_slice_origin_step_idx
                                           : src_slice_origin_step_idx + GetCoordinateBackStep();

        // is it OK to construct a new step every time?
#if 0 // hack
        const auto adjusted_step = make_dynamic_tensor_coordinate_step(
            src_desc, adjusted_step_idx);
#elif 1
        // for padded input tensor
        const auto adjusted_step = make_dynamic_tensor_coordinate_step_hack(
            src_desc, adjusted_step_idx, Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2>{});
#elif 1
        // for non-paded input tensor
        const auto adjusted_step = make_dynamic_tensor_coordinate_step_hack(
            src_desc, adjusted_step_idx, Sequence<0, 0, 0, 0, 0, 1, 2>{});
#endif

        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, adjusted_step);
    }

    private:
    static constexpr auto buffer_desc_ =
        make_dynamic_naive_tensor_descriptor_packed<nDim>(to_multi_index(SliceLengths{}));

    static constexpr index_t buffer_size_ = buffer_desc_.GetElementSpaceSize();

    StaticallyIndexedArray<SrcData, buffer_size_> buffer_;

    SrcCoord src_slice_origin_;
    DstCoord dst_slice_origin_;
};

} // namespace ck
#endif
