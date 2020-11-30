#ifndef CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP
#define CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

// this version tends to have scratch memory issue, due to:
// 1. It keeps reference to tensor descriptor
// 2. It constructs new tensor coordinate in this->Run()
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
        // comment: construction tensor coordinate here tends to cause scratch memory issue
        auto src_coord = src_slice_origin_;
        auto dst_coord = dst_slice_origin_;

        // TODO use constexpr for coordinate-step to make sure compiler behave correctly
        const auto src_step_0_p1 =
            make_dynamic_tensor_coordinate_step(src_desc_, make_multi_index(0, 1));
        const auto src_step_0_m1 =
            make_dynamic_tensor_coordinate_step(src_desc_, make_multi_index(0, -1));
        const auto src_step_p1_0 =
            make_dynamic_tensor_coordinate_step(src_desc_, make_multi_index(1, 0));
        const auto src_step_m1_0 =
            make_dynamic_tensor_coordinate_step(src_desc_, make_multi_index(-1, 0));

        const auto dst_step_0_p1 =
            make_dynamic_tensor_coordinate_step(dst_desc_, make_multi_index(0, 1));
        const auto dst_step_0_m1 =
            make_dynamic_tensor_coordinate_step(dst_desc_, make_multi_index(0, -1));
        const auto dst_step_p1_0 =
            make_dynamic_tensor_coordinate_step(dst_desc_, make_multi_index(1, 0));
        const auto dst_step_m1_0 =
            make_dynamic_tensor_coordinate_step(dst_desc_, make_multi_index(-1, 0));

        constexpr index_t Len0 = SliceLengths{}[0];
        constexpr index_t Len1 = SliceLengths{}[1];

        bool forward_dim0 = true;
        bool forward_dim1 = true;

        // hardcoded for 2d loop for now
#pragma unroll
        for(index_t i0 = 0; i0 < Len0; ++i0)
        {
#pragma unroll
            for(index_t i1 = 0; i1 < Len1; ++i1)
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
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc_,
                                                                                src_coord),
                    src_desc_.GetElementSpaceSize(),
                    p_dst,
                    dst_coord.GetOffset(),
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc_,
                                                                                dst_coord),
                    dst_desc_.GetElementSpaceSize());

                // move dim1 iterator
                if(i1 < Len1 - 1)
                {
                    if(forward_dim1)
                    {
                        move_dynamic_tensor_coordinate(src_desc_, src_coord, src_step_0_p1);
                        move_dynamic_tensor_coordinate(dst_desc_, dst_coord, dst_step_0_p1);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(src_desc_, src_coord, src_step_0_m1);
                        move_dynamic_tensor_coordinate(dst_desc_, dst_coord, dst_step_0_m1);
                    }
                }
            }

            // switch dim1 iteration direction
            forward_dim1 = !forward_dim1;

            // move dim0 iterator
            if(i0 < Len0 - 1)
            {
                if(forward_dim0)
                {
                    move_dynamic_tensor_coordinate(src_desc_, src_coord, src_step_p1_0);
                    move_dynamic_tensor_coordinate(dst_desc_, dst_coord, dst_step_p1_0);
                }
                else
                {
                    move_dynamic_tensor_coordinate(src_desc_, src_coord, src_step_m1_0);
                    move_dynamic_tensor_coordinate(dst_desc_, dst_coord, dst_step_m1_0);
                }
            }
        }
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
        // is it OK to construct a new step every time?
        const auto src_slice_origin_step =
            make_dynamic_tensor_coordinate_step(src_desc_, src_slice_origin_step_idx);

        move_dynamic_tensor_coordinate(src_desc_, src_slice_origin_, src_slice_origin_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const Index& dst_slice_origin_step_idx)
    {
        // is it OK to construct a new step every time?
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

// this version is less likely to have scratch memory issue, due to:
// 1. It does not keep reference to tensor descriptor
// 2. It does not construct new tensor coordinate for this->Run()
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
          index_t DstScalarStrideInVector,
          bool MoveBackSrcCoord = true,
          bool MoveBackDstCoord = true>
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
            for(index_t i0 = 0; i0 < Len0; ++i0)
            {
#pragma unroll
                for(index_t i1 = 0; i1 < Len1; ++i1)
                {
#if 1 // debug
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
#else
                    if constexpr(SrcAddressSpace == AddressSpace::Global &&
                                 DstAddressSpace == AddressSpace::Vgpr)
                    {
                        if(coordinate_has_valid_offset_assuming_visible_index_is_valid(
                               dst_desc, dst_slice_origin_))
                        {
                            const SrcData tmp = amd_buffer_load<SrcData, 1>(
                                p_src,
                                src_slice_origin_.GetOffset(),
                                coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                    src_desc, src_slice_origin_),
                                src_desc.GetElementSpaceSize());

                            const index_t dst_offset = dst_slice_origin_.GetOffset();

                            p_dst[dst_offset] = tmp;
                        }
                    }
                    else if constexpr(SrcAddressSpace == AddressSpace::Vgpr &&
                                      DstAddressSpace == AddressSpace::Global)
                    {
                        const SrcData zeros = 0;

                        const bool src_valid =
                            coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                src_desc, src_slice_origin_);

                        const bool dst_valid =
                            coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                dst_desc, dst_slice_origin_);

                        amd_buffer_store<SrcData, 1>(
                            src_valid ? &(p_src[src_slice_origin_.GetOffset()]) : &zeros,
                            p_dst,
                            dst_slice_origin_.GetOffset(),
                            dst_valid,
                            dst_desc.GetElementSpaceSize());
                    }
                    else
                    {
                        if(coordinate_has_valid_offset_assuming_visible_index_is_valid(
                               dst_desc, dst_slice_origin_))
                        {
                            if(coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                   src_desc, src_slice_origin_))
                            {
                                p_dst[dst_slice_origin_.GetOffset()] =
                                    p_src[src_slice_origin_.GetOffset()];
                            }
                            else
                            {
                                p_dst[dst_slice_origin_.GetOffset()] = 0;
                            }
                        }
                    }
#endif

                    // move dim1 iterator
                    if(i1 < Len1 - 1)
                    {
                        bool forward_dim1 = (i0 % 2 == 0);

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
                if(i0 < Len0 - 1)
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

            bool forward_dim0 = true;
            bool forward_dim1 = true;
            bool forward_dim2 = true;
            bool forward_dim3 = true;

#pragma unroll
            for(index_t i0 = 0; i0 < Len0; ++i0)
            {
#pragma unroll
                for(index_t i1 = 0; i1 < Len1; ++i1)
                {
#pragma unroll
                    for(index_t i2 = 0; i2 < Len2; ++i2)
                    {
#pragma unroll
                        for(index_t i3 = 0; i3 < Len3; ++i3)
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
                            if(i3 < Len3 - 1)
                            {
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

                        // switch dim3 iteration direction
                        forward_dim3 = !forward_dim3;

                        // move dim1 iterator
                        if(i2 < Len2 - 1)
                        {
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

                    // switch dim2 iteration direction
                    forward_dim2 = !forward_dim2;

                    // move dim1 iterator
                    if(i1 < Len1 - 1)
                    {
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

                // switch dim1 iteration direction
                forward_dim1 = !forward_dim1;

                // move dim0 iterator
                if(i0 < Len0 - 1)
                {
                    if(forward_dim0)
                    {
                        move_dynamic_tensor_coordinate(
                            src_desc, src_slice_origin_, src_step_p1_0_0_0);
                        move_dynamic_tensor_coordinate(
                            dst_desc, dst_slice_origin_, dst_step_p1_0_0_0);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(
                            src_desc, src_slice_origin_, src_step_m1_0_0_0);
                        move_dynamic_tensor_coordinate(
                            dst_desc, dst_slice_origin_, dst_step_m1_0_0_0);
                    }
                }
            }
        }

        // move src and dst coordinate back to their origins
        if constexpr(MoveBackSrcCoord)
        {
            const auto src_step_back =
                make_dynamic_tensor_coordinate_step(src_desc, GetCoordinateStepBack());

            move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_back);
        }

        if constexpr(MoveBackDstCoord)
        {
            const auto dst_step_back =
                make_dynamic_tensor_coordinate_step(dst_desc, GetCoordinateStepBack());

            move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_step_back);
        }
    }

    __device__ static constexpr auto GetCoordinateStepBack()
    {
        MultiIndex<nDim> step_back;

        step_back(Number<0>{}) = 1 - SliceLengths{}[0];

        static_for<1, nDim, 1>{}([&](auto i) {
            step_back(i) = (SliceLengths{}[i - Number<1>{}] % 2 == 0) ? 0 : (1 - SliceLengths{}[i]);
        });

        return step_back;
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

} // namespace ck
#endif
