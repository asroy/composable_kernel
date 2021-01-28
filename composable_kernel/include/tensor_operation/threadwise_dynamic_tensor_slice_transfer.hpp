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
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          InMemoryDataOperation DstInMemOp,
          index_t DstScalarStrideInVector,
          bool SrcResetCoordinateAfterRun,
          bool DstResetCoordinateAfterRun>
struct ThreadwiseDynamicTensorSliceTransfer_v1r3
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using DstCoord = decltype(make_dynamic_tensor_coordinate(DstDesc{}, Index{}));

    using DstCoordStep = decltype(make_dynamic_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseDynamicTensorSliceTransfer_v1r3(
        const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
        : dst_slice_origin_coord_(make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_idx))
    {
    }

    __device__ constexpr ThreadwiseDynamicTensorSliceTransfer_v1r3()
        : ThreadwiseDynamicTensorSliceTransfer_v1r3(DstDesc{}, make_zero_multi_index<nDim>())
    {
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_slice_origin_coord_ = make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    __device__ void Run_hack(const SrcData* p_src, const DstDesc& dst_desc, DstData* p_dst)
    {
        // hardcoded for 4D
        // TODO implemente N-D
        static_assert(remove_reference_t<SrcDesc>::GetNumOfDimension() == 4,
                      "wrong! hardcoded for 4D tensor");

        constexpr auto dst_scalar_per_access = [&]() {
            Index dst_scalar_per_access;

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(i == DstVectorDim)
                {
                    dst_scalar_per_access(i) = DstScalarPerVector;
                }
                else
                {
                    dst_scalar_per_access(i) = 1;
                }
            });

            return dst_scalar_per_access;
        }();

        constexpr auto dst_scalar_step_in_vector = [&]() {
            Index dst_scalar_step_in_vector;

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(i == DstVectorDim)
                {
                    dst_scalar_step_in_vector(i) = 1;
                }
                else
                {
                    dst_scalar_step_in_vector(i) = 0;
                }
            });

            return dst_scalar_step_in_vector;
        }();

        constexpr auto access_lengths = [&]() {
            Index access_lengths;

            static_for<0, nDim, 1>{}(
                [&](auto i) { access_lengths(i) = SliceLengths{}[i] / dst_scalar_per_access[i]; });

            return access_lengths;
        }();

#if 0
        const auto dst_forward_steps =
            make_tuple(make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(1, 0, 0, 0) * dst_scalar_per_access),
                       make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(0, 1, 0, 0) * dst_scalar_per_access),
                       make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(0, 0, 1, 0) * dst_scalar_per_access),
                       make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(0, 0, 0, 1) * dst_scalar_per_access),

        const auto dst_backward_steps =
            make_tuple(make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(-1, 0, 0, 0) * dst_scalar_per_access),
                       make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(0, -1, 0, 0) * dst_scalar_per_access),
                       make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(0, 0, -1, 0) * dst_scalar_per_access),
                       make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(0, 0, 0, -1) * dst_scalar_per_access));
#else
        // hack for NKHW output tensor
        const auto dst_forward_steps =
            make_tuple(make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(1, 0, 0, 0) * dst_scalar_per_access),
                       make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(0, 1, 0, 0) * dst_scalar_per_access),
                       make_dynamic_tensor_coordinate_step_hack(dst_desc,
                                                                make_multi_index(0, 0, 1, 0) *
                                                                    dst_scalar_per_access,
                                                                Sequence<0, 0, 1, 0, 0>{}),
                       make_dynamic_tensor_coordinate_step_hack(dst_desc,
                                                                make_multi_index(0, 0, 0, 1) *
                                                                    dst_scalar_per_access,
                                                                Sequence<0, 0, 1, 0, 0>{}));

        const auto dst_backward_steps =
            make_tuple(make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(-1, 0, 0, 0) * dst_scalar_per_access),
                       make_dynamic_tensor_coordinate_step(
                           dst_desc, make_multi_index(0, -1, 0, 0) * dst_scalar_per_access),
                       make_dynamic_tensor_coordinate_step_hack(dst_desc,
                                                                make_multi_index(0, 0, -1, 0) *
                                                                    dst_scalar_per_access,
                                                                Sequence<0, 0, 2, 0, 0>{}),
                       make_dynamic_tensor_coordinate_step_hack(dst_desc,
                                                                make_multi_index(0, 0, 0, -1) *
                                                                    dst_scalar_per_access,
                                                                Sequence<0, 0, 2, 0, 0>{}));
#endif

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        index_t counter = 0;

        // loop over dim0
        static_for<0,
                   SliceLengths{}[DimAccessOrder{}[I0]],
                   dst_scalar_per_access[DimAccessOrder{}[I0]]>{}([&](auto iter0) {
            constexpr index_t i0 = iter0;

            constexpr bool forward_dim1 =
                (iter0 / dst_scalar_per_access[DimAccessOrder{}[I0]]) % 2 == 0;

            // loop over dim1
            static_for<0,
                       SliceLengths{}[DimAccessOrder{}[I1]],
                       dst_scalar_per_access[DimAccessOrder{}[I1]]>{}([&](auto iter1) {
                constexpr index_t i1 =
                    forward_dim1 ? iter1
                                 : SliceLengths{}[DimAccessOrder{}[I1]] -
                                       dst_scalar_per_access[DimAccessOrder{}[I1]] - iter1;

                constexpr bool forward_dim2 =
                    ((iter0 / dst_scalar_per_access[DimAccessOrder{}[I0]]) *
                         access_lengths[DimAccessOrder{}[I1]] +
                     (iter1 / dst_scalar_per_access[DimAccessOrder{}[I1]])) %
                        2 ==
                    0;

                // loop over dim2
                static_for<0,
                           SliceLengths{}[DimAccessOrder{}[I2]],
                           dst_scalar_per_access[DimAccessOrder{}[I2]]>{}([&](auto iter2) {
                    constexpr index_t i2 =
                        forward_dim2 ? iter2
                                     : SliceLengths{}[DimAccessOrder{}[I2]] -
                                           dst_scalar_per_access[DimAccessOrder{}[I2]] - iter2;

                    constexpr bool forward_dim3 =
                        (((iter0 / dst_scalar_per_access[DimAccessOrder{}[I0]]) *
                              access_lengths[DimAccessOrder{}[I1]] +
                          (iter1 / dst_scalar_per_access[DimAccessOrder{}[I1]])) *
                             access_lengths[DimAccessOrder{}[I2]] +
                         (iter2 / dst_scalar_per_access[DimAccessOrder{}[I2]])) %
                            2 ==
                        0;

                    // loop over dim3
                    static_for<0,
                               SliceLengths{}[DimAccessOrder{}[I3]],
                               dst_scalar_per_access[DimAccessOrder{}[I3]]>{}([&](auto iter3) {
                        constexpr index_t i3 =
                            forward_dim3 ? iter3
                                         : SliceLengths{}[DimAccessOrder{}[I3]] -
                                               dst_scalar_per_access[DimAccessOrder{}[I3]] - iter3;

                        // do work
                        // hardcoding for buffer_store
                        // TODO refactor transfer_data() to encapsulate this
                        static_assert(SrcAddressSpace == AddressSpace::Vgpr &&
                                          DstAddressSpace == AddressSpace::Global,
                                      "wrong! hardcoded to use buffer_store");

                        using DstVectorType =
                            typename vector_type<DstData, DstScalarPerVector>::MemoryType;

                        vector_type<DstData, DstScalarPerVector> dst_vector;

                        // this is hardcoded for src that has compile-time tensor descriptor
                        static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                            // hack: assume src_slice_origin_idx is 0
                            constexpr index_t src_offset = SrcDesc::CalculateOffset(
                                container_reorder_given_old2new(make_multi_index(i0, i1, i2, i3),
                                                                DimAccessOrder{}) +
                                i * dst_scalar_step_in_vector);

                            dst_vector(i) = p_src[Number<src_offset>{}];
                        });

                        amd_buffer_store_v2<DstData, DstScalarPerVector>(
                            dst_vector.Vector(),
                            p_dst,
                            dst_slice_origin_coord_.GetOffset(),
                            coordinate_has_valid_offset_assuming_visible_index_is_valid(
                                dst_desc, dst_slice_origin_coord_),
                            dst_desc.GetElementSpaceSize());

                        // move along dim3
                        if constexpr(iter3 < SliceLengths{}[DimAccessOrder{}[I3]] -
                                                 dst_scalar_per_access[DimAccessOrder{}[I3]])
                        {
                            if constexpr(forward_dim3)
                            {
                                move_dynamic_tensor_coordinate(
                                    dst_desc,
                                    dst_slice_origin_coord_,
                                    dst_forward_steps[DimAccessOrder{}[I3]]);
                            }
                            else
                            {
                                move_dynamic_tensor_coordinate(
                                    dst_desc,
                                    dst_slice_origin_coord_,
                                    dst_backward_steps[DimAccessOrder{}[I3]]);
                            }
                        }
                    });

                    // move along dim2
                    if constexpr(iter2 < SliceLengths{}[DimAccessOrder{}[I2]] -
                                             dst_scalar_per_access[DimAccessOrder{}[I2]])
                    {
                        if constexpr(forward_dim2)
                        {
                            move_dynamic_tensor_coordinate(dst_desc,
                                                           dst_slice_origin_coord_,
                                                           dst_forward_steps[DimAccessOrder{}[I2]]);
                        }
                        else
                        {
                            move_dynamic_tensor_coordinate(
                                dst_desc,
                                dst_slice_origin_coord_,
                                dst_backward_steps[DimAccessOrder{}[I2]]);
                        }
                    }
                });

                // move along dim1
                if constexpr(iter1 < SliceLengths{}[DimAccessOrder{}[I1]] -
                                         dst_scalar_per_access[DimAccessOrder{}[I1]])
                {
                    if constexpr(forward_dim1)
                    {
                        move_dynamic_tensor_coordinate(dst_desc,
                                                       dst_slice_origin_coord_,
                                                       dst_forward_steps[DimAccessOrder{}[I1]]);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(dst_desc,
                                                       dst_slice_origin_coord_,
                                                       dst_backward_steps[DimAccessOrder{}[I1]]);
                    }
                }
            });

            // move along dim0
            if constexpr(iter0 < SliceLengths{}[DimAccessOrder{}[I0]] -
                                     dst_scalar_per_access[DimAccessOrder{}[I0]])
            {
                move_dynamic_tensor_coordinate(
                    dst_desc, dst_slice_origin_coord_, dst_forward_steps[DimAccessOrder{}[I0]]);
            }
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_back_step =
                make_dynamic_tensor_coordinate_step(dst_desc, GetDstCoordinateBackStep());

            move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_coord_, dst_back_step);
        }
    }

    __device__ static constexpr auto GetDstCoordinateBackStep()
    {
        constexpr auto dst_scalar_per_access = [&]() {
            Index dst_scalar_per_access;

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(i == DstVectorDim)
                {
                    dst_scalar_per_access(i) = DstScalarPerVector;
                }
                else
                {
                    dst_scalar_per_access(i) = 1;
                }
            });

            return dst_scalar_per_access;
        }();

        MultiIndex<nDim> dst_back_step;

        // TODO: this is wrong, need to consider DimAccessOrder
        dst_back_step(Number<0>{}) = dst_scalar_per_access[Number<0>{}] - SliceLengths{}[0];

        static_for<1, nDim, 1>{}([&](auto i) {
            constexpr auto i_m1 = i - Number<1>{};

            // TODO: this is wrong
            dst_back_step(i) = (SliceLengths{}[i_m1] % (2 * dst_scalar_per_access[i_m1]) == 0)
                                   ? 0
                                   : (dst_scalar_per_access[i] - SliceLengths{}[i]);
        });

        return dst_back_step;
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by RunWrite(), then need to adjust the step here
        const auto adjusted_step_idx = DstResetCoordinateAfterRun
                                           ? dst_slice_origin_step_idx
                                           : dst_slice_origin_step_idx + GetDstCoordinateBackStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_dynamic_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_coord_, adjusted_step);
    }

    private:
    DstCoord dst_slice_origin_coord_;
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
        // hardcoded for 2D
        // TODO implemente N-D
        static_assert(remove_reference_t<SrcDesc>::GetNumOfDimension() == 2,
                      "wrong! hardcoded for 2D tensor");

        constexpr auto src_scalar_per_access = [&]() {
            Index src_scalar_per_access;

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    src_scalar_per_access(i) = SrcScalarPerVector;
                }
                else
                {
                    src_scalar_per_access(i) = 1;
                }
            });

            return src_scalar_per_access;
        }();

        constexpr auto src_scalar_step_in_vector = [&]() {
            Index src_scalar_step_in_vector;

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    src_scalar_step_in_vector(i) = 1;
                }
                else
                {
                    src_scalar_step_in_vector(i) = 0;
                }
            });

            return src_scalar_step_in_vector;
        }();

        // TODO use constexpr for coordinate-step to make sure compiler behave correctly
        const auto src_step_0_p = make_dynamic_tensor_coordinate_step(
            src_desc, make_multi_index(0, 1) * src_scalar_per_access);

        const auto src_step_0_m = make_dynamic_tensor_coordinate_step(
            src_desc, make_multi_index(0, -1) * src_scalar_per_access);

        const auto src_step_p_0 = make_dynamic_tensor_coordinate_step(
            src_desc, make_multi_index(1, 0) * src_scalar_per_access);

        const auto src_step_m_0 = make_dynamic_tensor_coordinate_step(
            src_desc, make_multi_index(-1, 0) * src_scalar_per_access);

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        static_for<0, SliceLengths{}[I0], src_scalar_per_access[I0]>{}([&](auto iter0) {
            static_for<0, SliceLengths{}[I1], src_scalar_per_access[I1]>{}([&](auto iter1) {
                // step direction
                constexpr bool forward_dim1 = (iter0.value % (2 * src_scalar_per_access[I0]) == 0);

                constexpr index_t i0 = iter0.value;
                constexpr index_t i1 =
                    forward_dim1 ? iter1.value
                                 : SliceLengths{}[I1] - src_scalar_per_access[I1] - iter1.value;

                // do work
                // hardcoding for buffer_load
                // TODO refactor transfer_data() to encapsulate this
                static_assert(SrcAddressSpace == AddressSpace::Global,
                              "wrong! hardcoded to use buffer_load, src must be global mem");

                vector_type<SrcData, SrcScalarPerVector> src_vector;

                using SrcVectorType = typename vector_type<SrcData, SrcScalarPerVector>::MemoryType;

                src_vector.Vector() = amd_buffer_load<SrcData, SrcScalarPerVector>(
                    p_src, src_slice_origin_.GetOffset(), true, src_desc.GetElementSpaceSize());

                const bool is_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                    src_desc, src_slice_origin_);

                src_vector.Vector() = is_valid ? src_vector.Vector() : SrcVectorType{0};

                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t buffer_offset = buffer_desc_.CalculateOffset(
                        make_multi_index(i0, i1) + i * src_scalar_step_in_vector);

                    buffer_(Number<buffer_offset>{}) = src_vector[i];
                });

                // move dim1 iterator
                if constexpr(iter1.value < SliceLengths{}[I1] - src_scalar_per_access[I1])
                {
                    if constexpr(forward_dim1)
                    {
                        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_0_p);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_0_m);
                    }
                }
            });

            // move dim0 iterator
            if constexpr(iter0.value < SliceLengths{}[I0] - src_scalar_per_access[I0])
            {
                move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_p_0);
            }
        });

        // move src coordinate back to its slice origin
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_back_step =
                make_dynamic_tensor_coordinate_step(src_desc, GetSrcCoordinateBackStep());

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
            const auto dst_step_0_p =
                make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, 1));
            const auto dst_step_0_m =
                make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(0, -1));

            const auto dst_step_p_0 =
                make_dynamic_tensor_coordinate_step(dst_desc, make_multi_index(1, 0));
            const auto dst_step_m_0 =
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
                                dst_desc, dst_slice_origin_, dst_step_0_p);
                        }
                        else
                        {
                            move_dynamic_tensor_coordinate(
                                dst_desc, dst_slice_origin_, dst_step_0_m);
                        }
                    }
                });

                // move dim0 iterator
                if constexpr(iter0.value < Len0 - 1)
                {
                    move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_step_p_0);
                }
            });
        }

        // move dst coordinate back to its slice origin
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_back_step =
                make_dynamic_tensor_coordinate_step(dst_desc, GetDstCoordinateBackStep());

            move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_back_step);
        }
    }

    __device__ void RunRead_hack(const SrcDesc& src_desc, const SrcData* p_src)
    {
        // hardcoded for 2D
        // TODO implemente N-D
        static_assert(remove_reference_t<SrcDesc>::GetNumOfDimension() == 2,
                      "wrong! hardcoded for 2D tensor");

        constexpr auto src_scalar_per_access = [&]() {
            Index src_scalar_per_access;

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    src_scalar_per_access(i) = SrcScalarPerVector;
                }
                else
                {
                    src_scalar_per_access(i) = 1;
                }
            });

            return src_scalar_per_access;
        }();

        constexpr auto src_scalar_step_in_vector = [&]() {
            Index src_scalar_step_in_vector;

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    src_scalar_step_in_vector(i) = 1;
                }
                else
                {
                    src_scalar_step_in_vector(i) = 0;
                }
            });

            return src_scalar_step_in_vector;
        }();

        constexpr auto access_lengths = [&]() {
            Index access_lengths;

            static_for<0, nDim, 1>{}(
                [&](auto i) { access_lengths(i) = SliceLengths{}[i] / src_scalar_per_access[i]; });

            return access_lengths;
        }();

#if 0 // hack
      // TODO use constexpr for coordinate-step to make sure compiler behave correctly
        const auto src_step_0_p = make_dynamic_tensor_coordinate_step(
            src_desc, make_multi_index(0, 1) * src_scalar_per_access);

        const auto src_step_0_m = make_dynamic_tensor_coordinate_step(
            src_desc, make_multi_index(0, -1) * src_scalar_per_access);

        const auto src_step_p_0 = make_dynamic_tensor_coordinate_step(
            src_desc, make_multi_index(1, 0) * src_scalar_per_access);

        const auto src_step_m_0 = make_dynamic_tensor_coordinate_step(
            src_desc, make_multi_index(-1, 0) * src_scalar_per_access);
#elif 1
        // for padded input tensor
        const auto src_step_0_p =
            make_dynamic_tensor_coordinate_step_hack(src_desc,
                                                     make_multi_index(0, 1) * src_scalar_per_access,
                                                     Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1>{});
        const auto src_step_0_m = make_dynamic_tensor_coordinate_step_hack(
            src_desc,
            make_multi_index(0, -1) * src_scalar_per_access,
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2>{});

        const auto src_step_p_0 =
            make_dynamic_tensor_coordinate_step_hack(src_desc,
                                                     make_multi_index(1, 0) * src_scalar_per_access,
                                                     Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0>{});
        const auto src_step_m_0 = make_dynamic_tensor_coordinate_step_hack(
            src_desc,
            make_multi_index(-1, 0) * src_scalar_per_access,
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0>{});
#elif 0
        // for non-padded input tensor
        const auto src_step_0_p =
            make_dynamic_tensor_coordinate_step_hack(src_desc,
                                                     make_multi_index(0, 1) * src_scalar_per_access,
                                                     Sequence<0, 0, 0, 0, 0, 0, 1>{});

        const auto src_step_0_m = make_dynamic_tensor_coordinate_step_hack(
            src_desc,
            make_multi_index(0, -1) * src_scalar_per_access,
            Sequence<0, 0, 0, 0, 0, 0, 2>{});

        const auto src_step_p_0 =
            make_dynamic_tensor_coordinate_step_hack(src_desc,
                                                     make_multi_index(1, 0) * src_scalar_per_access,
                                                     Sequence<0, 0, 0, 0, 0, 1, 0>{});

        const auto src_step_m_0 = make_dynamic_tensor_coordinate_step_hack(
            src_desc,
            make_multi_index(-1, 0) * src_scalar_per_access,
            Sequence<0, 0, 0, 0, 0, 2, 0>{});
#elif 1
        // for 1x1, input tensor
        const auto src_step_0_p = make_dynamic_tensor_coordinate_step_hack(
            src_desc, make_multi_index(0, 1) * src_scalar_per_access, Sequence<0, 0, 1>{});

        const auto src_step_0_m = make_dynamic_tensor_coordinate_step_hack(
            src_desc, make_multi_index(0, -1) * src_scalar_per_access, Sequence<0, 0, 2>{});

        const auto src_step_p_0 = make_dynamic_tensor_coordinate_step(
            src_desc, make_multi_index(1, 0) * src_scalar_per_access);

        const auto src_step_m_0 = make_dynamic_tensor_coordinate_step(
            src_desc, make_multi_index(-1, 0) * src_scalar_per_access);
#endif

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        static_for<0, SliceLengths{}[I0], src_scalar_per_access[I0]>{}([&](auto iter0) {
            static_for<0, SliceLengths{}[I1], src_scalar_per_access[I1]>{}([&](auto iter1) {
                // step direction
                constexpr bool forward_dim1 = (iter0.value % (2 * src_scalar_per_access[I0]) == 0);

                constexpr index_t i0 = iter0.value;
                constexpr index_t i1 =
                    forward_dim1 ? iter1.value
                                 : SliceLengths{}[I1] - src_scalar_per_access[I1] - iter1.value;

                // do work
                // hardcoding for buffer_load
                // TODO refactor transfer_data() to encapsulate this
                static_assert(SrcAddressSpace == AddressSpace::Global,
                              "wrong! hardcoded to use buffer_load, src must be global mem");

                using SrcVectorType = typename vector_type<SrcData, SrcScalarPerVector>::MemoryType;

                vector_type<SrcData, SrcScalarPerVector> src_vector;

#if 1
                src_vector.Vector() = amd_buffer_load<SrcData, SrcScalarPerVector>(
                    p_src, src_slice_origin_.GetOffset(), true, src_desc.GetElementSpaceSize());

                const bool is_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                    src_desc, src_slice_origin_);

                src_vector.Vector() = is_valid ? src_vector.Vector() : SrcVectorType{0};

                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t buffer_offset = buffer_desc_.CalculateOffset(
                        make_multi_index(i0, i1) + i * src_scalar_step_in_vector);

                    buffer_(Number<buffer_offset>{}) = src_vector[i];
                });
#elif 1
                const bool is_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                    src_desc, src_slice_origin_);

                src_vector.Vector() = amd_buffer_load<SrcData, SrcScalarPerVector>(
                    p_src, src_slice_origin_.GetOffset(), is_valid, src_desc.GetElementSpaceSize());

                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t buffer_offset = buffer_desc_.CalculateOffset(
                        make_multi_index(i0, i1) + i * src_scalar_step_in_vector);

                    buffer_(Number<buffer_offset>{}) = src_vector[i];
                });
#elif 0
                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t buffer_offset = buffer_desc_.CalculateOffset(
                        make_multi_index(i0, i1) + i * src_scalar_step_in_vector);

                    int32x2_t is_valid_i32 = is_valid;

                    asm volatile("\n \
						 v_cmp_gt_u32_e64 is_valid_flag, is_valid_i32, 0 \n \  
						 v_cndmask_b32_e64 src_data, 0, src_data, is_valid_flag \n \
						"
                                 : "=s"(is_valid_flag), "=v"(src_data),
                                 : "v"(is_valid_i32), "2"(is_valid_flag), "3"(src_data));

                    buffer_(Number<buffer_offset>{}) = src_data;
                });
#endif

                // move dim1 iterator
                if constexpr(iter1.value < access_lengths[I1] - 1)
                {
                    if constexpr(forward_dim1)
                    {
                        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_0_p);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_0_m);
                    }
                }
            });

            // move dim0 iterator
            if constexpr(iter0.value < access_lengths[I0] - 1)
            {
                move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_step_p_0);
            }
        });

        // move src coordinate back to its slice origin
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_back_step =
                make_dynamic_tensor_coordinate_step(src_desc, GetSrcCoordinateBackStep());

            move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_back_step);
        }
    }

    __device__ static constexpr auto GetSrcCoordinateBackStep()
    {
        constexpr auto src_scalar_per_access = [&]() {
            Index src_scalar_per_access;

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    src_scalar_per_access(i) = SrcScalarPerVector;
                }
                else
                {
                    src_scalar_per_access(i) = 1;
                }
            });

            return src_scalar_per_access;
        }();

        MultiIndex<nDim> src_back_step;

        src_back_step(Number<0>{}) = src_scalar_per_access[Number<0>{}] - SliceLengths{}[0];

        static_for<1, nDim, 1>{}([&](auto i) {
            constexpr auto i_m1 = i - Number<1>{};

            src_back_step(i) = (SliceLengths{}[i_m1] % (2 * src_scalar_per_access[i_m1]) == 0)
                                   ? 0
                                   : (src_scalar_per_access[i] - SliceLengths{}[i]);
        });

        return src_back_step;
    }

    __device__ static constexpr auto GetDstCoordinateBackStep()
    {
        constexpr auto dst_scalar_per_access = [&]() {
            Index dst_scalar_per_access;

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(i == DstVectorDim)
                {
                    dst_scalar_per_access(i) = DstScalarPerVector;
                }
                else
                {
                    dst_scalar_per_access(i) = 1;
                }
            });

            return dst_scalar_per_access;
        }();

        MultiIndex<nDim> dst_back_step;

        dst_back_step(Number<0>{}) = dst_scalar_per_access[Number<0>{}] - SliceLengths{}[0];

        static_for<1, nDim, 1>{}([&](auto i) {
            constexpr auto i_m1 = i - Number<1>{};

            dst_back_step(i) = (SliceLengths{}[i_m1] % (2 * dst_scalar_per_access[i_m1]) == 0)
                                   ? 0
                                   : (dst_scalar_per_access[i] - SliceLengths{}[i]);
        });

        return dst_back_step;
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx = SrcResetCoordinateAfterRun
                                           ? src_slice_origin_step_idx
                                           : src_slice_origin_step_idx + GetSrcCoordinateBackStep();

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
                                           : dst_slice_origin_step_idx + GetDstCoordinateBackStep();

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
                                           : src_slice_origin_step_idx + GetSrcCoordinateBackStep();

        // is it OK to construct a new step every time?
#if 0 // hack
        const auto adjusted_step = make_dynamic_tensor_coordinate_step(
            src_desc, adjusted_step_idx);
#elif 1
        // for padded input tensor
        const auto adjusted_step = make_dynamic_tensor_coordinate_step_hack(
            src_desc, adjusted_step_idx, Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2>{});
#elif 0
        // for non-padded input tensor
        const auto adjusted_step = make_dynamic_tensor_coordinate_step_hack(
            src_desc, adjusted_step_idx, Sequence<0, 0, 0, 0, 0, 1, 2>{});
#elif 1
        // for 1x1, input tensor
        const auto adjusted_step = make_dynamic_tensor_coordinate_step_hack(
            src_desc, adjusted_step_idx, Sequence<0, 1, 2>{});
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
