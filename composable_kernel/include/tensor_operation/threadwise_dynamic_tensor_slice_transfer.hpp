#ifndef CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP
#define CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

// TODO: How to fix this? It uses an struct instead of lambda because lambda
// doesn't have constructor
template <index_t DstVectorDim, index_t DstScalarPerVector>
struct lambda_ThreadwiseDynamicTensorSliceTransfer_v1r3_dst_scalar_per_access
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == DstVectorDim) ? DstScalarPerVector : 1;
    }
};

template <index_t DstVectorDim>
struct lambda_ThreadwiseDynamicTensorSliceTransfer_v1r3_dst_scalar_step_in_vector
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == DstVectorDim) ? 1 : 0;
    }
};

// this version is less likely to have scratch memory issue, due to:
//   1. It does not keep reference to tensor descriptor
//   2. It does not construct new tensor coordinate for this->Run()
// Assume src_slice_origin_idx is 0
// TODO: support non-zero src_slice_oring_idx
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

    using DstCoordIterator = decltype(make_dynamic_tensor_coordinate_iterator(DstDesc{}, Index{}));

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

    template <typename DstIteratorHacks>
    __device__ void Run(const SrcData* p_src,
                        const DstDesc& dst_desc,
                        DstData* p_dst,
                        const DstIteratorHacks& dst_iterator_hacks)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr auto dst_scalar_per_access = generate_sequence(
            lambda_ThreadwiseDynamicTensorSliceTransfer_v1r3_dst_scalar_per_access<
                DstVectorDim,
                DstScalarPerVector>{},
            Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector = generate_sequence(
            lambda_ThreadwiseDynamicTensorSliceTransfer_v1r3_dst_scalar_step_in_vector<
                DstVectorDim>{},
            Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        // make forward iterators
        const auto dst_forward_iterators = generate_tuple(
            [&](auto i) {
                Index forward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step(j) = (i.value == j.value) ? dst_scalar_per_access[i] : 0;
                });

                return make_dynamic_tensor_coordinate_iterator(
                    dst_desc, forward_step, dst_iterator_hacks[I0][i]);
            },
            Number<nDim>{});

        // make backward iterators
        const auto dst_backward_iterators = generate_tuple(
            [&](auto i) {
                Index backward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step(j) = (i.value == j.value) ? -dst_scalar_per_access[i] : 0;
                });

                return make_dynamic_tensor_coordinate_iterator(
                    dst_desc, backward_step, dst_iterator_hacks[I1][i]);
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_access_idx) {

            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep;

                forward_sweep(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_access_idx[I0];

                    static_for<0, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_access_lengths[j] + ordered_access_idx[j];
                    });

                    forward_sweep(i) = tmp % 2 == 0;
                });

                return forward_sweep;
            }();

            // calculate dst data index
            constexpr auto dst_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i]
                                         ? ordered_access_idx[i]
                                         : ordered_access_lengths[i] - 1 - ordered_access_idx[i];
                });

                auto dst_data_idx = container_reorder_given_old2new(ordered_idx, dim_access_order) *
                                    dst_scalar_per_access;

                return dst_data_idx;
            }();

            // copy data
            // hardcoding for buffer_store
            // TODO refactor transfer_data() to encapsulate this
            static_assert(SrcAddressSpace == AddressSpace::Vgpr &&
                              DstAddressSpace == AddressSpace::Global,
                          "wrong! hardcoded to use buffer_store");

            vector_type<DstData, DstScalarPerVector> dst_vector;

            // this is hardcoded for src that has compile-time tensor descriptor
            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                // assume src_slice_origin_idx is 0
                // TODO: support non-zero src_slice_oring_idx
                constexpr index_t src_offset =
                    SrcDesc::CalculateOffset(dst_data_idx + i * dst_scalar_step_in_vector);

                dst_vector(i) = p_src[Number<src_offset>{}];
            });

#if 1
            amd_buffer_store_v2<DstData, DstScalarPerVector>(
                dst_vector.Vector(),
                p_dst,
                dst_slice_origin_coord_.GetOffset(),
                coordinate_has_valid_offset_assuming_visible_index_is_valid(
                    dst_desc, dst_slice_origin_coord_),
                dst_desc.GetElementSpaceSize());
#else
            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                amd_buffer_store_v2<DstData, 1>(
                    dst_vector[i],
                    p_dst,
                    dst_slice_origin_coord_.GetOffset() + i.value,
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(
                        dst_desc, dst_slice_origin_coord_),
                    dst_desc.GetElementSpaceSize());
            });
#endif

            constexpr auto move_on_dim = [&]() constexpr
            {
                StaticallyIndexedArray<bool, nDim> move_on_dim;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim(i) = ordered_access_idx[i] < ordered_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim(i) &= ordered_access_idx[j] == ordered_access_lengths[j] - 1;
                    });
                });

                return move_on_dim;
            }
            ();

            // move
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_dynamic_tensor_coordinate(dst_desc,
                                                       dst_slice_origin_coord_,
                                                       dst_forward_iterators[dim_access_order[i]]);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(dst_desc,
                                                       dst_slice_origin_coord_,
                                                       dst_backward_iterators[dim_access_order[i]]);
                    }
                }
            });
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_iterator =
                make_dynamic_tensor_coordinate_iterator(dst_desc, GetDstCoordinateResetStep());

            move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_coord_, dst_reset_iterator);
        }
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto dst_scalar_per_access = [&]() {
            Index dst_scalar_per_access;

            static_for<0, nDim, 1>{}([&](auto i) {
                dst_scalar_per_access(i) = (i == DstVectorDim) ? DstScalarPerVector : 1;
            });

            return dst_scalar_per_access;
        }();

        MultiIndex<nDim> dst_reset_iterator;

        // TODO: this is wrong, need to consider DimAccessOrder
        dst_reset_iterator(Number<0>{}) = dst_scalar_per_access[Number<0>{}] - SliceLengths{}[0];

        static_for<1, nDim, 1>{}([&](auto i) {
            constexpr auto i_m1 = i - Number<1>{};

            // TODO: this is wrong
            dst_reset_iterator(i) = (SliceLengths{}[i_m1] % (2 * dst_scalar_per_access[i_m1]) == 0)
                                        ? 0
                                        : (dst_scalar_per_access[i] - SliceLengths{}[i]);
        });

        return dst_reset_iterator;
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by RunWrite(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step =
            make_dynamic_tensor_coordinate_iterator(dst_desc, adjusted_step_idx);

        move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_coord_, adjusted_step);
    }

    private:
    DstCoord dst_slice_origin_coord_;
}; // namespace ck

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

    using SrcCoordIterator = decltype(make_dynamic_tensor_coordinate_iterator(SrcDesc{}, Index{}));
    using DstCoordIterator = decltype(make_dynamic_tensor_coordinate_iterator(DstDesc{}, Index{}));

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

    template <typename SrcIteratorHacks>
    __device__ void RunRead(const SrcDesc& src_desc,
                            const SrcData* p_src,
                            const SrcIteratorHacks& src_iterator_hacks)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // hardcoded for 2D
        // TODO implemente N-D
        static_assert(remove_reference_t<SrcDesc>::GetNumOfDimension() == 2,
                      "wrong! hardcoded for 2D tensor");

        constexpr auto src_scalar_per_access = [&]() {
            Index src_scalar_per_access;

            static_for<0, nDim, 1>{}([&](auto i) {
                src_scalar_per_access(i) = (i == SrcVectorDim) ? SrcScalarPerVector : 1;
            });

            return src_scalar_per_access;
        }();

        constexpr auto src_scalar_step_in_vector = [&]() {
            Index src_scalar_step_in_vector;

            static_for<0, nDim, 1>{}(
                [&](auto i) { src_scalar_step_in_vector(i) = (i == SrcVectorDim) ? 1 : 0; });

            return src_scalar_step_in_vector;
        }();

        constexpr auto access_lengths = [&]() {
            Index access_lengths;

            static_for<0, nDim, 1>{}(
                [&](auto i) { access_lengths(i) = SliceLengths{}[i] / src_scalar_per_access[i]; });

            return access_lengths;
        }();

        const auto src_forward_iterators = make_tuple(
            make_dynamic_tensor_coordinate_iterator(src_desc,
                                                    make_multi_index(1, 0) * src_scalar_per_access,
                                                    src_iterator_hacks[I0][I0]),
            make_dynamic_tensor_coordinate_iterator(src_desc,
                                                    make_multi_index(0, 1) * src_scalar_per_access,
                                                    src_iterator_hacks[I0][I1]));

        const auto src_backward_iterators = make_tuple(
            make_dynamic_tensor_coordinate_iterator(src_desc,
                                                    make_multi_index(-1, 0) * src_scalar_per_access,
                                                    src_iterator_hacks[I1][I0]),
            make_dynamic_tensor_coordinate_iterator(src_desc,
                                                    make_multi_index(0, -1) * src_scalar_per_access,
                                                    src_iterator_hacks[I1][I1]));

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
#else
                const bool is_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                    src_desc, src_slice_origin_);

                src_vector.Vector() = amd_buffer_load<SrcData, SrcScalarPerVector>(
                    p_src, src_slice_origin_.GetOffset(), is_valid, src_desc.GetElementSpaceSize());

                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t buffer_offset = buffer_desc_.CalculateOffset(
                        make_multi_index(i0, i1) + i * src_scalar_step_in_vector);

                    buffer_(Number<buffer_offset>{}) = src_vector[i];
                });
#endif

                // move dim1 iterator
                if constexpr(iter1.value < access_lengths[I1] - 1)
                {
                    if constexpr(forward_dim1)
                    {
                        move_dynamic_tensor_coordinate(
                            src_desc, src_slice_origin_, src_forward_iterators[I1]);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(
                            src_desc, src_slice_origin_, src_backward_iterators[I1]);
                    }
                }
            });

            // move dim0 iterator
            if constexpr(iter0.value < access_lengths[I0] - 1)
            {
                move_dynamic_tensor_coordinate(
                    src_desc, src_slice_origin_, src_forward_iterators[I0]);
            }
        });

        // move src coordinate back to its slice origin
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_iterator =
                make_dynamic_tensor_coordinate_iterator(src_desc, GetSrcCoordinateResetStep());

            move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, src_reset_iterator);
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
                make_dynamic_tensor_coordinate_iterator(dst_desc, make_multi_index(0, 1));
            const auto dst_step_0_m =
                make_dynamic_tensor_coordinate_iterator(dst_desc, make_multi_index(0, -1));

            const auto dst_step_p_0 =
                make_dynamic_tensor_coordinate_iterator(dst_desc, make_multi_index(1, 0));
            const auto dst_step_m_0 =
                make_dynamic_tensor_coordinate_iterator(dst_desc, make_multi_index(-1, 0));

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
            const auto dst_reset_iterator =
                make_dynamic_tensor_coordinate_iterator(dst_desc, GetDstCoordinateResetStep());

            move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, dst_reset_iterator);
        }
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
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

        MultiIndex<nDim> src_reset_iterator;

        src_reset_iterator(Number<0>{}) = src_scalar_per_access[Number<0>{}] - SliceLengths{}[0];

        static_for<1, nDim, 1>{}([&](auto i) {
            constexpr auto i_m1 = i - Number<1>{};

            src_reset_iterator(i) = (SliceLengths{}[i_m1] % (2 * src_scalar_per_access[i_m1]) == 0)
                                        ? 0
                                        : (src_scalar_per_access[i] - SliceLengths{}[i]);
        });

        return src_reset_iterator;
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
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

        MultiIndex<nDim> dst_reset_iterator;

        dst_reset_iterator(Number<0>{}) = dst_scalar_per_access[Number<0>{}] - SliceLengths{}[0];

        static_for<1, nDim, 1>{}([&](auto i) {
            constexpr auto i_m1 = i - Number<1>{};

            dst_reset_iterator(i) = (SliceLengths{}[i_m1] % (2 * dst_scalar_per_access[i_m1]) == 0)
                                        ? 0
                                        : (dst_scalar_per_access[i] - SliceLengths{}[i]);
        });

        return dst_reset_iterator;
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step =
            make_dynamic_tensor_coordinate_iterator(src_desc, adjusted_step_idx);

        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, adjusted_step);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    template <typename SrcMoveSliceWindowIteratorHack>
    __device__ void
    MoveSrcSliceWindow(const SrcDesc& src_desc,
                       const Index& src_slice_origin_step_idx,
                       const SrcMoveSliceWindowIteratorHack& src_move_slice_window_iterator_hack)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_dynamic_tensor_coordinate_iterator(
            src_desc, adjusted_step_idx, src_move_slice_window_iterator_hack);

        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_, adjusted_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by RunWrite(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step =
            make_dynamic_tensor_coordinate_iterator(dst_desc, adjusted_step_idx);

        move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_, adjusted_step);
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
