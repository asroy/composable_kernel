#ifndef CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP
#define CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

// TODO: How to fix this? It uses an struct instead of lambda because lambda
// doesn't have constructor
template <index_t VectorDim, index_t ScalarPerVector>
struct lambda_scalar_per_access
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? ScalarPerVector : 1;
    }
};

template <index_t VectorDim>
struct lambda_scalar_step_in_vector
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? 1 : 0;
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

        // TODO: don't use this
        constexpr auto dst_scalar_per_access = generate_sequence(
            lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

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

    __device__ void Run(const SrcData* p_src, const DstDesc& dst_desc, DstData* p_dst)
    {
        constexpr index_t ntransform_dst = DstDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_dst, 0>::type{};

        constexpr auto dst_iterator_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        Run(p_src, dst_desc, p_dst, dst_iterator_hacks);
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
        : src_slice_origin_coord_(make_dynamic_tensor_coordinate(src_desc, src_slice_origin)),
          dst_slice_origin_coord_(make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin))
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
        src_slice_origin_coord_ = make_dynamic_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_slice_origin_coord_ = make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcIteratorHacks>
    __device__ void RunRead(const SrcDesc& src_desc,
                            const SrcData* p_src,
                            const SrcIteratorHacks& src_iterator_hacks)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // TODO: don't use this
        constexpr auto src_scalar_per_access = generate_sequence(
            lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_scalar_step_in_vector =
            generate_sequence(lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, src_dim_access_order);

        // make forward iterators
        const auto src_forward_iterators = generate_tuple(
            [&](auto i) {
                Index forward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step(j) = (i.value == j.value) ? src_scalar_per_access[i] : 0;
                });

                return make_dynamic_tensor_coordinate_iterator(
                    src_desc, forward_step, src_iterator_hacks[I0][i]);
            },
            Number<nDim>{});

        // make backward iterators
        const auto src_backward_iterators = generate_tuple(
            [&](auto i) {
                Index backward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step(j) = (i.value == j.value) ? -src_scalar_per_access[i] : 0;
                });

                return make_dynamic_tensor_coordinate_iterator(
                    src_desc, backward_step, src_iterator_hacks[I1][i]);
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

            // calculate src data index
            constexpr auto data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i]
                                         ? ordered_access_idx[i]
                                         : ordered_access_lengths[i] - 1 - ordered_access_idx[i];
                });

                auto data_idx = container_reorder_given_old2new(ordered_idx, src_dim_access_order) *
                                src_scalar_per_access;

                return data_idx;
            }();

            // copy data
            // hardcoding for buffer_load
            // TODO refactor transfer_data() to encapsulate this
            static_assert(SrcAddressSpace == AddressSpace::Global,
                          "wrong! hardcoded to use buffer_load, src must be global mem");

            vector_type<SrcData, SrcScalarPerVector> src_vector;

            using SrcVectorType = typename vector_type<SrcData, SrcScalarPerVector>::MemoryType;

#if 1
            src_vector.Vector() = amd_buffer_load<SrcData, SrcScalarPerVector>(
                p_src, src_slice_origin_coord_.GetOffset(), true, src_desc.GetElementSpaceSize());

            const bool is_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                src_desc, src_slice_origin_coord_);

            src_vector.Vector() = is_valid ? src_vector.Vector() : SrcVectorType{0};

            static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t buffer_offset =
                    buffer_desc_.CalculateOffset(data_idx + i * src_scalar_step_in_vector);

                buffer_(Number<buffer_offset>{}) = src_vector[i];
            });
#else
            const bool is_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                src_desc, src_slice_origin_coord_);

            src_vector.Vector() =
                amd_buffer_load<SrcData, SrcScalarPerVector>(p_src,
                                                             src_slice_origin_coord_.GetOffset(),
                                                             is_valid,
                                                             src_desc.GetElementSpaceSize());

            static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t buffer_offset =
                    buffer_desc_.CalculateOffset(data_idx + i * src_scalar_step_in_vector);

                buffer_(Number<buffer_offset>{}) = src_vector[i];
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
                        move_dynamic_tensor_coordinate(
                            src_desc,
                            src_slice_origin_coord_,
                            src_forward_iterators[src_dim_access_order[i]]);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(
                            src_desc,
                            src_slice_origin_coord_,
                            src_backward_iterators[src_dim_access_order[i]]);
                    }
                }
            });
        });

        // move src coordinate back to slice origin (or not)
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_iterator =
                make_dynamic_tensor_coordinate_iterator(src_desc, GetSrcCoordinateResetStep());

            move_dynamic_tensor_coordinate(src_desc, src_slice_origin_coord_, src_reset_iterator);
        }
    }

    template <typename DstIteratorHacks>
    __device__ void
    RunWrite(const DstDesc& dst_desc, DstData* p_dst, const DstIteratorHacks& dst_iterator_hacks)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // TODO: don't use this
        constexpr auto dst_scalar_per_access = generate_sequence(
            lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dst_dim_access_order);

        // make forward iterators
        const auto dst_forward_iterators = generate_tuple(
            [&](auto i) {
                Index forward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step(j) = (i.value == j.value) ? dst_scalar_per_access[i] : 0;
                });

                const auto forward_iterator = make_dynamic_tensor_coordinate_iterator(
                    dst_desc, forward_step, dst_iterator_hacks[I0][i]);

                return forward_iterator;
            },
            Number<nDim>{});

        // make backward iterators
        const auto dst_backward_iterators = generate_tuple(
            [&](auto i) {
                Index backward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step(j) = (i.value == j.value) ? -dst_scalar_per_access[i] : 0;
                });

                const auto backward_iterator = make_dynamic_tensor_coordinate_iterator(
                    dst_desc, backward_step, dst_iterator_hacks[I1][i]);

                return backward_iterator;
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

                auto dst_data_idx =
                    container_reorder_given_old2new(ordered_idx, dst_dim_access_order) *
                    dst_scalar_per_access;

                return dst_data_idx;
            }();

            // copy data
            // hardcoding for ds_write
            // TODO refactor transfer_data() to encapsulate this
            static_assert(DstAddressSpace == AddressSpace::Lds &&
                              DstInMemOp == InMemoryDataOperation::Set,
                          "wrong! hardcoded for ds_write");

            vector_type<DstData, DstScalarPerVector> dst_vector;

            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t buffer_offset =
                    buffer_desc_.CalculateOffset(dst_data_idx + i * dst_scalar_step_in_vector);

                dst_vector(i) = buffer_[Number<buffer_offset>{}];
            });

            using DstVectorType = typename vector_type<DstData, DstScalarPerVector>::MemoryType;

            *reinterpret_cast<DstVectorType*>(p_dst + dst_slice_origin_coord_.GetOffset()) =
                dst_vector.Vector();

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
                        move_dynamic_tensor_coordinate(
                            dst_desc,
                            dst_slice_origin_coord_,
                            dst_forward_iterators[dst_dim_access_order[i]]);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(
                            dst_desc,
                            dst_slice_origin_coord_,
                            dst_backward_iterators[dst_dim_access_order[i]]);
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

    __device__ void RunRead(const SrcDesc& src_desc, const SrcData* p_src)
    {
        constexpr index_t ntransform_src = SrcDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_src, 0>::type{};

        constexpr auto src_iterator_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        RunRead(src_desc, p_src, src_iterator_hacks);
    }

    __device__ void RunWrite(const DstDesc& dst_desc, DstData* p_dst)
    {
        constexpr index_t ntransform_dst = DstDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_dst, 0>::type{};

        constexpr auto dst_iterator_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        RunWrite(dst_desc, p_dst, dst_iterator_hacks);
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

        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_coord_, adjusted_step);
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

        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_coord_, adjusted_step);
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
    static constexpr auto buffer_desc_ =
        make_dynamic_naive_tensor_descriptor_packed<nDim>(to_multi_index(SliceLengths{}));

    static constexpr index_t buffer_size_ = buffer_desc_.GetElementSpaceSize();

    StaticallyIndexedArray<SrcData, buffer_size_> buffer_;

    SrcCoord src_slice_origin_coord_;
    DstCoord dst_slice_origin_coord_;
};

} // namespace ck
#endif
