#ifndef CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_coordinate.hpp"

namespace ck {

// This threadwise copy allow vector access of src and dst.
// It allows the vector size to be different on src and dst.
// The dimensions of vector access should be the same on src and dst.
// The dimension access order should be the same on src and dst.
// Will do valid mapping check on src data: Read 0 if src data has a invalid mapping
// Will do valid mapping check on dst data: No write if dst data has a invalid mapping
template <typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SrcDstDimAccessOrder,
          index_t SrcDstVectorReadWriteDim,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite,
          AddressSpace SrcAddressSpace     = AddressSpace::generic,
          AddressSpace DstAddressSpace     = AddressSpace::generic,
          InMemoryDataOperation DstInMemOp = InMemoryDataOperation::none>
struct ThreadwiseGenericTensorSliceCopy_v4r2
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = typename TensorCoordinate<SrcDesc>::type;
    using DstCoord = typename TensorCoordinate<DstDesc>::type;

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v4r2(const Index& src_slice_origin,
                                                               const Index& dst_slice_origin)
        : mSrcSliceOrigin(src_slice_origin), mDstSliceOrigin(dst_slice_origin)
    {
        static_assert(nDim == SrcDesc::GetNumOfDimension() &&
                          nDim == DstDesc::GetNumOfDimension() && nDim == SliceLengths::Size() &&
                          nDim == SrcDstDimAccessOrder::Size(),
                      "wrong! # of dimensions not the same");

        static_assert(is_valid_sequence_map<SrcDstDimAccessOrder>{}, "wrong! map is not valid");

        static_assert(SliceLengths{}[SrcDstVectorReadWriteDim] %
                              math::lcm(SrcDataPerRead, DstDataPerWrite) ==
                          0,
                      "wrong! cannot evenly divide");

        // TODO:: sanity-check if vectorized memory read/write is allowed on src and dst
    }

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v4r2()
        : ThreadwiseGenericTensorSliceCopy_v4r2(make_zero_array<index_t, nDim>(),
                                                make_zero_array<index_t, nDim>())
    {
    }

    __device__ void SetSrcSliceOrigin(SrcCoord src_slice_origin)
    {
        mSrcSliceOrigin = src_slice_origin;
    }

    __device__ void SetDstSliceOrigin(DstCoord dst_slice_origin)
    {
        mDstSliceOrigin = dst_slice_origin;
    }

    template <typename SrcData, typename DstData>
    __device__ void Run(const SrcData* p_src, DstData* p_dst) const
    {
        constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerRead>{};
        constexpr auto dst_data_per_access = Number<DstDataPerWrite>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerRead, DstDataPerWrite)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        ford<decltype(long_vector_access_lengths), SrcDstDimAccessOrder>{}([&](
            auto long_vector_access_id) {

            // data id w.r.t slicing-window
            auto long_vector_data_begin_id = long_vector_access_id;
            long_vector_data_begin_id(vector_access_dim) =
                long_vector_size * long_vector_access_id[vector_access_dim];

            // buffer to hold a src long-vector
            SrcData p_src_long_vector[long_vector_size];

            // zero out buffer
            for(index_t i = 0; i < long_vector_size; ++i)
            {
                p_src_long_vector[i] = 0;
            }

            // load data from src to the long-vector buffer
            for(index_t i = 0; i < long_vector_size / src_data_per_access; ++i)
            {
                auto scalar_id               = make_zero_array<index_t, nDim>();
                scalar_id(vector_access_dim) = i * src_data_per_access;

                const index_t buffer_offset = i * src_data_per_access;

                const auto src_coord = mSrcSliceOrigin + (long_vector_data_begin_id + scalar_id);

                // Check src data's valid mapping situation, only check the first data in this src
                //   vector. It's user's responsiblity to make sure all data in the src vector
                //   has the valid/invalid mapping situation
                if(src_coord.IsOffsetValidAssumingUpperIndexIsValid())
                {
                    transfer_data<SrcData,
                                  SrcDataPerRead,
                                  SrcAddressSpace,
                                  AddressSpace::vgpr,
                                  InMemoryDataOperation::none>(
                        p_src, src_coord.GetOffset(), p_src_long_vector, buffer_offset);
                }
            }

            // SrcData to DstData conversion
            DstData p_dst_long_vector[long_vector_size];

            for(index_t i = 0; i < long_vector_size; ++i)
            {
                p_dst_long_vector[i] = type_convert<DstData>{}(p_src_long_vector[i]);
            }

            // store data from the long-vector buffer to dst
            for(index_t i = 0; i < long_vector_size / dst_data_per_access; ++i)
            {
                auto scalar_id               = make_zero_array<index_t, nDim>();
                scalar_id(vector_access_dim) = i * dst_data_per_access;

                const index_t buffer_offset = i * dst_data_per_access;

                const auto dst_coord = mDstSliceOrigin + (long_vector_data_begin_id + scalar_id);

                // Check dst data's valid mapping situation, only check the first data in this dst
                //   vector. It's user's responsiblity to make sure all data in the dst vector
                //   has the valid/invalid mapping situation
                if(dst_coord.IsOffsetValidAssumingUpperIndexIsValid())
                {
                    transfer_data<DstData,
                                  DstDataPerWrite,
                                  AddressSpace::vgpr,
                                  DstAddressSpace,
                                  DstInMemOp>(
                        p_dst_long_vector, buffer_offset, p_dst, dst_coord.GetOffset());
                }
            }
        });
    }

    // Modify Length to 1, if Mask is set to false
    // Used for isolating linear dimension from non-linear dimensions
    template <index_t... Lengths, index_t... Mask>
    __device__ static constexpr auto mask_lengths(Sequence<Lengths...>, Sequence<Mask...>)
    {
        return Sequence<(Mask ? Lengths : 1)...>{};
    }

    // Will do valid mapping check on src data: Read 0 if src data has a invalid mapping
    // Will do valid mapping check on dst data: No write if dst data has a invalid mapping
    // This version is optimized for address calculation of src tensor
    // TODO: this function is not compiled to expected ISA
    template <typename SrcData, typename DstData>
    __device__ void Run_optimized_src_address_calculation(const SrcData* p_src,
                                                          DstData* p_dst) const
    {
        constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerRead>{};
        constexpr auto dst_data_per_access = Number<DstDataPerWrite>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerRead, DstDataPerWrite)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        // separate linear dimensions from non-linear dimensions
        constexpr auto src_linear_dim_mask    = SrcDesc::GetLinearDimensionMask();
        constexpr auto src_nonlinear_dim_mask = SrcDesc::GetNonLinearDimensionMask();

        static_assert(
            src_linear_dim_mask.At(SrcDstVectorReadWriteDim) || long_vector_size == SrcDataPerRead,
            "Warning! SrcDstVectorReadWriteDim is not SrcDesc's linear dimension, performance "
            "would drop");

        // separate steps into linear and non-linear components, accoording to src tensor
        constexpr auto linear_long_vector_access_lengths =
            mask_lengths(long_vector_access_lengths, src_linear_dim_mask);

        constexpr auto nonlinear_long_vector_access_lengths =
            mask_lengths(long_vector_access_lengths, src_nonlinear_dim_mask);

        // loop over src's non-linear dimensions
        ford<decltype(nonlinear_long_vector_access_lengths)>{}([&](
            auto nonlinear_dim_long_vector_access_id) {

            // calculate step-sizes along src's nonlinear dimensions
            auto nonlinear_dim_data_steps = nonlinear_dim_long_vector_access_id;
            nonlinear_dim_data_steps(vector_access_dim) =
                long_vector_size * nonlinear_dim_long_vector_access_id[vector_access_dim];

            // move src cooridnate along nonlinear dimensions
            // this coordinate contains run-time per-thread offset
            const auto src_nonlinear_coord = mSrcSliceOrigin + nonlinear_dim_data_steps;

            // loop over src's linear dimensions
            ford<decltype(linear_long_vector_access_lengths)>{}([&](
                auto linear_dim_long_vector_access_id) {

                // step-sizes along src's linear dimensions
                auto linear_dim_data_steps = linear_dim_long_vector_access_id;
                linear_dim_data_steps(vector_access_dim) =
                    long_vector_size * linear_dim_long_vector_access_id[vector_access_dim];

                // buffer to hold a long-vector
                SrcData p_src_long_vector[long_vector_size];

                // zero out buffer
                for(index_t i = 0; i < long_vector_size; ++i)
                {
                    p_src_long_vector[i] = 0;
                }

                // Loop over SrcDstVectorReadWriteDim, and load data from src to the
                //   long-vector buffer.
                // If SrcDstVectorReadWriteDim is src's linear dimension, then src's
                //   offset-diff due to this looping is known at compile-time. If
                //   SrcDstVectorReadWriteDim is src's nonlinear dimension, then src's
                //   offset-diff due to this looping is only known at run-time. For best
                //   performance, SrcDstVectorReadWriteDim, should be src's linear dimension
                for(index_t i = 0; i < long_vector_size / src_data_per_access; ++i)
                {
                    auto scalar_id               = make_zero_array<index_t, nDim>();
                    scalar_id(vector_access_dim) = i * src_data_per_access;

                    const index_t buffer_offset = i * src_data_per_access;

                    // move src cooridnate along linear dimensions
                    const auto src_coord =
                        src_nonlinear_coord + (linear_dim_data_steps + scalar_id);

#if CK_EXPERIMENTAL_TENSOR_COORDINATE_USE_CALCULATE_OFFSET_DIFF // tweaking
                    // this is src compile-time offset
                    const index_t src_linear_offset =
                        src_nonlinear_coord.CalculateOffsetDiff(linear_dim_data_steps + scalar_id);
#else
                    // this is src compile-time offset
                    const index_t src_linear_offset =
                        src_coord.GetOffset() - src_nonlinear_coord.GetOffset();
#endif

                    // Check src data's valid mapping situation, only check the first data in this
                    // src
                    //   vector. It's user's responsiblity to make sure all data in the src vector
                    //   has the valid/invalid mapping situation
                    if(src_coord.IsOffsetValidAssumingUpperIndexIsValid())
                    {
                        transfer_data<SrcData,
                                      SrcDataPerRead,
                                      SrcAddressSpace,
                                      AddressSpace::vgpr,
                                      InMemoryDataOperation::none>(p_src,
                                                                   src_nonlinear_coord.GetOffset() +
                                                                       src_linear_offset,
                                                                   p_src_long_vector,
                                                                   buffer_offset);
                    }
                }

                // SrcData to DstData conversion
                DstData p_dst_long_vector[long_vector_size];

                for(index_t i = 0; i < long_vector_size; ++i)
                {
                    p_dst_long_vector[i] = type_convert<DstData>{}(p_src_long_vector[i]);
                }

                // store data from the long-vector buffer to dst
                for(index_t i = 0; i < long_vector_size / dst_data_per_access; ++i)
                {
                    auto scalar_id               = make_zero_array<index_t, nDim>();
                    scalar_id(vector_access_dim) = i * dst_data_per_access;

                    const index_t buffer_offset = i * dst_data_per_access;

                    // dst offset is calculated here, without explicitly separating into
                    //   compile-time and per-thread component
                    const auto dst_coord = mDstSliceOrigin + (nonlinear_dim_data_steps +
                                                              linear_dim_data_steps + scalar_id);

                    // Check dst data's valid mapping situation, only check the first data in this
                    // dst
                    //   vector. It's user's responsiblity to make sure all data in the dst vector
                    //   has the valid/invalid mapping situation
                    if(dst_coord.IsOffsetValidAssumingUpperIndexIsValid())
                    {
                        transfer_data<DstData,
                                      DstDataPerWrite,
                                      AddressSpace::vgpr,
                                      DstAddressSpace,
                                      DstInMemOp>(
                            p_dst_long_vector, buffer_offset, p_dst, dst_coord.GetOffset());
                    }
                }
            });
        });
    }

    // This version is optimized for address calculation of dst tensor
    // TODO: this function is not compiled to expected ISA
    template <typename SrcData, typename DstData>
    __device__ void Run_optimized_dst_address_calculation(const SrcData* p_src,
                                                          DstData* p_dst) const
    {
        constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerRead>{};
        constexpr auto dst_data_per_access = Number<DstDataPerWrite>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerRead, DstDataPerWrite)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        // separate linear dimensions from non-linear dimensions
        constexpr auto dst_linear_dim_mask    = DstDesc::GetLinearDimensionMask();
        constexpr auto dst_nonlinear_dim_mask = DstDesc::GetNonLinearDimensionMask();

        static_assert(
            dst_linear_dim_mask.At(SrcDstVectorReadWriteDim) || long_vector_size == DstDataPerWrite,
            "Warning! SrcDstVectorReadWriteDim is not DstDesc's linear dimension, performance "
            "would drop");

        // separate steps into linear and non-linear components, accoording to dst tensor
        constexpr auto linear_long_vector_access_lengths =
            mask_lengths(long_vector_access_lengths, dst_linear_dim_mask);

        constexpr auto nonlinear_long_vector_access_lengths =
            mask_lengths(long_vector_access_lengths, dst_nonlinear_dim_mask);

        // loop over dst's non-linear dimensions
        ford<decltype(nonlinear_long_vector_access_lengths)>{}([&](
            auto nonlinear_dim_long_vector_access_id) {

            // calculate step-sizes along dst's nonlinear dimensions
            auto nonlinear_dim_data_steps = nonlinear_dim_long_vector_access_id;
            nonlinear_dim_data_steps(vector_access_dim) =
                long_vector_size * nonlinear_dim_long_vector_access_id[vector_access_dim];

            // move dst cooridnate along nonlinear dimensions
            // this coordinate contains run-time per-thread offset
            const auto dst_nonlinear_coord = mDstSliceOrigin + nonlinear_dim_data_steps;

            // loop over dst's linear dimensions
            ford<decltype(linear_long_vector_access_lengths)>{}([&](
                auto linear_dim_long_vector_access_id) {

                // step-sizes along dst's linear dimensions
                auto linear_dim_data_steps = linear_dim_long_vector_access_id;
                linear_dim_data_steps(vector_access_dim) =
                    long_vector_size * linear_dim_long_vector_access_id[vector_access_dim];

                // buffer to hold a long-vector
                SrcData p_src_long_vector[long_vector_size];

                // zero out buffer
                for(index_t i = 0; i < long_vector_size; ++i)
                {
                    p_src_long_vector[i] = 0;
                }

                // Loop over SrcDstVectorReadWriteDim, and load data from src to the
                //   long-vector buffer.
                // If SrcDstVectorReadWriteDim is dst's linear dimension, then dst's
                //   offset-diff due to this looping is known at compile-time. If
                //   SrcDstVectorReadWriteDim is dst's nonlinear dimension, then dst's
                //   offset-diff due to this looping is only known at run-time. For best
                //   performance, SrcDstVectorReadWriteDim, should be dst's linear dimension
                for(index_t i = 0; i < long_vector_size / src_data_per_access; ++i)
                {
                    auto scalar_id               = make_zero_array<index_t, nDim>();
                    scalar_id(vector_access_dim) = i * src_data_per_access;

                    const index_t buffer_offset = i * src_data_per_access;

                    // src offset is calculated here, without explicitly separating into
                    //   compile-time and per-thread component
                    const auto src_coord = mSrcSliceOrigin + (nonlinear_dim_data_steps +
                                                              linear_dim_data_steps + scalar_id);

                    // Check src data's valid mapping situation, only check the first data in this
                    // src
                    //   vector. It's user's responsiblity to make sure all data in the src vector
                    //   has the valid/invalid mapping situation
                    if(src_coord.IsOffsetValidAssumingUpperIndexIsValid())
                    {
                        transfer_data<SrcData,
                                      SrcDataPerRead,
                                      SrcAddressSpace,
                                      AddressSpace::vgpr,
                                      InMemoryDataOperation::none>(
                            p_src, src_coord.GetOffset(), p_src_long_vector, buffer_offset);
                    }
                }

                // SrcData to DstData conversion
                DstData p_dst_long_vector[long_vector_size];

                for(index_t i = 0; i < long_vector_size; ++i)
                {
                    p_dst_long_vector[i] = type_convert<DstData>{}(p_src_long_vector[i]);
                }

                // store data from the long-vector buffer to dst
                for(index_t i = 0; i < long_vector_size / dst_data_per_access; ++i)
                {
                    auto scalar_id               = make_zero_array<index_t, nDim>();
                    scalar_id(vector_access_dim) = i * dst_data_per_access;

                    const index_t buffer_offset = i * dst_data_per_access;

                    // move dst cooridnate along linear dimensions
                    const auto dst_coord =
                        dst_nonlinear_coord + (linear_dim_data_steps + scalar_id);

#if CK_EXPERIMENTAL_TENSOR_COORDINATE_USE_CALCULATE_OFFSET_DIFF // tweaking
                    // this is dst compile-time offset
                    const index_t dst_linear_offset =
                        dst_nonlinear_coord.CalculateOffsetDiff(linear_dim_data_steps + scalar_id);
#else
                    // this is dst compile-time offset
                    const index_t dst_linear_offset =
                        dst_coord.GetOffset() - dst_nonlinear_coord.GetOffset();
#endif

                    // Check dst data's valid mapping situation, only check the first data in this
                    // dst
                    //   vector. It's user's responsiblity to make sure all data in the dst vector
                    //   has the valid/invalid mapping situation
                    if(dst_coord.IsOffsetValidAssumingUpperIndexIsValid())
                    {
                        transfer_data<DstData,
                                      DstDataPerWrite,
                                      AddressSpace::vgpr,
                                      DstAddressSpace,
                                      DstInMemOp>(p_dst_long_vector,
                                                  buffer_offset,
                                                  p_dst,
                                                  dst_nonlinear_coord.GetOffset() +
                                                      dst_linear_offset);
                    }
                }
            });
        });
    }

    __device__ static constexpr bool HasWorkingOptimizedAddressCalculation()
    {
#if CK_EXPERIMENTAL_THREADWISE_COPY_V4R2_USE_OPTIMIZED_ADDRESS_CACLULATION // tweaking
        return true;
#else
        return false;
#endif
    }

    template <typename T, bool PositiveDirection>
    __device__ void MoveSrcSliceWindow(const T& step_sizes_,
                                       integral_constant<bool, PositiveDirection>)
    {
        const auto step_sizes = to_array(step_sizes_);

        static_if<PositiveDirection>{}([&](auto) {
            mSrcSliceOrigin += to_array(step_sizes);
        }).Else([&](auto) { mSrcSliceOrigin -= step_sizes; });
    }

    template <typename T, bool PositiveDirection>
    __device__ void MoveDstSliceWindow(const T& step_sizes_,
                                       integral_constant<bool, PositiveDirection>)
    {
        const auto step_sizes = to_array(step_sizes_);

        static_if<PositiveDirection>{}([&](auto) {
            mDstSliceOrigin += step_sizes;
        }).Else([&](auto) { mDstSliceOrigin -= step_sizes; });
    }

    private:
    SrcCoord mSrcSliceOrigin;
    DstCoord mDstSliceOrigin;
};

template <typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SrcDstDimAccessOrder,
          index_t SrcDstVectorReadWriteDim,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite,
          AddressSpace SrcAddressSpace     = AddressSpace::generic,
          AddressSpace DstAddressSpace     = AddressSpace::generic,
          InMemoryDataOperation DstInMemOp = InMemoryDataOperation::none>
struct ThreadwiseGenericTensorSliceCopy_v4r2_src_runtime_offsets
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = typename TensorCoordinate<SrcDesc>::type;
    using DstCoord = typename TensorCoordinate<DstDesc>::type;

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v4r2_src_runtime_offsets(const Index& src_slice_origin,
                                                               const Index& dst_slice_origin,
                                                               const Index& src_runtime_strides_div,
                                                               const Index& src_runtime_strides_mul)
        : mSrcSliceOrigin(src_slice_origin), mDstSliceOrigin(dst_slice_origin),
        mSrcRuntimeStridesDiv(src_runtime_strides_div), mSrcRuntimeStridesMul(src_runtime_strides_mul),
        mSrcRuntimeOffsetsPerThread(0)
    {
        static_assert(nDim == SrcDesc::GetNumOfDimension() &&
                          nDim == DstDesc::GetNumOfDimension() && nDim == SliceLengths::Size() &&
                          nDim == SrcDstDimAccessOrder::Size(),
                      "wrong! # of dimensions not the same");

        static_assert(is_valid_sequence_map<SrcDstDimAccessOrder>{}, "wrong! map is not valid");

        static_assert(SliceLengths{}[SrcDstVectorReadWriteDim] %
                              math::lcm(SrcDataPerRead, DstDataPerWrite) ==
                          0,
                      "wrong! cannot evenly divide");

        // TODO:: sanity-check if vectorized memory read/write is allowed on src and dst

        // TODO: be careful of constructor time update
        // UpdateSrcSliceOriginWithRuntimeOffsets();
    }

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v4r2_src_runtime_offsets()
        : ThreadwiseGenericTensorSliceCopy_v4r2_src_runtime_offsets(make_zero_array<index_t, nDim>(),
                                                make_zero_array<index_t, nDim>(),
                                                sequence2array(typename uniform_sequence_gen<nDim, 1>::type{}),
                                                sequence2array(typename uniform_sequence_gen<nDim, 1>::type{}))
    {
    }

    __device__ void UpdateSrcSliceOriginWithRuntimeOffsets(){
        Index cur_upper_index = mSrcSliceOrigin.GetIndex();
        index_t cur_offset = 0;
        for(index_t i=0; i< nDim; i++){
            index_t p = cur_upper_index[i] / mSrcRuntimeStridesDiv[i];
            index_t q = cur_upper_index[i] % mSrcRuntimeStridesDiv[i];
            cur_upper_index.At(i) = q;
            cur_offset += p * mSrcRuntimeStridesMul[i];
        }
        mSrcSliceOrigin = SrcCoord{cur_upper_index};
        mSrcRuntimeOffsetsPerThread += cur_offset;
    }

    __device__ void SetSrcRuntimeStrides(Index src_runtime_strides_div, Index src_runtime_strides_mul)
    {
        mSrcRuntimeStridesDiv = src_runtime_strides_div;
        mSrcRuntimeStridesMul = src_runtime_strides_mul;
    }

    __device__ void SetSrcSliceOrigin(SrcCoord src_slice_origin)
    {
        mSrcSliceOrigin = src_slice_origin;
        UpdateSrcSliceOriginWithRuntimeOffsets();
    }

    __device__ void SetDstSliceOrigin(DstCoord dst_slice_origin)
    {
        mDstSliceOrigin = dst_slice_origin;
    }

    __device__ index_t CalculateSliceSrcRuntimeOffsets(Index & vector_id) const
    {
        index_t slice_offset = 0;
        for(index_t i=0; i< nDim; i++){
            index_t id = vector_id[i];
            slice_offset += (id / mSrcRuntimeStridesDiv[i]) * mSrcRuntimeStridesMul[i];
            vector_id.At(i) = id % mSrcRuntimeStridesDiv[i];
        }
        return slice_offset;
    }

    template <typename SrcData, typename DstData>
    __device__ void Run(const SrcData* p_src, DstData* p_dst) const
    {
        constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerRead>{};
        constexpr auto dst_data_per_access = Number<DstDataPerWrite>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerRead, DstDataPerWrite)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        ford<decltype(long_vector_access_lengths), SrcDstDimAccessOrder>{}([&](
            auto long_vector_access_id) {

            // data id w.r.t slicing-window
            auto long_vector_data_begin_id = long_vector_access_id;
            long_vector_data_begin_id(vector_access_dim) =
                long_vector_size * long_vector_access_id[vector_access_dim];

            // buffer to hold a src long-vector
            SrcData p_src_long_vector[long_vector_size];

            // zero out buffer
            for(index_t i = 0; i < long_vector_size; ++i)
            {
                p_src_long_vector[i] = 0;
            }

            // load data from src to the long-vector buffer
            for(index_t i = 0; i < long_vector_size / src_data_per_access; ++i)
            {
                auto scalar_id               = make_zero_array<index_t, nDim>();
                scalar_id(vector_access_dim) = i * src_data_per_access;

                const index_t buffer_offset = i * src_data_per_access;
                auto vector_id = long_vector_data_begin_id + scalar_id;
                index_t src_runtime_offset = CalculateSliceSrcRuntimeOffsets(vector_id);

                const auto src_coord = mSrcSliceOrigin + vector_id;

                // Check src data's valid mapping situation, only check the first data in this src
                //   vector. It's user's responsiblity to make sure all data in the src vector
                //   has the valid/invalid mapping situation
                if(src_coord.IsOffsetValidAssumingUpperIndexIsValid())
                {
                    transfer_data<SrcData,
                                  SrcDataPerRead,
                                  SrcAddressSpace,
                                  AddressSpace::vgpr,
                                  InMemoryDataOperation::none>(
                        p_src,
                        src_coord.GetOffset() + mSrcRuntimeOffsetsPerThread + src_runtime_offset,
                        p_src_long_vector, buffer_offset);
                }
            }

            // SrcData to DstData conversion
            DstData p_dst_long_vector[long_vector_size];

            for(index_t i = 0; i < long_vector_size; ++i)
            {
                p_dst_long_vector[i] = type_convert<DstData>{}(p_src_long_vector[i]);
            }

            // store data from the long-vector buffer to dst
            for(index_t i = 0; i < long_vector_size / dst_data_per_access; ++i)
            {
                auto scalar_id               = make_zero_array<index_t, nDim>();
                scalar_id(vector_access_dim) = i * dst_data_per_access;

                const index_t buffer_offset = i * dst_data_per_access;

                const auto dst_coord = mDstSliceOrigin + (long_vector_data_begin_id + scalar_id);

                // Check dst data's valid mapping situation, only check the first data in this dst
                //   vector. It's user's responsiblity to make sure all data in the dst vector
                //   has the valid/invalid mapping situation
                if(dst_coord.IsOffsetValidAssumingUpperIndexIsValid())
                {
                    transfer_data<DstData,
                                  DstDataPerWrite,
                                  AddressSpace::vgpr,
                                  DstAddressSpace,
                                  DstInMemOp>(
                        p_dst_long_vector, buffer_offset, p_dst, dst_coord.GetOffset());
                }
            }
        });
    }

    template <typename T, bool PositiveDirection>
    __device__ void MoveSrcSliceWindow(const T& step_sizes_,
                                       integral_constant<bool, PositiveDirection>)
    {
        const auto step_sizes = to_array(step_sizes_);

        static_if<PositiveDirection>{}([&](auto) {
            mSrcSliceOrigin += to_array(step_sizes);
        }).Else([&](auto) { mSrcSliceOrigin -= step_sizes; });
        UpdateSrcSliceOriginWithRuntimeOffsets();
    }

    template <typename T, bool PositiveDirection>
    __device__ void MoveDstSliceWindow(const T& step_sizes_,
                                       integral_constant<bool, PositiveDirection>)
    {
        const auto step_sizes = to_array(step_sizes_);

        static_if<PositiveDirection>{}([&](auto) {
            mDstSliceOrigin += step_sizes;
        }).Else([&](auto) { mDstSliceOrigin -= step_sizes; });
    }

    private:
    SrcCoord mSrcSliceOrigin;
    DstCoord mDstSliceOrigin;
    index_t mSrcRuntimeOffsetsPerThread;
    Index mSrcRuntimeStridesDiv;
    Index mSrcRuntimeStridesMul;
};

template <typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SrcDstDimAccessOrder,
          index_t SrcDstVectorReadWriteDim,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite,
          AddressSpace SrcAddressSpace     = AddressSpace::generic,
          AddressSpace DstAddressSpace     = AddressSpace::generic,
          InMemoryDataOperation DstInMemOp = InMemoryDataOperation::none>
struct ThreadwiseGenericTensorSliceCopy_v4r2_dst_runtime_offsets
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = typename TensorCoordinate<SrcDesc>::type;
    using DstCoord = typename TensorCoordinate<DstDesc>::type;

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v4r2_dst_runtime_offsets(const Index& src_slice_origin,
                                                               const Index& dst_slice_origin,
                                                               const Index& dst_runtime_strides_div,
                                                               const Index& dst_runtime_strides_mul)
        : mSrcSliceOrigin(src_slice_origin), mDstSliceOrigin(dst_slice_origin),
        mDstRuntimeStridesDiv(dst_runtime_strides_div), mDstRuntimeStridesMul(dst_runtime_strides_mul),
        mDstRuntimeOffsetsPerThread(0)
    {
        static_assert(nDim == SrcDesc::GetNumOfDimension() &&
                          nDim == DstDesc::GetNumOfDimension() && nDim == SliceLengths::Size() &&
                          nDim == SrcDstDimAccessOrder::Size(),
                      "wrong! # of dimensions not the same");

        static_assert(is_valid_sequence_map<SrcDstDimAccessOrder>{}, "wrong! map is not valid");

        static_assert(SliceLengths{}[SrcDstVectorReadWriteDim] %
                              math::lcm(SrcDataPerRead, DstDataPerWrite) ==
                          0,
                      "wrong! cannot evenly divide");

        // TODO:: sanity-check if vectorized memory read/write is allowed on src and dst

        // TODO: be careful of constructor time update
        UpdateDstSliceOriginWithRuntimeOffsets();
    }

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v4r2_dst_runtime_offsets()
        : ThreadwiseGenericTensorSliceCopy_v4r2_dst_runtime_offsets(make_zero_array<index_t, nDim>(),
                                                make_zero_array<index_t, nDim>(),
                                                sequence2array(typename uniform_sequence_gen<nDim, 1>::type{}),
                                                sequence2array(typename uniform_sequence_gen<nDim, 1>::type{}))
    {
    }

    __device__ void UpdateDstSliceOriginWithRuntimeOffsets(){
        Index cur_upper_index = mDstSliceOrigin.GetIndex();
        index_t cur_offset = 0;
        for(index_t i=0; i< nDim; i++){
            index_t p = cur_upper_index[i] / mDstRuntimeStridesDiv[i];
            index_t q = cur_upper_index[i] % mDstRuntimeStridesDiv[i];
            cur_upper_index.At(i) = q;
            cur_offset += p * mDstRuntimeStridesMul[i];
        }
        mDstSliceOrigin = DstCoord{cur_upper_index};
        mDstRuntimeOffsetsPerThread += cur_offset;
    }

    __device__ void SetDstRuntimeStrides(Index dst_runtime_strides_div, Index dst_runtime_strides_mul)
    {
        mDstRuntimeStridesDiv = dst_runtime_strides_div;
        mDstRuntimeStridesMul = dst_runtime_strides_mul;
    }

    __device__ void SetSrcSliceOrigin(SrcCoord src_slice_origin)
    {
        mSrcSliceOrigin = src_slice_origin;
    }

    __device__ void SetDstSliceOrigin(DstCoord dst_slice_origin)
    {
        mDstSliceOrigin = dst_slice_origin;
        UpdateDstSliceOriginWithRuntimeOffsets();
    }

    __device__ index_t CalculateSliceDstRuntimeOffsets(Index & vector_id) const
    {
        index_t slice_offset = 0;
        for(index_t i=0; i< nDim; i++){
            index_t id = vector_id[i];
            slice_offset += (id / mDstRuntimeStridesDiv[i]) * mDstRuntimeStridesMul[i];
            vector_id.At(i) = id % mDstRuntimeStridesDiv[i];
        }
        return slice_offset;
    }

    template <typename SrcData, typename DstData>
    __device__ void Run(const SrcData* p_src, DstData* p_dst) const
    {
        constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerRead>{};
        constexpr auto dst_data_per_access = Number<DstDataPerWrite>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerRead, DstDataPerWrite)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        ford<decltype(long_vector_access_lengths), SrcDstDimAccessOrder>{}([&](
            auto long_vector_access_id) {

            // data id w.r.t slicing-window
            auto long_vector_data_begin_id = long_vector_access_id;
            long_vector_data_begin_id(vector_access_dim) =
                long_vector_size * long_vector_access_id[vector_access_dim];

            // buffer to hold a src long-vector
            SrcData p_src_long_vector[long_vector_size];

            // zero out buffer
            for(index_t i = 0; i < long_vector_size; ++i)
            {
                p_src_long_vector[i] = 0;
            }

            // load data from src to the long-vector buffer
            for(index_t i = 0; i < long_vector_size / src_data_per_access; ++i)
            {
                auto scalar_id               = make_zero_array<index_t, nDim>();
                scalar_id(vector_access_dim) = i * src_data_per_access;

                const index_t buffer_offset = i * src_data_per_access;

                const auto src_coord = mSrcSliceOrigin + (long_vector_data_begin_id + scalar_id);

                // Check src data's valid mapping situation, only check the first data in this src
                //   vector. It's user's responsiblity to make sure all data in the src vector
                //   has the valid/invalid mapping situation
                if(src_coord.IsOffsetValidAssumingUpperIndexIsValid())
                {
                    transfer_data<SrcData,
                                  SrcDataPerRead,
                                  SrcAddressSpace,
                                  AddressSpace::vgpr,
                                  InMemoryDataOperation::none>(
                        p_src, src_coord.GetOffset(), p_src_long_vector, buffer_offset);
                }
            }

            // SrcData to DstData conversion
            DstData p_dst_long_vector[long_vector_size];

            for(index_t i = 0; i < long_vector_size; ++i)
            {
                p_dst_long_vector[i] = type_convert<DstData>{}(p_src_long_vector[i]);
            }

            // store data from the long-vector buffer to dst
            for(index_t i = 0; i < long_vector_size / dst_data_per_access; ++i)
            {
                auto scalar_id               = make_zero_array<index_t, nDim>();
                scalar_id(vector_access_dim) = i * dst_data_per_access;

                const index_t buffer_offset = i * dst_data_per_access;
                auto vector_id = long_vector_data_begin_id + scalar_id; 
                index_t dst_runtime_offset = CalculateSliceDstRuntimeOffsets(vector_id);

                const auto dst_coord = mDstSliceOrigin + vector_id;

                // Check dst data's valid mapping situation, only check the first data in this dst
                //   vector. It's user's responsiblity to make sure all data in the dst vector
                //   has the valid/invalid mapping situation
                if(dst_coord.IsOffsetValidAssumingUpperIndexIsValid())
                {
                    transfer_data<DstData,
                                  DstDataPerWrite,
                                  AddressSpace::vgpr,
                                  DstAddressSpace,
                                  DstInMemOp>(
                        p_dst_long_vector, buffer_offset, p_dst,
                        dst_coord.GetOffset() + mDstRuntimeOffsetsPerThread + dst_runtime_offset);
                }
            }
        });
    }

    template <typename T, bool PositiveDirection>
    __device__ void MoveSrcSliceWindow(const T& step_sizes_,
                                       integral_constant<bool, PositiveDirection>)
    {
        const auto step_sizes = to_array(step_sizes_);

        static_if<PositiveDirection>{}([&](auto) {
            mSrcSliceOrigin += to_array(step_sizes);
        }).Else([&](auto) { mSrcSliceOrigin -= step_sizes; });
    }

    template <typename T, bool PositiveDirection>
    __device__ void MoveDstSliceWindow(const T& step_sizes_,
                                       integral_constant<bool, PositiveDirection>)
    {
        const auto step_sizes = to_array(step_sizes_);

        static_if<PositiveDirection>{}([&](auto) {
            mDstSliceOrigin += step_sizes;
        }).Else([&](auto) { mDstSliceOrigin -= step_sizes; });
        UpdateDstSliceOriginWithRuntimeOffsets();
    }

    private:
    SrcCoord mSrcSliceOrigin;
    DstCoord mDstSliceOrigin;
    index_t mDstRuntimeOffsetsPerThread;
    Index mDstRuntimeStridesDiv;
    Index mDstRuntimeStridesMul;
};

} // namespace ck
#endif
