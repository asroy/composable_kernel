#ifndef CK_DYNAMIC_TENSOR_DESCRIPTOR_V2_HPP
#define CK_DYNAMIC_TENSOR_DESCRIPTOR_V2_HPP

#include "common_header.hpp"
#include "dynamic_multi_index_transform.hpp"

namespace ck {

template <index_t NDimHidden, typename VisibleDimensionIds>
struct DynamicTensorCoordinate_v2;

template <index_t NTransform, index_t NDimVisible>
struct DynamicTensorCoordinateStep_v2;

// Transforms: Tuple<transforms...>
// LowerDimensionIdss : Tuple<Sequence<...>, ...>
// UpperDimensionIdss : Tuple<Sequence<...>, ...>
// VisibleDimensionIds> : Sequence<...>
template <typename Transforms,
          typename LowerDimensionIdss,
          typename UpperDimensionIdss,
          typename VisibleDimensionIds>
struct DynamicTensorDescriptor_v2
{
    constexpr static index_t ntransform_   = GetNumOfTransform();
    constexpr static index_t ndim_visible_ = GetNumOfVisibleDimension();
    constexpr static index_t ndim_hidden_  = GetNumOfHiddenDimension();

    using VisibleIndex = MultiIndex<ndim_visible_>;
    using HiddenIndex  = MultiIndex<ndim_hidden_>;

    __host__ __device__ explicit constexpr DynamicTensorDescriptor_v2(const Transforms& transforms,
                                                                      index_t element_space_size)
        : transforms_{transforms},
          hidden_lengths_{InitializeHiddenLengths(transforms_, element_space_size)},
          visble_lengths_{hidden_lengths_}
    {
        static_assert(Transforms::Size() == ntransforms_ &&
                          LowerDimensionIdss::Size() == ntransforms_ &&
                          UpperDimensionIdss::Size() == ntransforms_,
                      "wrong! inconsistent # of transformations");

        // TODO check dependency of dimensions is valid
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension() const
    {
        return GetNumOfVisibleDimension();
    }

    __host__ __device__ constexpr index_t GetLength(index_t idim) const
    {
        return visible_lengths_[idim];
    }

    __host__ __device__ constexpr const auto& GetLengths() const { return visible_lengths_; }

    // maybe this result should be saved as a member variable
    __host__ __device__ constexpr index_t GetElementSize() const
    {
        return reduce_on_array(GetLengths(), math::multiplies<index_t>{}, index_t{1});
    }

    __host__ __device__ constexpr index_t GetElementSpaceSize() const { return hidden_lengths_[0]; }

    template <typename Idx>
    __host__ __device__ constexpr index_t CalculateOffset(const Idx& idx) const
    {
        static_assert(Idx::Size() == GetNumOfDimension(), "wrong! inconsistent # of dimension");

        return make_tensor_coordinate_v2(*this, idx).GetOffset();
    }

    private:
    __host__ __device__ static constexpr index_t GetNumOfVisibleDimension()
    {
        return VisibleDimensionIds::Size();
    }

    __host__ __device__ static constexpr index_t GetNumOfHiddenDimension()
    {
        constexpr auto all_low_dim_ids =
            unpack([](auto&&... xs) constexpr { return merge_sequences(xs...); },
                   LowerDimsionIdss{});

        constexpr auto all_up_dim_ids =
            unpack([](auto&&... xs) constexpr { return merge_sequences(xs...); },
                   UpperDimsionIdss{});

        constexpr auto all_dim_ids = merge_sequenses(all_low_dim_ids, all_up_dim_ids);

        using unique_sort_all_dim_ids = sequence_unique_sort<decltype(all_dim_ids),
                                                             math::less<index_t>,
                                                             math::equal<index_t>>::type;

        return uniqie_sort_all_dim_ids::type::Size();
    }

    __host__ __device__ static constexpr index_t GetNumOfTransform() { return Transforms::Size(); }

    __host__ __device__ constexpr const auto& GetTransforms() const { return transforms_; }

    __host__ __device__ static constexpr auto GetLowerDimensionIdss()
    {
        return LowerDimensionIdss{};
    }

    __host__ __device__ static constexpr auto GetUpperDimensionIdss()
    {
        return UpperDimensionIdss{};
    }

    __host__ __device__ static constexpr index_t GetVisibleDimensionIds()
    {
        return VisibleDimensionIds{};
    }

    __host__ __device__ static constexpr auto InitializeHiddenLengths(const Transforms& transforms,
                                                                      index_t element_space_size)
    {
        HiddenIndex lengths_hidden = make_zero_multi_index<ndim_hidden_>();

        // this is the orignal tensor element space size
        lengths_hidden(0) = element_space_size;

        // lengths for all other hidden dimensions
        static_for<0, ntransform_, 1>{}([&](auto itran) {
            const auto& tran = transforms.At(itran);

            constexpr auto up_dim_ids = UpperDimensionIdss::At(itran);

            const auto lengths_up_pick = pick_array_element(lengths_hidden, up_dim_ids);

#pragma unroll
            for(index_t i = 0; i < lengths_low.Size(); ++i)
            {
                lengths_low_pick(i) = tran.GetUpperLengths()[i];
            }
        });

        return lengths_hidden;
    }

    // private member variables
    const Transforms transforms_;
    // TODO maybe hidden_lengths_ should use reference_wrapper to save space on stack?
    const HiddenIndex hidden_lengths_;
    // visible_lenths_ contains a reference to hidden_lengths_
    const ArrayElementPicker<HiddenIndex, VisibleDimensionIds> visible_lengths_;

    // friend functions for making and updating tensor coordinate
    __host__
        __device__ friend constexpr DynamicTensorCoordinate_v2<ndim_hidden_, VisibleDimensionIds>
        make_tensor_coordinate_v2(const DynamicTensorDescriptor_v2& /* tensor_desc */,
                                  const VisibleIndex& /* idx_visible */);

    __host__ __device__ friend constexpr DynamicTensorCoordinateStep_v2<ntransform_, ndim_visible_>
    make_tensor_coordinate_step_v2(const DynamicTensorDescriptor_v2& /* tensor_desc */,
                                   const VisibleIndex& /* idx_diff_visible */);

    __host__ __device__ friend void move_tensor_coordinate_v2(
        const DynamicTensorDescriptor_v2& /* tensor_desc */,
        DynamicTensorCoordinate_v2<ndim_hidden_, VisibleDimensionIds>& /* coord */,
        const DynamicTensorCoordinateStep_v2<ntransform_, ndim_visible_>& /* coord_step */);
};

template <index_t NDimHidden, typename VisibleDimensionIds>
struct DynamicTensorCoordinate_v2
{
    constexpr index_t ndim_visible_ = VisbleDimension::Size();

    using HiddenIndex  = MultiIndex<NDimHidden>;
    using VisibleIndex = MultiIndex<ndim_visible_>;

    __host__ __device__ explicit constexpr DynamicTensorCoordinate_v2(const HiddenIndex& idx_hidden)
        : idx_hidden_{idx_hidden}, idx_visible_{idx_hidden_}
    {
    }

    __host__ __device__ constexpr const auto& GetIndex() const { GetVisibleIndex(); }

    __host__ __device__ constexpr index_t GetOffset() const { return idx_hidden_[0]; }

    private:
    __host__ __device__ constexpr const auto& GetHiddenIndex() const { return idx_hidden_; }

    __host__ __device__ auto& GetHiddenIndex() { return idx_hidden_; }

    __host__ __device__ constexpr const auto& GetVisibleIndex() const { return idx_visible_; }

    __host__ __device__ auto& GetVisibleIndex() { return idx_visible_; }

    // private member variables
    HiddenIndex idx_hidden_;
    // idx_visible_ contains a reference to idx_hidden_
    ArrayElementPicker<HiddenIndex, VisibleDimensionIds> idx_visible_;

    // friend functions for making and updating tensor coordinate
    template <typename TensorDesc>
    __host__ __device__ friend constexpr DynamicTensorCoordinate_v2
    make_tensor_coordinate_v2(const TensorDesc& /* tensor_desc */,
                              const VisibleIndex& /* idx_visible */);

    template <typename TensorDesc>
    __host__ __device__ friend void move_tensor_coordinate_v2(
        const TensorDesc& /* tensor_desc */,
        DynamicTensorCoordinate_v2& /* coord */,
        const DynamicTensorCoordinateStep_v2<TensorDesc::GetNumOfTransform(),
                                             ndim_visible_>& /* coord_step */);
};

template <index_t NTransform, index_t NDimVisible>
struct DynamicTensorCoordinateStep_v2
{
    using VisibleIndex = MultiIndex<NDimVisible>;

    __host__ __device__ explicit constexpr DynamicTensorCoordinateStep_v2(
        const VisibleIndex& idx_diff_visible, const Array<bool, NTransform>& do_transforms)
        : idx_diff_visible_{idx_diff_visible}, do_transforms_{do_transforms}
    {
    }

    private:
    const VisibleIndex idx_diff_visible_;
    const Array<bool, NTransform> do_transforms_;

    // friend functions for updating tensor coordinate
    template <typename TensorDesc>
    __host__ __device__ friend constexpr DynamicTensorCoordinateStep_v2
    make_tensor_coordinate_step_v2(const TensorDesc& /* tensor_desc */,
                                   const VisibleIndex& /* idx_visible */);

    template <typename TensorDesc, index_t NDimHidden, typename VisibleDimensionIds>
    __host__ __device__ friend void move_tensor_coordinate_v2(
        const TensorDesc& /* tensor_desc */,
        DynamicTensorCoordinate_v2<NDimHidden, VisibleDimensionIds>& /* coord */,
        const DynamicTensorCoordinateStep_v2& /* coord_step */);
};

template <typename TensorDesc, typename VisibleIndex>
__host__ __device__ constexpr auto make_tensor_coordinate_v2(const TensorDesc& tensor_desc,
                                                             const VisibleIndex& idx_visible)
{
    static_assert(tensor_desc.GetNumOfDimension() == idx_visible.Size(),
                  "wrong! # of dimension inconsistent");

    constexpr index_t ntransform   = tensor_desc.GetNumOfTransformation();
    constexpr index_t ndim_hidden  = tensor_desc.GetNumOfHiddenDimension();
    constexpr index_t ndim_visible = tensor_desc.GetNumOfVisibleDimension();

    MultiIndex<ndim_hidden> idx_hidden;

    auto idx_visible_pick = pick_array_element(idx_hidden, tensor_desc.GetVisibleDimensionIds());

    // initialize visible index
#pragma unroll
    for(index_t i < ndim_visible; i < ndim_visible, ++i)
    {
        idx_visible_pick(i) = idx_visible[i];
    }

    // calculate hidden index
    static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
        const auto& tran        = transforms_.At(itran);
        constexpr auto dims_low = LowerDimensionIdss::At(itran);
        constexpr auto dims_up  = UpperDimensionIdss::At(itran);

        const auto idx_up = pick_array_element(idx_hidden_, dim_up);
        auto idx_low      = pick_array_element(idx_hidden_, dim_low);

        tran.CalculateLowerIndex(idx_up, idx_low);
    });

    // better to use std::move?
    return DynamicTensorCoordinate_v2{idx_hidden};
}

template <typename TensorDesc, typename VisibleIndex>
__host__ __device__ constexpr auto
make_tensor_coordinate_step_v2(const TensorDesc& tensor_desc, const VisibleIndex& idx_diff_visible)
{
    static_assert(tensor_desc.GetNumOfDimension() == idx_visible.Size(),
                  "wrong! # of dimension inconsistent");

    constexpr index_t ntransform   = tensor_desc.GetNumOfTransformation();
    constexpr index_t ndim_hidden  = tensor_desc.GetNumOfHiddenDimension();
    constexpr index_t ndim_visible = tensor_desc.GetNumOfVisibleDimension();

    Array<bool, ntransform> do_transforms = {false};

    Array<bool, ndim_hidden> non_zero_diff = {false};

    auto non_zero_diff_pick_visible =
        pick_array_element(non_zero_diff, tensor_desc.GetVisibleDimensionIds());

#pragma unroll
    for(index_t i < ndim_visible; i < ndim_visible, ++i)
    {
        non_zero_diff_pick_visible(i) = (idx_diff_visible[i] != 0);
    }

    static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
        const auto& tran        = tensor_desec.GetTransforms().At(itran);
        constexpr auto dims_low = tensor_desc.GetLowerDimensionIdss().At(itran);
        constexpr auto dims_up  = tensor_Desc.GetUpperDimensionIdss().At(itran);

        const auto non_zero_diff_pick_up = pick_array_element(non_zero_diff, dims_up);
        auto non_zero_diff_pick_low      = pick_array_element(non_zero_diff, dims_low);

        // if any of upper index diff components is non-zero, then
        //   1) Need to do this transform
        //   2) all components of lower index diff will assume to be non-zero and need to be
        //   computed
        const bool idx_diff_up_has_non_zero =
            reduce_on_array(non_zero_diff_pick_up, [](auto a, auto b) { return a or b; }, false);

        do_transforms(itran) = idx_diff_up_has_non_zero;

#pragma unroll
        for(index_t i = 0; i < dims_low.Size(); ++i)
        {
            non_zero_diff_pick_low(i) = idx_diff_up_has_non_zero;
        }
    });

    return do_transforms;
}

template <typename TensorDesc, typename TensorCoord, typename TensorCoordStep>
__host__ __device__ void move_tensor_coordinate_v2(const TensorDesc& tensor_desc,
                                                   TensorCoord& coord,
                                                   const TensorCoordStep& coord_step)
{
    constexpr index_t ndim_hidden  = tensor_desc.GetNumOfHiddenDimension();
    constexpr index_t ndim_visible = tensor_desc.GetNumOfVisibleDimension();
    constexpr index_t ntransform   = tensor_desc.GetNumOfTransform();

    // this is what needs to be calculated
    auto idx_diff_hidden = make_zero_multi_index<ndim_hidden>();
    const auto idx_diff_visible_pick =
        pick_array_element(idx_diff_hidden, tensor_desc.GetVisibleDimensionIds());

    // initialize visible index diff
#pragma unroll
    for(index_t i = 0; i < ndim_visible_; ++i)
    {
        idx_diff_visible_pick(i) = coord_step.GetVisibleIndexDiff()[i];
    }

    // this is what needs to be updated
    auto& idx_hidden = coord.GetHiddenIndex();

    // update hidden index
    static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
        const auto& tran        = tensor_desc.GetTransformations().At(itran);
        constexpr auto dims_low = tensor_desc.GetLowerDimensionIdss().At(itran);
        constexpr auto dims_up  = tensor_desc.GetUpperDimensionIdss().At(itran);

        // this const is for ArrayElementPicker, Array itself may not be const
        const auto idx_up  = pick_array_element(idx_hidden, dim_up);
        const auto idx_low = pick_array_element(idx_hidden, dim_low);

        const auto idx_diff_up  = pick_array_element(idx_diff_hidden, dim_up);
        const auto idx_diff_low = pick_array_element(idx_diff_hidden, dim_low);

        tran.CalculateLowerIndexDiff(idx_diff_low, idx_diff_up, idx_low, idx_up);

        // update idx_low
        idx_low += idx_diff_low;
    });
}

template <typename TensorDesc, typename TensorCoord>
__host__ __device__ bool constexpr coordinate_has_valid_offset_assuming_visible_index_is_valid(
    const TensorDesc& tensor_desc, const TensorCoord& coord)
{
    bool valid = true;

    constexpr index_t ntransform = tensor_desc.GetNumOfTransform();

    const auto& idx_hidden = coord.GetHiddenIndex();

    static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
        const auto tran = tensor_desc.GetTransforms().At(itran);

        // check validity, only if current transformation does not always has a valid mapping
        if constexpr(!decltype(tran)::IsValidUpperIndexAlwaysMappedToValidLowerIndex())
        {
            const auto idx_up =
                pick_array_element(idx_hidden, tensor_desc.GetUpperDimensionIdss().At(itran));

            valid = valid && tran.IsValidUpperIndexMappedToValidLowerIndex(idx_up);
        }
    });

    return valid;
}

template <typename TensorDesc, typename TensorCoord>
__host__ __device__ bool constexpr coordinate_has_valid_offset(const TensorDesc& tensor_desc,
                                                               const TensorCoord& coord)
{
    // check visible index
    const auto& idx_visible = coord.GetVisibleIndex();

    bool is_visible_index_valid = true;

#pragma unroll
    for(index_t i = 0; i < tensor_desc.GetNumOfDimension(); ++i)
    {
        is_visible_index_valid = is_visible_index_valid &&
                                 (idx_visible[i] >= 0 && idx_visible[i] < tensor_desc.GetLength(i));
    }

    // check other hidden index
    return is_visible_index_valid &&
           coordinate_has_valid_offset_assuming_visible_index_is_valid(tensor_desc, coord);
}

} // namespace ck
#endif
