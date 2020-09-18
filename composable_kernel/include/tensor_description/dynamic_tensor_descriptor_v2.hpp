#ifndef CK_DYNAMIC_TENSOR_DESCRIPTOR_V2_HPP
#define CK_DYNAMIC_TENSOR_DESCRIPTOR_V2_HPP

#include "common_header.hpp"
#include "dynamic_multi_index_transform.hpp"

namespace ck {

template <index_t NDimHidden, typename VisibleDimensionIds>
struct DynamicTensorCoordinate_v2;

template <index_t NTransform, index_t NDimVisible>
struct DynamicTensorCoordinateStep_v2;

template <typename TensorDesc, typename VisibleIndex>
__host__ __device__ constexpr auto
make_dynamic_tensor_coordinate_v2(const TensorDesc& tensor_desc, const VisibleIndex& idx_visible);

template <typename TensorDesc, typename VisibleIndex>
__host__ __device__ constexpr auto
make_dynamic_tensor_coordinate_step_v2(const TensorDesc&, const VisibleIndex& idx_diff_visible);

template <typename TensorDesc, typename TensorCoord, typename TensorCoordStep>
__host__ __device__ void move_dynamic_tensor_coordinate_v2(const TensorDesc& tensor_desc,
                                                           TensorCoord& coord,
                                                           const TensorCoordStep& coord_step);

template <typename TensorDesc, typename TensorCoord>
__host__ __device__ constexpr bool
coordinate_has_valid_offset_assuming_visible_index_is_valid(const TensorDesc& tensor_desc,
                                                            const TensorCoord& coord);

template <typename TensorDesc, typename TensorCoord>
__host__ __device__ constexpr bool coordinate_has_valid_offset(const TensorDesc& tensor_desc,
                                                               const TensorCoord& coord);

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
    // private:
    __host__ __device__ static constexpr index_t GetNumOfTransform() { return Transforms::Size(); }

    __host__ __device__ static constexpr index_t GetNumOfVisibleDimension()
    {
        return VisibleDimensionIds::Size();
    }

    __host__ __device__ static constexpr index_t GetNumOfHiddenDimension()
    {
        constexpr auto all_low_dim_ids =
            unpack([](auto&&... xs) constexpr { return merge_sequences(xs...); },
                   LowerDimensionIdss{});

        constexpr auto all_up_dim_ids =
            unpack([](auto&&... xs) constexpr { return merge_sequences(xs...); },
                   UpperDimensionIdss{});

        constexpr auto all_dim_ids = merge_sequences(all_low_dim_ids, all_up_dim_ids);

        using unique_sort_all_dim_ids = typename sequence_unique_sort<decltype(all_dim_ids),
                                                                      math::less<index_t>,
                                                                      math::equal<index_t>>::type;

        return unique_sort_all_dim_ids::Size();
    }

    constexpr static index_t ntransform_   = GetNumOfTransform();
    constexpr static index_t ndim_visible_ = GetNumOfVisibleDimension();
    constexpr static index_t ndim_hidden_  = GetNumOfHiddenDimension();

    using VisibleIndex   = MultiIndex<ndim_visible_>;
    using HiddenIndex    = MultiIndex<ndim_hidden_>;
    using Coordinate     = DynamicTensorCoordinate_v2<ndim_hidden_, VisibleDimensionIds>;
    using CoordinateStep = DynamicTensorCoordinateStep_v2<ntransform_, ndim_visible_>;

    // public:
    __host__ __device__ explicit constexpr DynamicTensorDescriptor_v2(const Transforms& transforms,
                                                                      index_t element_space_size)
        : transforms_{transforms},
          hidden_lengths_{InitializeHiddenLengths(transforms_, element_space_size)},
          visible_lengths_{hidden_lengths_}
    {
        static_assert(Transforms::Size() == ntransform_ &&
                          LowerDimensionIdss::Size() == ntransform_ &&
                          UpperDimensionIdss::Size() == ntransform_,
                      "wrong! inconsistent # of transformations");

        // TODO check dependency of dimensions is valid
    }

    __host__ __device__ explicit constexpr DynamicTensorDescriptor_v2()
        : DynamicTensorDescriptor_v2(Transforms{}, index_t{0})
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension()
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

        return make_dynamic_tensor_coordinate_v2(*this, idx).GetOffset();
    }

    // private:
    __host__ __device__ constexpr const auto& GetTransforms() const { return transforms_; }

    __host__ __device__ static constexpr auto GetLowerDimensionIdss()
    {
        return LowerDimensionIdss{};
    }

    __host__ __device__ static constexpr auto GetUpperDimensionIdss()
    {
        return UpperDimensionIdss{};
    }

    __host__ __device__ static constexpr auto GetVisibleDimensionIds()
    {
        return VisibleDimensionIds{};
    }

    __host__ __device__ static constexpr auto InitializeHiddenLengths(const Transforms& transforms,
                                                                      index_t element_space_size)
    {
        // zero initialization
        HiddenIndex hidden_lengths{{0}};

        // this is the orignal tensor element space size
        hidden_lengths(0) = element_space_size;

        // lengths for all other hidden dimensions
        static_for<0, ntransform_, 1>{}([&transforms, &hidden_lengths](auto itran) {
            const auto& tran = transforms.At(itran);

            constexpr auto up_dim_ids = UpperDimensionIdss{}.At(itran);

            // lengths_hidden_pick_up contains a reference to lengths_hidden
            auto hidden_lengths_pick_up = pick_array_element(hidden_lengths, up_dim_ids);

            hidden_lengths_pick_up = tran.GetUpperLengths();
        });

        return hidden_lengths;
    }

    // private member variables
    const Transforms transforms_;
    // TODO maybe hidden_lengths_ should use reference_wrapper (reference to transforms_'s member
    // variable lengths_) to save space on stack?
    const HiddenIndex hidden_lengths_;
    // visible_lenths_ contains a reference to hidden_lengths_
    const ArrayElementPicker<const HiddenIndex, VisibleDimensionIds> visible_lengths_;

#if 0
    // friend class
    friend Coordinate;
    friend CoordinateStep;

    // friend function to transform tensor descriptor
    template <typename OldTensorDescriptor,
              typename NewTransforms,
              typename NewLowerDimensionOldVisibleIdss,
              typename NewUpperDimensionNewVisibleIdss>
    __host__ __device__ friend constexpr auto
    transform_dynamic_tensor_descriptor_v2(const OldTensorDescriptor& /* old_tensor_desc */,
                                           const NewTransforms& /* new_transforms */,
                                           NewLowerDimensionOldVisibleIdss,
                                           NewUpperDimensionNewVisibleIdss);

    // friend functions for making and moving tensor coordinate
    template <typename VisibleIndex>
    __host__ __device__ friend constexpr Coordinate
    make_dynamic_tensor_coordinate_v2(const DynamicTensorDescriptor_v2& /* tensor_desc */,
                                      const VisibleIndex& /* idx_visible */);

    template <typename VisibleIndex>
    __host__ __device__ friend constexpr CoordinateStep
    make_dynamic_tensor_coordinate_step_v2(const DynamicTensorDescriptor_v2& /* tensor_desc */,
                                           const VisibleIndex& /* idx_diff_visible */);

    __host__ __device__ friend void
    move_dynamic_tensor_coordinate_v2(const DynamicTensorDescriptor_v2& /* tensor_desc */,
                              Coordinate& /* coord */,
                              const CoordinateStep& /* coord_step */);

    // friend functions for valid offset check
    __host__ __device__ friend constexpr bool
    coordinate_has_valid_offset_assuming_visible_index_is_valid(
        const DynamicTensorDescriptor_v2& tensor_desc, const Coordinate& coord);

    __host__ __device__ friend constexpr bool
    coordinate_has_valid_offset(const DynamicTensorDescriptor_v2& tensor_desc,
                                const Coordinate& coord);
#endif
};

template <index_t NDimHidden, typename VisibleDimensionIds>
struct DynamicTensorCoordinate_v2
{
    // private:
    static constexpr index_t ndim_visible_ = VisibleDimensionIds::Size();

    using HiddenIndex  = MultiIndex<NDimHidden>;
    using VisibleIndex = MultiIndex<ndim_visible_>;

    // public:
    __host__ __device__ explicit constexpr DynamicTensorCoordinate_v2(const HiddenIndex& idx_hidden)
        : idx_hidden_{idx_hidden}, idx_visible_{idx_hidden_}
    {
    }

    __host__ __device__ constexpr const auto& GetIndex() const { return GetVisibleIndex(); }

    __host__ __device__ constexpr index_t GetOffset() const { return idx_hidden_[0]; }

    // private:
    __host__ __device__ constexpr const auto& GetHiddenIndex() const { return idx_hidden_; }

    __host__ __device__ auto& GetHiddenIndex() { return idx_hidden_; }

    __host__ __device__ constexpr const auto& GetVisibleIndex() const { return idx_visible_; }

    __host__ __device__ auto& GetVisibleIndex() { return idx_visible_; }

    // private member variables
    HiddenIndex idx_hidden_;
    // idx_visible_ contains a reference to idx_hidden_
    ArrayElementPicker<HiddenIndex, VisibleDimensionIds> idx_visible_;

#if 0
    // friend functions for making and updating tensor coordinate
    template <typename TensorDesc>
    __host__ __device__ friend constexpr DynamicTensorCoordinate_v2
    make_dynamic_tensor_coordinate_v2(const TensorDesc& /* tensor_desc */,
                                      const VisibleIndex& /* idx_visible */);

    template <typename TensorDesc, typename TensorCoordStep>
    __host__ __device__ friend void move_dynamic_tensor_coordinate_v2(
        const TensorDesc& /* tensor_desc */,
        DynamicTensorCoordinate_v2& /* coord */,
        const TensorCoordStep& /* coord_step */);
#endif
};

template <index_t NTransform, index_t NDimVisible>
struct DynamicTensorCoordinateStep_v2
{
    // private:
    using VisibleIndex = MultiIndex<NDimVisible>;

    // public:
    __host__ __device__ explicit constexpr DynamicTensorCoordinateStep_v2(
        const VisibleIndex& idx_diff_visible, const Array<bool, NTransform>& do_transforms)
        : idx_diff_visible_{idx_diff_visible}, do_transforms_{do_transforms}
    {
    }

    __host__ __device__ constexpr const auto& GetIndexDiff() const { return GetVisibleIndexDiff(); }

    // private:
    __host__ __device__ constexpr const auto& GetVisibleIndexDiff() const
    {
        return idx_diff_visible_;
    }

    //  private:
    const VisibleIndex idx_diff_visible_;
    const Array<bool, NTransform> do_transforms_;

#if 0
    // friend functions for updating tensor coordinate
    template <typename TensorDesc>
    __host__ __device__ friend constexpr DynamicTensorCoordinateStep_v2
    make_dynamic_tensor_coordinate_step_v2(const TensorDesc& /* tensor_desc */,
                                           const VisibleIndex& /* idx_visible */);

    template <typename TensorDesc, index_t NDimHidden, typename VisibleDimensionIds>
    __host__ __device__ friend void move_dynamic_tensor_coordinate_v2(
        const TensorDesc& /* tensor_desc */,
        DynamicTensorCoordinate_v2<NDimHidden, VisibleDimensionIds>& /* coord */,
        const DynamicTensorCoordinateStep_v2& /* coord_step */);
#endif
};

// TODO: Fix this! This is insane, to use an ugly struct instead of lambda because lambda
// doesn't have constructor, and to put it outside the scope where it is used
// (transform_dynamic_tensor_descriptor_v2) because template cannot be defined inside a function
// template
template <typename NewTransforms>
struct lambda_get_up_dim_num
{
    template <typename I>
    __host__ __device__ constexpr auto operator()(I) const
    {
        using Tran = remove_reference_t<decltype(NewTransforms{}.At(I{}))>;
        return Number<Tran::GetNumOfUpperDimension()>{};
    }
};

template <typename OldTensorDescriptor,
          typename NewTransforms,
          typename NewLowerDimensionOldVisibleIdss,
          typename NewUpperDimensionNewVisibleIdss>
__host__ __device__ constexpr auto
transform_dynamic_tensor_descriptor_v2(const OldTensorDescriptor& old_tensor_desc,
                                       const NewTransforms& new_transforms,
                                       NewLowerDimensionOldVisibleIdss,
                                       NewUpperDimensionNewVisibleIdss)
{
    // lower dimension's hidden idss
    // convert lower dimension visible idss (tuple of sequences) to hidden idss (tuple of
    // sequences)
    constexpr auto low_dim_hidden_idss = transform_tuples(
        // convert lower dimension visible ids (a sequence) to hidden ids (a sequence)
        [](auto low_dim_visible_ids) constexpr {
            return transform_sequences(
                // convert lower dimension visible id to hidden id
                [](auto low_dim_visible_id) constexpr {
                    return OldTensorDescriptor::GetVisibleDimensionIds()[low_dim_visible_id];
                },
                low_dim_visible_ids);
        },
        NewLowerDimensionOldVisibleIdss{});

    constexpr index_t num_new_transform = NewTransforms::Size();

    // upper dimension's hidden idss
    constexpr index_t old_hidden_dim_number = OldTensorDescriptor::GetNumOfHiddenDimension();

    constexpr auto up_dim_numbers =
        generate_sequence(lambda_get_up_dim_num<NewTransforms>{}, Number<num_new_transform>{});

    constexpr auto up_dim_numbers_scan = merge_sequences(
        Sequence<0>{}, inclusive_scan_sequence(up_dim_numbers, math::plus<index_t>{}, Number<0>{}));

    constexpr auto up_dim_hidden_idss =
        generate_tuple([ old_hidden_dim_number, up_dim_numbers_scan ](auto i) constexpr {
            return
                typename arithmetic_sequence_gen<old_hidden_dim_number + up_dim_numbers_scan[i],
                                                 old_hidden_dim_number + up_dim_numbers_scan[i + 1],
                                                 1>::type{};
        },
                       Number<num_new_transform>{});

    // new visible dimension's hidden ids
    constexpr auto unordered_new_visible_dim_hidden_ids =
        unpack([](auto... xs) { return merge_sequences(xs...); }, up_dim_hidden_idss);

    constexpr auto new_visible_dim_unordered2ordered = unpack(
        [](auto... xs) { return merge_sequences(xs...); }, NewUpperDimensionNewVisibleIdss{});

    constexpr auto new_visible_dim_hidden_ids =
        unordered_new_visible_dim_hidden_ids.ReorderGivenOld2New(new_visible_dim_unordered2ordered);

    // put everything together
    const auto all_transforms = merge_tuples(old_tensor_desc.GetTransforms(), new_transforms);

    constexpr auto all_low_dim_hidden_idss =
        merge_tuples(OldTensorDescriptor::GetLowerDimensionIdss(), low_dim_hidden_idss);

    constexpr auto all_up_dim_hidden_idss =
        merge_tuples(OldTensorDescriptor::GetUpperDimensionIdss(), up_dim_hidden_idss);

    return DynamicTensorDescriptor_v2<decltype(all_transforms),
                                      decltype(all_low_dim_hidden_idss),
                                      decltype(all_up_dim_hidden_idss),
                                      decltype(new_visible_dim_hidden_ids)>{
        all_transforms, old_tensor_desc.GetElementSpaceSize()};
}

template <typename TensorDesc, typename VisibleIndex>
__host__ __device__ constexpr auto
make_dynamic_tensor_coordinate_v2(const TensorDesc& tensor_desc, const VisibleIndex& idx_visible)
{
    static_assert(TensorDesc::GetNumOfDimension() == VisibleIndex::Size(),
                  "wrong! # of dimension inconsistent");

    constexpr index_t ntransform   = TensorDesc::GetNumOfTransform();
    constexpr index_t ndim_hidden  = TensorDesc::GetNumOfHiddenDimension();
    constexpr index_t ndim_visible = TensorDesc::GetNumOfVisibleDimension();
    constexpr auto visible_dim_ids = TensorDesc::GetVisibleDimensionIds();

    MultiIndex<ndim_hidden> idx_hidden;

    // initialize visible index
    auto idx_hidden_pick_visible = pick_array_element(idx_hidden, visible_dim_ids);
    idx_hidden_pick_visible      = idx_visible;

    // calculate hidden index
    static_for<ntransform - 1, -1, -1>{}([&tensor_desc, &idx_hidden](auto itran) {
        const auto& tran        = tensor_desc.GetTransforms().At(itran);
        constexpr auto dims_low = TensorDesc::GetLowerDimensionIdss().At(itran);
        constexpr auto dims_up  = TensorDesc::GetUpperDimensionIdss().At(itran);

        const auto idx_up = pick_array_element(idx_hidden, dims_up);
        auto idx_low      = pick_array_element(idx_hidden, dims_low);

        tran.CalculateLowerIndex(idx_low, idx_up);
    });

    // better to use std::move?
    return DynamicTensorCoordinate_v2<ndim_hidden, decltype(visible_dim_ids)>{idx_hidden};
}

template <typename TensorDesc, typename VisibleIndex>
__host__ __device__ constexpr auto
make_dynamic_tensor_coordinate_step_v2(const TensorDesc&, const VisibleIndex& idx_diff_visible)
{
    static_assert(TensorDesc::GetNumOfDimension() == VisibleIndex::Size(),
                  "wrong! # of dimension inconsistent");

    constexpr index_t ntransform   = TensorDesc::GetNumOfTransform();
    constexpr index_t ndim_hidden  = TensorDesc::GetNumOfHiddenDimension();
    constexpr index_t ndim_visible = TensorDesc::GetNumOfVisibleDimension();
    constexpr auto visible_dim_ids = TensorDesc::GetVisibleDimensionIds();

    Array<bool, ntransform> do_transforms{false};

    Array<bool, ndim_hidden> non_zero_diff{false};

    auto non_zero_diff_pick_visible = pick_array_element(non_zero_diff, visible_dim_ids);

    static_for<0, ndim_visible, 1>{}([&non_zero_diff_pick_visible, &idx_diff_visible](auto i) {
        non_zero_diff_pick_visible(i) = (idx_diff_visible[i] != 0);
    });

    static_for<ntransform - 1, -1, -1>{}([&do_transforms, &non_zero_diff](auto itran) {
        constexpr auto dims_low = TensorDesc::GetLowerDimensionIdss().At(itran);
        constexpr auto dims_up  = TensorDesc::GetUpperDimensionIdss().At(itran);

        const auto non_zero_diff_pick_up = pick_array_element(non_zero_diff, dims_up);
        auto non_zero_diff_pick_low      = pick_array_element(non_zero_diff, dims_low);

        // if any of upper index diff components is non-zero, then
        //   1) Need to do this transform
        //   2) all components of lower index diff will assume to be non-zero and need to be
        //   computed
        const bool idx_diff_up_has_non_zero =
            reduce_on_array(non_zero_diff_pick_up, [](auto a, auto b) { return a or b; }, false);

        do_transforms(itran) = idx_diff_up_has_non_zero;

        static_for<0, dims_low.Size(), 1>{}(
            [&non_zero_diff_pick_low, &idx_diff_up_has_non_zero](auto i) {
                non_zero_diff_pick_low(i) = idx_diff_up_has_non_zero;
            });
    });

    return DynamicTensorCoordinateStep_v2<ntransform, ndim_visible>{idx_diff_visible,
                                                                    do_transforms};
}

template <typename TensorDesc, typename TensorCoord, typename TensorCoordStep>
__host__ __device__ void move_dynamic_tensor_coordinate_v2(const TensorDesc& tensor_desc,
                                                           TensorCoord& coord,
                                                           const TensorCoordStep& coord_step)
{
    constexpr index_t ndim_hidden  = TensorDesc::GetNumOfHiddenDimension();
    constexpr index_t ndim_visible = TensorDesc::GetNumOfVisibleDimension();
    constexpr index_t ntransform   = TensorDesc::GetNumOfTransform();

    using HiddenIndex = MultiIndex<ndim_hidden>;

    // this is what needs to be calculated
    auto idx_diff_hidden = HiddenIndex{{0}};

    // initialize visible index diff
    //   idx_diff_hidden_pick_visible contains reference to idx_diff_hidden
    auto idx_diff_hidden_pick_visible =
        pick_array_element(idx_diff_hidden, TensorDesc::GetVisibleDimensionIds());

    idx_diff_hidden_pick_visible = coord_step.GetVisibleIndexDiff();

    // this is what needs to be updated
    auto& idx_hidden = coord.GetHiddenIndex();

    // update visible index
    auto idx_hidden_pick_visible =
        pick_array_element(idx_hidden, TensorDesc::GetVisibleDimensionIds());
    idx_hidden_pick_visible += coord_step.GetIndexDiff();

    // update rest of hidden index
    static_for<ntransform - 1, -1, -1>{}([&tensor_desc, &idx_hidden, &idx_diff_hidden](auto itran) {
        const auto& tran        = tensor_desc.GetTransforms().At(itran);
        constexpr auto dims_low = TensorDesc::GetLowerDimensionIdss().At(itran);
        constexpr auto dims_up  = TensorDesc::GetUpperDimensionIdss().At(itran);

        // this const is for ArrayElementPicker, Array itself may not be const
        const auto idx_up = pick_array_element(idx_hidden, dims_up);
        auto idx_low      = pick_array_element(idx_hidden, dims_low);

        const auto idx_diff_up = pick_array_element(idx_diff_hidden, dims_up);
        auto idx_diff_low      = pick_array_element(idx_diff_hidden, dims_low);

        tran.CalculateLowerIndexDiff(idx_diff_low, idx_diff_up, idx_low, idx_up);

        // update idx_low
        idx_low += idx_diff_low;
    });
}

template <typename TensorDesc, typename TensorCoord>
__host__ __device__ constexpr bool
coordinate_has_valid_offset_assuming_visible_index_is_valid(const TensorDesc& tensor_desc,
                                                            const TensorCoord& coord)
{
    bool valid = true;

    constexpr index_t ntransform = TensorDesc::GetNumOfTransform();

    const auto& idx_hidden = coord.GetHiddenIndex();

    static_for<ntransform - 1, -1, -1>{}([&tensor_desc, &idx_hidden, &valid](auto itran) {
        const auto tran = tensor_desc.GetTransforms().At(itran);

        // check validity, only if current transformation does not always has a valid mapping
        if constexpr(!decltype(tran)::IsValidUpperIndexAlwaysMappedToValidLowerIndex())
        {
            const auto idx_up =
                pick_array_element(idx_hidden, TensorDesc::GetUpperDimensionIdss().At(itran));

            valid = valid && tran.IsValidUpperIndexMappedToValidLowerIndex(idx_up);
        }
    });

    return valid;
}

template <typename TensorDesc, typename TensorCoord>
__host__ __device__ constexpr bool coordinate_has_valid_offset(const TensorDesc& tensor_desc,
                                                               const TensorCoord& coord)
{
    // check visible index
    const auto& idx_visible = coord.GetVisibleIndex();

    bool is_visible_index_valid = true;

    static_for<0, TensorDesc::GetNumOfDimension(), 1>{}(
        [&is_visible_index_valid, &idx_visible, &tensor_desc](auto i) {
            is_visible_index_valid =
                is_visible_index_valid &&
                (idx_visible[i] >= 0 && idx_visible[i] < tensor_desc.GetLength(i));
        });

    // check other hidden index
    return is_visible_index_valid &&
           coordinate_has_valid_offset_assuming_visible_index_is_valid(tensor_desc, coord);
}

} // namespace ck
#endif
