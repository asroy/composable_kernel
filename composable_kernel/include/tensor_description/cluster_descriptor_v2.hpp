#ifndef CK_CLUSTER_DESCRIPTOR_V2_HPP
#define CK_CLUSTER_DESCRIPTOR_V2_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

// Transforms: Tuple<transforms...>
// LowerDimensionIdss : Tuple<Sequence<...>, ...>
// UpperDimensionIdss : Tuple<Sequence<...>, ...>
// LowestDimensionIds> : Sequence<...>
// UppestDimensionIds> : Sequence<...>
template <typename Transforms,
          typename LowerDimensionIdss,
          typename UpperDimensionIdss,
          typename BottomDimensionIds,
          typename TopDimensionIds>
struct ChainedMultiIndexTransform
{
    __host__ __device__ static constexpr auto InitializeElementSize(const Transforms& transforms)
    {
        const auto lengths = generate_tuple(
            [&](auto idim_top) {
                constexpr auto tmp = GetTransformAndItsUpperDimension(idim_top);

                constexpr index_t itran   = tmp[Number<0>{}];
                constexpr index_t idim_up = tmp[Number<1>{}];
                constexpr bool found      = tmp[Number<2>{}];

                static_assert(found == true,
                              "wrong! not found matching transformation and upper-dimension");

                const auto length =
                    transforms[Number<itran>{}].GetUpperLengths()[Number<idim_up>{}];

                return length;
            },
            Number<ndim_top_>{});

        // TODO: make container_reduce support tuple of Number and index_t
        return container_reduce(lengths, math::multiplies_v2{}, Number<1>{});
    }

    template <index_t IDim>
    __host__ __device__ static constexpr auto GetTransformAndItsUpperDimension(Number<IDim>)
    {
        constexpr auto idim_top = Number<IDim>{};

        constexpr index_t idim_hidden = topDimensionIds::At(idim_top);

        index_t itran_found   = 0;
        index_t idim_up_found = 0;
        bool found            = false;

        static_for<0, ntransform_, 1>{}([&](auto itran) {
            constexpr auto up_dim_ids = UpperDimensionIdss{}[itran];

            static_for<0, up_dim_ids.Size(), 1>{}([&](auto idim_up) {
                if constexpr(up_dim_ids[idim_up] == idim_hidden)
                {
                    itran_found   = itran;
                    idim_up_found = idim_up;
                    found         = true;
                }
            });
        });

        return make_tuple(itran_found, idim_up_found, found);
    }

    constexpr static index_t ntransform_   = GetNumOfTransform();
    constexpr static index_t ndim_hidden_  = GetNumOfHiddenDimension();

    using HiddenIndex  = MultiIndex<ndim_hidden_>;
    using BottomIndex  = MultiIndex<ndim_bottom_>;
    using TopIndex  = MultiIndex<ndim_top_>;

    // may be index_t or Number<>
    using ElementSize = remove_cv_t<decltype(InitializeElementSize(Transforms{}))>;

    public:
    __host__ __device__ constexpr ChainedMultiIndexTransform() = default;

    __host__ __device__ constexpr ChainedMultiIndexTransform(const Transforms& transforms)
        : transforms_{transforms},
          element_size_{InitializeElementSize(transforms)}
    {
        static_assert(Transforms::Size() == ntransform_ &&
                          LowerDimensionIdss::Size() == ntransform_ &&
                          UpperDimensionIdss::Size() == ntransform_,
                      "wrong! inconsistent # of transformations");

        // TODO check dependency of dimensions is valid
    }

    __host__ __device__ constexpr auto GetElementSize() const { return element_size_; }

    template<typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        static_assert(TopIdx::Size() == TopDimensionIds::Size(),
                      "wrong! # of dimension inconsistent");

        constexpr index_t ntransform   = GetNumOfTransform();
        constexpr index_t ndim_hidden  = GetNumOfHiddenDimension();

        MultiIndex<ndim_hidden> idx_hidden;

        // initialize uppest index
        set_container_subset(idx_hidden, GetTopDimensionIds(), idx_top);

        // calculate hidden index
        static_for<ntransform, 0, -1>{}([&](auto itran_p1) {
            auto itran              = itran_p1 - Number<1>{};
            const auto& tran        = GetTransforms().At(itran);
            constexpr auto dims_low = GetLowerDimensionIdss().At(itran);
            constexpr auto dims_up  = GetUpperDimensionIdss().At(itran);

            const auto idx_up = get_container_subset(idx_hidden, dims_up);

            MultiIndex<dims_low.Size()> idx_low;

            tran.CalculateLowerIndex(idx_low, idx_up);

            set_container_subset(idx_hidden, dims_low, idx_low);
        });

        return get_container_subset(idx_hidden, BottomDimensionIds{});
    }

    private:
    // TODO make these private
    Transforms transforms_;
    ElementSize element_size_;
};

template <
          typename Transforms,
          typename LowerDimensionIdss,
          typename UpperDimensionIdss>
__host__ __device__ constexpr auto
make_chained_multi_index_transform(const Transforms& transforms,
                                    LowerDimensionIdss,
                                    UpperDimensionIdss)
{
}

template <typename OldChainedTransform,
          typename NewTransforms,
          typename NewLowerDimensionOldTopIdss,
          typename NewUpperDimensionNewTopIdss>
__host__ __device__ constexpr auto
append_chain_multi_index_transform(const OldChainedTransform& old_chained_transform,
                                    const NewTransforms& new_transforms,
                                    NewLowerDimensionOldTopIdss,
                                    NewUpperDimensionNewTopIdss)
{
    // lower dimension's hidden idss
    // convert lower dimension old top idss (tuple of sequences) to new hidden idss (tuple of
    // sequences)
    constexpr auto low_dim_hidden_idss = transform_tuples(
        // convert lower dimension top ids (a sequence) to hidden ids (a sequence)
        [](auto low_dim_top_ids) constexpr {
            return transform_sequences(
                // convert lower dimension top id to hidden id
                [](auto low_dim_top_id) constexpr {
                    return OldChainedTransform::GetTopDimensionIds()[low_dim_top_id];
                },
                low_dim_top_ids);
        },
        NewLowerDimensionOldTopIdss{});

    constexpr index_t num_new_transform = NewTransforms::Size();

    // upper dimension's hidden idss
    constexpr index_t old_hidden_dim_number = OldChainedTransform::GetNumOfHiddenDimension();

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

    // new top dimension's hidden ids
    constexpr auto unordered_new_top_dim_hidden_ids =
        unpack([](auto... xs) constexpr { return merge_sequences(xs...); }, up_dim_hidden_idss);

    constexpr auto new_top_dim_unordered2ordered =
        unpack([](auto... xs) constexpr { return merge_sequences(xs...); },
               NewUpperDimensionNewTopIdss{});

    constexpr auto new_top_dim_hidden_ids =
        unordered_new_top_dim_hidden_ids.ReorderGivenOld2New(new_top_dim_unordered2ordered);

    // put everything together
    const auto all_transforms = container_cat(old_tensor_desc.GetTransforms(), new_transforms);

    constexpr auto all_low_dim_hidden_idss =
        container_cat(OldChainedTransform::GetLowerDimensionIdss(), low_dim_hidden_idss);

    constexpr auto all_up_dim_hidden_idss =
        container_cat(OldChainedTransform::GetUpperDimensionIdss(), up_dim_hidden_idss);

    return ChainedMultiIndexTransform<remove_cv_t<decltype(all_transforms)>,
                                   remove_cv_t<decltype(all_low_dim_hidden_idss)>,
                                   remove_cv_t<decltype(all_up_dim_hidden_idss)>,
                                   remove_cv_t<decltype(OldChainedTransform::GetBottomDimensionIds())>,
                                   remove_cv_t<decltype(new_top_dim_hidden_ids)>>{all_transforms};
}


} // namespace ck
#endif
