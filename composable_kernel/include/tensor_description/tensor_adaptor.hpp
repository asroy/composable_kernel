#ifndef CK_TENSOR_ADAPTOR_HPP
#define CK_TENSOR_ADAPTOR_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

// Transforms: Tuple<transforms...>
// LowerDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// UpperDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// BottomDimensionHiddenIds : Sequence<...>
// TopDimensionHiddenIds : Sequence<...>
template <typename Transforms,
          typename LowerDimensionHiddenIdss,
          typename UpperDimensionHiddenIdss,
          typename BottomDimensionHiddenIds,
          typename TopDimensionHiddenIds>
struct TensorAdaptor
{
    private:
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

        constexpr index_t idim_hidden = TopDimensionHiddenIds::At(idim_top);

        index_t itran_found   = 0;
        index_t idim_up_found = 0;
        bool found            = false;

        static_for<0, ntransform_, 1>{}([&](auto itran) {
            constexpr auto up_dim_ids = UpperDimensionHiddenIdss{}[itran];

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

    __host__ __device__ static constexpr index_t GetNumOfHiddenDimension()
    {
        constexpr auto all_low_dim_ids =
            unpack([](auto&&... xs) constexpr { return merge_sequences(xs...); },
                   LowerDimensionHiddenIdss{});

        constexpr auto all_up_dim_ids =
            unpack([](auto&&... xs) constexpr { return merge_sequences(xs...); },
                   UpperDimensionHiddenIdss{});

        constexpr auto all_dim_ids = merge_sequences(all_low_dim_ids, all_up_dim_ids);

        using unique_sort_all_dim_ids = typename sequence_unique_sort<decltype(all_dim_ids),
                                                                      math::less<index_t>,
                                                                      math::equal<index_t>>::type;

        return unique_sort_all_dim_ids::Size();
    }

    constexpr static index_t ntransform_  = GetNumOfTransform();
    constexpr static index_t ndim_hidden_ = GetNumOfHiddenDimension();
    constexpr static index_t ndim_bottom_ = BottomDimensionHiddenIds::Size();
    constexpr static index_t ndim_top_    = TopDimensionHiddenIds::Size();

    using HiddenIndex = MultiIndex<ndim_hidden_>;
    using BottomIndex = MultiIndex<ndim_bottom_>;
    using TopIndex    = MultiIndex<ndim_top_>;

    // may be index_t or Number<>
    using ElementSize = remove_cv_t<decltype(InitializeElementSize(Transforms{}))>;

    public:
    __host__ __device__ constexpr TensorAdaptor() = default;

    __host__ __device__ constexpr TensorAdaptor(const Transforms& transforms)
        : transforms_{transforms}, element_size_{InitializeElementSize(transforms)}
    {
        static_assert(Transforms::Size() == ntransform_ &&
                          LowerDimensionHiddenIdss::Size() == ntransform_ &&
                          UpperDimensionHiddenIdss::Size() == ntransform_,
                      "wrong! inconsistent # of transformations");

        // TODO check dependency of dimensions is valid
    }

    __host__ __device__ constexpr auto GetElementSize() const { return element_size_; }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        static_assert(TopIdx::Size() == TopDimensionHiddenIds::Size(),
                      "wrong! # of dimension inconsistent");

        constexpr index_t ntransform  = GetNumOfTransform();
        constexpr index_t ndim_hidden = GetNumOfHiddenDimension();

        MultiIndex<ndim_hidden> idx_hidden;

        // initialize uppest index
        set_container_subset(idx_hidden, GetTopDimensionHiddenIds(), idx_top);

        // calculate hidden index
        static_for<ntransform, 0, -1>{}([&](auto itran_p1) {
            auto itran              = itran_p1 - Number<1>{};
            const auto& tran        = GetTransforms().At(itran);
            constexpr auto dims_low = GetLowerDimensionHiddenIdss().At(itran);
            constexpr auto dims_up  = GetUpperDimensionHiddenIdss().At(itran);

            const auto idx_up = get_container_subset(idx_hidden, dims_up);

            MultiIndex<dims_low.Size()> idx_low;

            tran.CalculateLowerIndex(idx_low, idx_up);

            set_container_subset(idx_hidden, dims_low, idx_low);
        });

        return get_container_subset(idx_hidden, BottomDimensionHiddenIds{});
    }

    private:
    Transforms transforms_;
    ElementSize element_size_;
};

template <typename TensorAdaptor0, typename TensorAdaptor1>
__host__ __device__ constexpr auto chain_tensor_adaptors(const TensorAdaptor0& adaptor0,
                                                         const TensorAdaptor1& adaptor1)
{
    static_assert(TensorAdaptor0::GetNumOfTopDimension() ==
                      TensorAdaptor1::GetNumOfBottomDimension(),
                  "wrong!");

    // all_transforms = transform0 + transform1
    const auto all_transforms =
        container_concat(adaptor0.GetTransforms(), adaptor1.GetTransforms());

    // shift
    constexpr index_t adaptor0_max_hidden_id = [&]() {
        index_t adaptor0_max_hidden_id = NumericalMinValue<index_t>::value;

        static_for<0, TensorAdaptor0::GetNumOfTransform(), 1>{}([&](auto itran) {
            constexpr index_t ndim_low =
                TensorAdaptor0::GetTransforms()[itran].GetNumOfLowerDimension();

            static_for<0, ndim_low, 1>{}([&](auto idim_low) {
                adaptor0_max_hidden_id =
                    math::max(adaptor0_max_hidden_id,
                              TensorAdaptor0::GetLowerDimensionHiddenIdss()[itran][idim_low]);
            });

            constexpr index_t ndim_up =
                TensorAdaptor0::GetTransforms()[itran].GetNumOfUpperDimension();

            static_for<0, ndim_up, 1>{}([&](auto idim_up) {
                adaptor0_max_hidden_id =
                    math::max(adaptor0_max_hidden_id,
                              TensorAdaptor0::GetUpperDimensionHiddenIdss()[itran][idim_up]);
            });
        });

        return adaptor0_max_hidden_id;
    }();

    constexpr index_t adaptor1_min_hidden_id = [&]() {
        index_t adaptor1_min_hidden_id = NumericalMaxValue<index_t>::value;

        static_for<0, TensorAdaptor0::GetNumOfTransform(), 1>{}([&](auto itran) {
            constexpr index_t ndim_low =
                TensorAdaptor0::GetTransforms()[itran].GetNumOfLowerDimension();

            static_for<0, ndim_low, 1>{}([&](auto idim_low) {
                adaptor1_min_hidden_id =
                    math::min(adaptor0_max_hidden_id,
                              TensorAdaptor0::GetLowerDimensionHiddenIdss()[itran][idim_low]);
            });

            constexpr index_t ndim_up =
                TensorAdaptor0::GetTransforms()[itran].GetNumOfUpperDimension();

            static_for<0, ndim_up, 1>{}([&](auto idim_up) {
                adaptor0_max_hidden_id =
                    math::min(adaptor0_max_hidden_id,
                              TensorAdaptor0::GetUpperDimensionHiddenIdss()[itran][idim_up]);
            });
        });

        return adaptor1_min_hidden_id;
    }();

    constexpr index_t adaptor1_hidden_id_shift =
        adaptor1_min_hidden_id - adaptor0_max_hidden_id + 1;

    // all_low_dim_hidden_idss =
    // low_dim_hidden_idss_0 + shift_hidden_id_for_1(match_hidden_id_for_1(low_dim_hiden_idss_1))
    constexpr auto low_dim_hidden_idss_1 = generate_tuple(
        // generate sequence of ids for a transform
        [&](auto itran) {
            constexpr auto ndim_low_1 =
                TensorAdpator1::GetLowerDimensionsHiddenIdss()[itran].Size();

            constexpr auto low_dim_hidden_ids_1 =
                TensorAdpator1::GetLowerDimensionsHiddenIdss()[itran];

            // sequence in, sequence out
            constexpr auto low_dim_hidden_ids_1_mod = [&]() constexpr
            {
                constexpr auto low_dim_hidden_ids_1_mod = to_multi_index(low_dim_hidden_ids_1);

                // match hidden id
                static_for<0, ndim_low_1, 1>{}([&](auto idim_low_1) {
                    static_for<0, ndim_bottom_1, 1>{}([&](auto idim_bottom_1) {
                        if constexpr(low_dim_hidden_ids_1[idim_low_1] ==
                                     TensorAdaptor1::GetBottomDimensionHiddenIds()[idim_bottom_1])
                        {
                            low_dim_hidden_ids_1_mod(idim_low_1) =
                                TensorAdaptor0::GetTopDimensionHiddenIds()[idim_bottom_1];
                        }
                    });
                });

                // shift hidden id
                static_for<0, ndim_low_1, 1>{}[&](auto idim_low_1)
                {
                    low_dim_hidden_ids_1_mod(idim_low_1) -= adaptor1_hidden_id_shift;
                }

                return generate_sequence([&](auto i) constexpr { return low_dim_hidden_ids_1[i]; },
                                         Number<ndim_low_1>{});
            }
            ();

            return low_dim_hidden_ids_1_mod;
        },
        Number<TensorAdaptor1::GetNumOfTransform()>{});

    constexpr auto all_low_dim_hidden_idss =
        container_concat(TensorAdaptor0::GetLowerDimensionHiddenIdss(), low_dim_hidden_idss_1);

    // all_up_dim_hidden_idss =
    // up_dim_hidden_idss_0 + shift_hidden_id_for_1(up_dim_hiden_idss_1)
    constexpr auto up_dim_hidden_idss_1 = generate_tuple(
        // generate sequence of ids for a transform
        [&](auto itran) {
            constexpr auto ndim_up_1 = TensorAdpator1::GetUpperDimensionsHiddenIdss()[itran].Size();

            constexpr auto up_dim_hidden_ids_1 =
                TensorAdpator1::GetUpperDimensionsHiddenIdss()[itran];

            // sequence in, sequence out
            constexpr auto up_dim_hidden_ids_1_mod = [&]() constexpr
            {
                constexpr auto up_dim_hidden_ids_1_mod = to_multi_index(up_dim_hidden_ids_1);

                // shift hidden id
                static_for<0, ndim_up_1, 1>{}[&](auto idim_up_1)
                {
                    up_dim_hidden_ids_1_mod(idim_up_1) -= adaptor1_hidden_id_shift;
                }

                return generate_sequence(
                    [&](auto i) constexpr { return up_dim_hidden_ids_1_mod[i]; },
                    Number<ndim_up_1>{});
            }
            ();

            return up_dim_hidden_ids_1_mod;
        },
        Number<TensorAdaptor1::GetNumOfTransform()>{});

    constexpr auto all_up_dim_hidden_idss =
        container_concat(TensorAdaptor0::GetUpperDimensionHiddenIdss(), up_dim_hidden_idss_1);

    // bottom_dim_hidden_ids = bottom_dim_hidden_ids_0
    constexpr bottom_dim_hidden_ids = TensorAdaptor0::GetBottomDimensionHiddenIds();

    // top_dim_hidden_ids = shift_hidden_id(top_dim_hidden_ids_1)
    constexpr top_dim_hidden_ids =
        TensorAdaptor1::GetTopDimensionHiddenIds() - Number<adaptor1_hidden_id_shift>{};

    // put everything together
    return TensorAdaptor<decltype(all_transforms),
                         decltype(all_low_dim_hidden_idss),
                         decltype(all_up_dim_hidden_idss),
                         decltype(bottom_dim_hidden_ids),
                         decltype(top_dim_hidden_ids)>{all_transforms};
}

// Transforms: Tuple<transforms...>
// LowerDimensionOldTopIdss: Tuple<Sequence<...>, ...>
// UpperDimensionNewTopIdss: Tuple<Sequence<...>, ...>
template <typename Transforms, typename LowerDimensionOldTopIdss, typename UpperDimensionNewTopIdss>
__host__ __device__ constexpr auto make_simple_tensor_adaptor(const Transforms& transforms,
                                                              LowerDimensionOldTopIdss,
                                                              UpperDimensionNewTopIdss)
{
    // low_dim_hidden_idss
    // up_dim_hidden_idss
    // bottom_dim_hidden_ids
    // top_dim_hidden_ids

    return TensorAdaptor<Transform,
                         decltype(low_dim_hidden_idss),
                         decltype(up_dim_hidden_idss),
                         decltype(bottom_dim_hidden_ids),
                         decltype(top_dim_hidden_ids)>{transforms};
}

} // namespace ck
#endif
