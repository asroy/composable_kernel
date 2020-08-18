#ifndef CK_DYNAMIC_TENSOR_DESCRIPTOR_HPP
#define CK_DYNAMIC_TENSOR_DESCRIPTOR_HPP

#include "common_header.hpp"
#include "dynamic_multi_index_transform.hpp"

namespace ck {

template <index_t NDim>
struct DynamicNativeTensorDescriptor
{
    using Index = MultiIndex<NDim>;

    const Index lengths_;
    const Index strides_;

    __host__ __device__ explicit constexpr DynamicNativeTensorDescriptor(const Index& lengths,
                                                                         const Index& strides)
        : lengths_{lengths}, strides_{strides}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension() { return NDim; }

    __host__ __device__ constexpr auto GetLengths() const { return lengths_; }

    __host__ __device__ constexpr auto GetStrides() const { return strides_; }

    __host__ __device__ constexpr index_t GetLength(index_t idim) const { return lengths_[idim]; }

    __host__ __device__ constexpr index_t GetStride(index_t idim) const { return strides_[idim]; }

    __host__ __device__ constexpr index_t GetElementSize() const
    {
        return reduce_on_array(GetLengths(), math::multiplies<index_t>{}, index_t{1});
    }

    __host__ __device__ constexpr index_t GetElementSpace() const
    {
        index_t space = 1;

#pragma unroll
        for(index_t i = 0; i < NDim; ++i)
        {
            space += (GetLength(i) - 1) * GetStride(i);
        }

        return space;
    }

    template <typename Idx>
    __host__ __device__ constexpr index_t CalculateOffset(const Idx& idx) const
    {
        index_t offset = 0;

#pragma unroll
        for(index_t i = 0; i < NDim; ++i)
        {
            offset += idx[i] * GetStride(i);
        }

        return offset;
    }

    template <typename UpIdxDiff, typename UpIdx, typename LowIdx>
    __host__ __device__ constexpr index_t CalculateOffsetDiff(const UpIdxDiff& idx_up_diff,
                                                              const LowIdx& /* idx_low_old */,
                                                              const UpIdx& /* idx_up_old */) const
    {
        return CalculateOffset(idx_up_diff);
    }

    template <typename Idx>
    __host__ __device__ constexpr bool IsUpperIndexValid(const Idx& idx) const
    {
        bool flag = true;

#pragma unroll
        for(index_t i = 0; i < NDim; ++i)
        {
            flag = flag && idx[i] >= 0 && idx[i] < GetLength(i);
        }

        return flag;
    }
};

template <typename LowTensorDescriptor, // DynamicNativeTensorDescriptor or
                                        // DynamicTransformedTensorDescriptor
          typename Transforms,          // Tuple<MultIndexTransforms...>
          typename LowDimensionIds,     // Tuple<Sequence<...>>
          typename UpDimensionIds>      // Tuple<Sequence<...>>
struct DynamicTransformedTensorDescriptor
{
    const LowTensorDescriptor low_tensor_desc_;
    const Transforms transforms_;
    static constexpr index_t NTransform = Transforms::Size();

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension()
    {
        return LowTensorDescriptor::GetNumOfDimension();
    }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension()
    {
        index_t ndim_up = 0;

        static_for<0, NTransform, 1>{}([&](auto i) constexpr {
            constexpr auto tmp = UpDimensionIds{}.At(i);
            ndim_up += decltype(tmp)::Size();
        });

        return ndim_up;
    }

    static constexpr index_t NDimUp  = GetNumOfUpperDimension();
    static constexpr index_t NDimLow = GetNumOfLowerDimension();

    using UpperIndex = MultiIndex<NDimUp>;
    using LowerIndex = MultiIndex<NDimLow>;

    struct lambda_merge_sequences
    {
        template <typename... Xs>
        __host__ __device__ constexpr auto operator()(Xs... xs) const
        {
            return merge_sequences(xs...);
        }
    };

    struct lambda_merge_arrays
    {
        template <typename... Xs>
        __host__ __device__ constexpr auto operator()(Xs... xs) const
        {
            return merge_arrays(xs...);
        }
    };

    __host__ __device__ explicit constexpr DynamicTransformedTensorDescriptor(
        const LowTensorDescriptor& low_tensor_desc, const Transforms& transforms)
        : low_tensor_desc_{low_tensor_desc}, transforms_{transforms}
    {
        static_assert(NTransform == Transforms::Size() && NTransform == LowDimensionIds::Size() &&
                          NTransform == UpDimensionIds::Size(),
                      "wrong! # of transformations not the same");

        // sanity check:
        //   LowDimensionIds should include all low-dimensions,
        //   UpDimensionIds should include all up-dimensions
        using unsorted_up_dimension_ids =
            decltype(unpack(lambda_merge_sequences{}, UpDimensionIds{}));

        using sorted_up_dimension_ids =
            typename sequence_sort<unsorted_up_dimension_ids, math::less<index_t>>::type;

        static_assert(sorted_up_dimension_ids::Size() == NDimUp &&
                          is_valid_sequence_map<sorted_up_dimension_ids>{},
                      "wrong! UpDimensionIds is not configured correctly");

        using unsorted_low_dimension_ids =
            decltype(unpack(lambda_merge_sequences{}, LowDimensionIds{}));

        using sorted_low_dimension_ids =
            typename sequence_sort<unsorted_low_dimension_ids, math::less<index_t>>::type;

        static_assert(sorted_low_dimension_ids::Size() == NDimLow &&
                          is_valid_sequence_map<sorted_low_dimension_ids>{},
                      "wrong! LowDimensionIds is not configured correctly");

        // TODO: sanity check: while a up-dimension could be associated with
        // multille
        //   transformation, a low-dimension should be associated with only one
        //   transformation

        // TODO: sanity-check: GetLowerLengths of each transform should be
        // consistent with lengths
        //   of lower-tensor-descriptor
    }

    __host__ __device__ static constexpr auto GetNumOfDimension()
    {
        return GetNumOfUpperDimension();
    }

    __host__ __device__ constexpr auto GetUpperLengths() const
    {
        // sort upper-dimension-ids
        constexpr auto unsorted_up_dimension_ids =
            unpack(lambda_merge_sequences{}, UpDimensionIds{});

        using sort_up_dimension_ids = sequence_unique_sort<decltype(unsorted_up_dimension_ids),
                                                           math::less<index_t>,
                                                           math::equal<index_t>>;

        constexpr auto sorted2unsorted_map = typename sort_up_dimension_ids::sorted2unsorted_map{};

        // sort upper-lengths
        const auto tuple_of_up_lengths =
            transform_tuples([](const auto& tran) constexpr { return tran.GetUpperLengths(); },
                             transforms_);

        const auto unsorted_up_lengths = unpack(lambda_merge_arrays{}, tuple_of_up_lengths);

        const auto sorted_up_lengths =
            reorder_array_given_new2old(unsorted_up_lengths, sorted2unsorted_map);

        return sorted_up_lengths;
    }

    __host__ __device__ constexpr auto GetLengths() const { return GetUpperLengths(); }

    __host__ __device__ constexpr index_t GetLength(index_t idim) const
    {
        return GetLengths()[idim];
    }

    __host__ __device__ constexpr index_t GetElementSize() const
    {
        return reduce_on_array(GetLengths(), math::multiplies<index_t>{}, index_t{1});
    }

    __host__ __device__ constexpr index_t GetElementSpace() const
    {
        return low_tensor_desc_.GetElementSpace();
    }

    __host__ __device__ constexpr auto GetLowerTensorDescriptor() const { return low_tensor_desc_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up) const
    {
        static_for<0, NTransform, 1>{}([&](auto itran) constexpr {
            const auto tran = transforms_.At(itran);

            const auto idx_up_part = pick_array_element(idx_up, UpDimensionIds{}.At(itran));
            auto idx_low_part      = pick_array_element(idx_low, LowDimensionIds{}.At(itran));

            tran.CalculateLowerIndex(idx_low_part, idx_up_part);
        });
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ void CalculateLowerIndexDiff(LowIdxDiff& idx_low_diff,
                                                     const UpIdxDiff& idx_up_diff,
                                                     const LowIdx& idx_low_old,
                                                     const UpIdx& idx_up_old) const
    {
        static_for<0, NTransform, 1>{}([&](auto itran) {
            const auto tran = transforms_.At(itran);

            const auto idx_up_diff_part =
                pick_array_element(idx_up_diff, UpDimensionIds{}.At(itran));

            const auto idx_up_old_part = pick_array_element(idx_up_old, UpDimensionIds{}.At(itran));

            const auto idx_low_old_part =
                pick_array_element(idx_low_old, LowDimensionIds{}.At(itran));

            auto idx_low_diff_part = pick_array_element(idx_low_diff, LowDimensionIds{}.At(itran));

            tran.CalculateLowerIndexDiff(
                idx_low_diff_part, idx_up_diff_part, idx_low_old_part, idx_up_old_part);
        });
    }

    template <typename UpIdx>
    __host__ __device__ constexpr auto CalculateLowerIndex(const UpIdx& idx_up) const
    {
        LowerIndex idx_low;

        CalculateLowerIndex(idx_low, idx_up);

        return idx_low;
    }

    template <typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr auto CalculateLowerIndexDiff(const UpIdxDiff& idx_up_diff,
                                                               const LowIdx& idx_low_old,
                                                               const UpIdx& idx_up_old) const
    {
        LowerIndex idx_low_diff;

        CalculateLowerIndex(idx_low_diff, idx_up_diff, idx_low_old, idx_up_old);

        return idx_low_diff;
    }

    __host__ __device__ constexpr index_t CalculateOffset(const UpperIndex& idx_up) const
    {
        return low_tensor_desc_.CalculateOffset(CalculateLowerIndex(idx_up));
    }

    __host__ __device__ constexpr bool IsUpperIndexValid(const UpperIndex& idx_up) const
    {
        bool flag = true;

#pragma unroll
        for(index_t i = 0; i < NDimUp; ++i)
        {
            flag = flag && idx_up[i] >= 0 && idx_up[i] < GetLength(i);
        }

        return flag;
    }

    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpperIndex& idx_up) const
    {
        bool flag = true;

        static_for<0, NTransform, 1>{}([&](auto itran) {
            const auto tran = Transforms{}.At(itran);

            // check a indtransformation if it does not always has a valid mapping
            constexpr bool is_valid_up_always_mapped_to_valid_low =
                decltype(tran)::IsValidUpperIndexAlwaysMappedToValidLowerIndex();

            if constexpr(!is_valid_up_always_mapped_to_valid_low)
            {
                const auto up_dims_part = UpDimensionIds{}.At(itran);
                const auto idx_up_part  = pick_array_element(idx_up, up_dims_part);

                flag = flag && IsValidUpperIndexMappedToValidLowerIndex(idx_up_part);
            }
        });

        return flag;
    }
};

} // namespace ck
#endif
