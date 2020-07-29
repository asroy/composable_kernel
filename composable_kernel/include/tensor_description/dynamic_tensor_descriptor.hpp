#ifndef CK_DYNAMIC_TENSOR_DESCRIPTOR_HPP
#define CK_DYNAMIC_TENSOR_DESCRIPTOR_HPP

#include "common_header.hpp"

namespace ck {

template <index_t NDim>
struct DynamicNativeTensorDescriptor
{
    using Index = MultiIndex<NDim>;

    Array<index_t, NDim> lengths_;
    Array<index_t, NDim> strides_;
    index_t element_size_;
    index_t element_space_;

    template <typename Lengths, typename Strides>
    __host__ __device__ constexpr DynamicNativeTensorDescriptor(const Lengths& lengths,
                                                                const Strides& strides)
        : lengths_(lengths), strides_(strides)
    {
        element_size_ = 1;

        for(index_t i = 0; i < NDim; ++i)
        {
            element_size_ *= lengths_[i];
        }

        element_space_ = 1;

        for(index_t i = 0; i < NDim; ++i)
        {
            element_space_ += (lengths_[i] - 1) * strides_[i];
        }
    }

    __host__ __device__ static constexpr auto GetNumOfDimension() { return NDim; }
    __host__ __device__ constexpr auto GetLength(const index_t& i) const { return lengths_[i]; }
    __host__ __device__ constexpr auto GetStride(const index_t& i) const { return strides_[i]; }
    __host__ __device__ constexpr auto GetLengths() const { return lengths_; }
    __host__ __device__ constexpr auto GetStrides() const { return strides_; }
    __host__ __device__ constexpr auto GetElementSize() const { return element_size_; }
    __host__ __device__ constexpr auto GetElementSpace() const { return element_space_; }

    __host__ __device__ constexpr auto CalculateOffset(const Index& idx) const
    {
        index_t offset = 0;

#pragma unroll
        for(index_t i = 0; i < NDim; ++i)
        {
            offset += idx[i] * strides_[i];
        }

        return offset;
    }

    __host__ __device__ constexpr auto CalculateOffsetDiff(const Index& idx_diff) const
    {
        index_t offset_diff = 0;

#pragma unroll
        for(index_t i = 0; i < NDim; ++i)
        {
            offset_diff += idx_diff[i] * strides_[i];
        }

        return offset_diff;
    }

    __host__ __device__ constexpr bool IsUpperIndexValid(const Index& idx) const
    {
        bool flag = true;

#pragma unroll
        for(index_t i = 0; i < NDim; ++i)
        {
            flag = flag && idx[i] >= 0 && idx[i] < lengths_[i];
        }

        return flag;
    }
};

#if 0
// Tensor descriptor for "transformed tensor"
template <typename LowTensorDescriptor,
          typename Transforms,          // Tuple<DynamicMultIndexTransforms,...>
          typename LowDimensions,       // Tuple<Sequence<...>,...>
          typename UpDimensions>        // Tuple<Sequence<...>,...>
struct DynamicTransformedTensorDescriptor
{
    using Type = DynamicTransformedTensorDescriptor;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension()
    {
        // Here, we assume all lower-dimensions are active
        // TODO: sanity-check all lower-dimension are indeed active

        using duplicated_low_active_dims =
            decltype(unpack(lambda_merge_sequences{}, LowDimensions{}));

        using low_active_dims = typename sequence_unique_sort<duplicated_low_active_dims,
                                                              math::less<index_t>,
                                                              math::equal<index_t>>::type;

        return low_active_dims::Size();
    }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension()
    {
        using duplicated_up_active_dims =
            decltype(unpack(lambda_merge_sequences{}, UpDimensions{}));

        using up_active_dims = typename sequence_unique_sort<duplicated_up_active_dims,
                                                             math::less<index_t>,
                                                             math::equal<index_t>>::type;

        return up_active_dims::Size();
    }

    static constexpr index_t ndim_up_  = GetNumOfUpperDimension();
    static constexpr index_t ndim_low_ = GetNumOfLowerDimension();
    static constexpr index_t num_transform_ = Transforms::Size();

    using UpperIndex = MultiIndex<ndim_up_>;
    using LowerIndex = MultiIndex<ndim_low_>;

    const LowTensorDescriptor low_tensor_desc_;
    const Transforms transforms_;
    const LowDimensions low_dims_;
    const UpDimensions up_dims_;

    __host__ __device__ constexpr TransformedTensorDescriptor(const LowTensorDescriptor& low_tensor_desc,
                                                              const Transforms& transforms)
        : low_tensor_desc_(low_tensor_desc),
          transforms_(transforms)
    {
    }

    __host__ __device__ static constexpr auto GetNumOfDimension()
    {
        return GetNumOfUpperDimension();
    }

    __host__ __device__ constexpr auto GetLowerTensorDescriptor() const
    {
        return low_dims_;
    }

    __host__ __device__ constexpr auto GetUpperLengths() cons
    {
    }

    __host__ __device__ constexpr auto GetLengths() const { return GetUpperLengths(); }

    __host__ __device__ constexpr auto GetLength(index_t i) const
    {
        return GetLengths()[i];
    }

    __host__ __device__ constexpr auto GetElementSize() const
    {
        index_t element_size = 1;

        for(index_t i = 0; i < ndim_up_; ++i)
        {
            element_size *= GetLength(i);
        }

        return element_size;
    }

    __host__ __device__ constexpr auto GetElementSpace() const
    {
        return lower_tensor_desc_.GetElementSpace();
    }

    // TODO: right now return value is not constexpr because use of non-constexpr lambda
    __host__ __device__ constexpr LowerIndex CalculateLowerIndex(const UpperIndex& idx_up) const
    {
        LowerIndex idx_low;

        static_for<0, num_transform_, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms{}.At(itran);

            const auto idx_up_part = pick_array_element(idx_up, UpDimensions{}.At(itran));
            auto idx_low_part      = pick_array_element(idx_low, LowDimensions{}.At(itran));

            // this assume each lower (single) index is only assocaited with one transformation,
            //   which is required for index transformation, and has been checked during constructor
            //   of TransformedTensorDescriptor
            idx_low_part = tran.CalculateLowerIndex(to_array(idx_up_part));
        });

        return idx_low;
    }

    // TODO: right now return value is not constexpr because use of non-constepxr lambda
    __host__ __device__ static constexpr LowerIndex CalculateLowerIndexDiff(
        const UpperIndex& idx_up_diff, const UpperIndex& idx_up_old, const LowerIndex& idx_low_old)
    {
        LowerIndex idx_low_diff;

        static_for<0, nTransform, 1>{}([&](auto itran) {
            constexpr auto tran = Transforms{}.At(itran);

            const auto idx_up_diff_part =
                pick_array_element(idx_up_diff, UpDimensions{}.At(itran));

            const auto idx_up_old_part = pick_array_element(idx_up_old, UpDimensions{}.At(itran));

            const auto idx_low_old_part =
                pick_array_element(idx_low_old, LowDimensions{}.At(itran));

            auto idx_low_diff_part = pick_array_element(idx_low_diff, LowDimensions{}.At(itran));

            // this assume each lower (single) index is associated with only one transformation,
            //   which is required for index transformation, and has been checked during constructor
            //   of TransformedTensorDescriptor
            idx_low_diff_part = tran.CalculateLowerIndexDiff(
                to_array(idx_up_diff_part), to_array(idx_up_old_part), to_array(idx_low_old_part));
        });

        return idx_low_diff;
    }

    __host__ __device__ static constexpr index_t CalculateOffset(const UpperIndex& idx_up)
    {
        return GetLowerTensorDescriptor().CalculateOffset(CalculateLowerIndex(idx_up));
    }
};
#endif

} // namespace ck
#endif
