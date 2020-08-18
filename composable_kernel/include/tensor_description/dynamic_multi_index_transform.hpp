#ifndef CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP
#define CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP

#include "common_header.hpp"

namespace ck {

struct DynamicPassThrough
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    const index_t up_length_;

    __host__ __device__ explicit constexpr DynamicPassThrough(const index_t& low_length)
        : up_length_{low_length}
    {
    }

    __host__ __device__ explicit constexpr DynamicPassThrough() : up_length_{0} {}

    __host__ __device__ constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr auto GetUpperLengths() const { return UpperIndex{up_length_}; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(0) = idx_up[0];
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void CalculateLowerIndexDiff(LowIdxDiff& idx_low_diff,
                                                            const UpIdxDiff& idx_up_diff,
                                                            const LowIdx& /* idx_low_old */,
                                                            const UpIdx& /* idx_up_old */)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low_diff(0) = idx_up_diff[0];
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }
};

template <bool SkipIsValidCheck = false>
struct DynamicLeftPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    const index_t up_length_;
    const index_t left_pad_;

    __host__ __device__ explicit constexpr DynamicLeftPad(const index_t& low_length,
                                                          const index_t& left_pad)
        : up_length_{low_length + left_pad}, left_pad_{left_pad}
    {
    }

    __host__ __device__ explicit constexpr DynamicLeftPad() : up_length_{0}, left_pad_{0} {}

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr auto GetUpperLengths() const { return UpperIndex{up_length_}; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(0) = idx_up[0] - left_pad_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void CalculateLowerIndexDiff(LowIdxDiff& idx_low_diff,
                                                            const UpIdxDiff& idx_up_diff,
                                                            const LowIdx& /* idx_low_old */,
                                                            const UpIdx& /* idx_up_old */)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low_diff(0) = idx_up_diff[0];
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[0] >= left_pad_);
    }
};

template <bool SkipIsValidCheck = false>
struct DynamicRightPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    const index_t up_length_;
    const index_t low_length_;
    const index_t right_pad_;

    __host__ __device__ explicit constexpr DynamicRightPad(const index_t& low_length,
                                                           const index_t& right_pad)
        : up_length_{low_length + right_pad}, low_length_{low_length}, right_pad_{right_pad}
    {
    }

    __host__ __device__ explicit constexpr DynamicRightPad()
        : up_length_{0}, low_length_{0}, right_pad_{0}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr auto GetUpperLengths() const { return UpperIndex{up_length_}; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(0) = idx_up[0];
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void CalculateLowerIndexDiff(LowIdxDiff& idx_low_diff,
                                                            const UpIdxDiff& idx_up_diff,
                                                            const LowIdx& /* idx_low_old */,
                                                            const UpIdx& /* idx_up_old */)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low_diff(0) = idx_up_diff[0];
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[0] < low_length_);
    }
};

// idx_low = coefficients[0, ...nDimUp-1] * idx_up[0, ...nDimUp-1] + coefficients[nDimUp]
template <index_t NDimUp>
struct DynamicEmbed
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    const UpperIndex up_lengths_;
    const Array<index_t, NDimUp + 1> coefficients_;

    __host__
        __device__ explicit constexpr DynamicEmbed(const UpperIndex& up_lengths,
                                                   const Array<index_t, NDimUp + 1>& coefficients)
        : up_lengths_{up_lengths}, coefficients_{coefficients}
    {
        static_assert(UpperIndex::Size() == NDimUp, "wrong! # of dimensions not consistent");
    }

    __host__ __device__ explicit constexpr DynamicEmbed()
        : up_lengths_{make_zero_array<index_t, NDimUp>()},
          coefficients_{make_zero_array<index_t, NDimUp + 1>()}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimUp; }

    __host__ __device__ constexpr auto GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_low(0) = coefficients_[NDimUp];

#pragma unroll
        for(index_t i = 0; i < NDimUp; ++i)
        {
            idx_low(0) += idx_up[i] * coefficients_[i];
        }
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ void CalculateLowerIndexDiff(LowIdxDiff& idx_low_diff,
                                                     const UpIdxDiff& idx_up_diff,
                                                     const LowIdx& /* idx_low_old */,
                                                     const UpIdx& /* idx_up_old */) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == NDimUp &&
                          LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_low_diff(0) = 0;

#pragma unroll
        for(index_t i = 0; i < NDimUp; ++i)
        {
            idx_low_diff(0) += idx_up_diff[i] * coefficients_[i];
        }
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }
};

template <index_t NDimLow>
struct DynamicMerge
{
    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    const LowerIndex low_lengths_;
    const LowerIndex low_lengths_scan_;
    const index_t up_length_;

    __host__ __device__ explicit constexpr DynamicMerge(const LowerIndex& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{reverse_inclusive_scan_on_array(
              low_lengths, math::multiplies<index_t>{}, index_t{1})},
          up_length_{reduce_on_array(low_lengths, math::multiplies<index_t>(), 1)}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ explicit constexpr DynamicMerge()
        : low_lengths_{make_zero_array<index_t, NDimLow>()},
          low_lengths_scan_{make_zero_array<index_t, NDimLow>()},
          up_length_{0}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr auto GetUpperLengths() const { return UpperIndex{up_length_}; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t itmp = idx_up[0];

#pragma unroll
        for(index_t i; i < NDimLow - 1; ++i)
        {
            idx_low(i) = itmp / low_lengths_scan_[i];
            itmp -= idx_low[i] * low_lengths_scan_[i];
        }

        idx_low(NDimLow - 1) = itmp;
    }

    // idx_low_diff depends on idx_low_old, so idx_low need to be up-to-date
    // If idx_up_diff is known at compile-time, many calculations can be optimized
    // away by compiler
    // This function assume idx_low_old is not out-of-bound
    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ void CalculateLowerIndexDiff(LowIdxDiff& idx_low_diff,
                                                     const UpIdxDiff& idx_up_diff,
                                                     const LowIdx& idx_low_old,
                                                     const UpIdx& /* idx_up_old */) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        // CalculateLowerIndex(idx_low_diff_const) has multiple integer divisions.
        // However,
        //   1) If idx_up_diff is known at compile-time, then idx_low_diff_const
        //   can be calculated at compile-time.
        //   2) If idx_up_diff is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then idx_low_diff_const also
        //   doesn't change during the whole kernel execution. Compiler generated ISA should
        //   only caclculate idx_low_diff_const once and save it durinng the whole kernel execution
        // If neither 1) nor 2) is satisfied, then the calculation will also be computed at
        //   run-time each time this function is called, and can be very expensive.
        LowerIndex idx_low_diff_const = CalculateLowerIndex(idx_up_diff);

        // do carry check on each low dimension in reversed order
        // do not need to check the first dimension
        index_t carry = 0;

#pragma unroll
        for(index_t i = NDimLow - 1; i > 1; --i)
        {
            // this should be saved as well
            index_t idx_low_length_minus_idx_low_diff_const =
                low_lengths_[i] - idx_low_diff_const[i];
#if 0
            index_t idx_low_length_plus_idx_low_diff_const =
                low_lengths_[i] + idx_low_diff_const[i];
#endif

            index_t idx_low_tmp[i] = idx_low_old[i] + carry;

            bool do_carry = idx_low_tmp[i] >= idx_low_length_minus_idx_low_diff_const;
#if 0
            bool do_borrow = idx_low_tmp[i] < -idx_low_diff_const[i];
#endif

            idx_low_diff(i) =
                do_carry ? -idx_low_length_minus_idx_low_diff_const : idx_low_diff_const;
#if 0
            idx_low_diff(i) =
                do_borrow ? idx_low_length_plus_idx_low_diff_const : idx_low_diff[i];
#endif

            idx_low_diff(i) += carry;

            carry = do_carry ? 1 : 0;
#if 0
            carry = do_borrow ? -1 : carry;
#endif
        }

        idx_low_diff(0) = idx_low_diff_const[0] + carry;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }
};

template <index_t NDimUp>
struct DynamicUnMerge
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    const UpperIndex up_lengths_;
    const UpperIndex up_lengths_scan_;

    __host__ __device__ explicit constexpr DynamicUnMerge(const UpperIndex& up_lengths)
        : up_lengths_{up_lengths},
          up_lengths_scan_{
              reverse_exclusive_scan_on_array(up_lengths, math::multiplies<index_t>(), index_t{1})}
    {
    }

    __host__ __device__ explicit constexpr DynamicUnMerge()
        : up_lengths_{make_zero_array<index_t, NDimUp>()},
          up_lengths_scan_{make_zero_array<index_t, NDimUp>()}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    __host__ __device__ constexpr auto GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up) const
    {
        idx_low(0) = idx_up[NDimUp];

#pragma unroll
        for(index_t i = 0; i < NDimUp - 1; ++i)
        {
            idx_low(0) += idx_up[i] * up_lengths_scan_[i];
        }
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ void CalculateLowerIndexDiff(LowIdxDiff& idx_low_diff,
                                                     const UpIdxDiff& idx_up_diff,
                                                     const LowIdx& /* idx_low_old */,
                                                     const UpIdx& /* idx_up_old */) const
    {
        CalculateLowerIndex(idx_low_diff, idx_up_diff);
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }
};

struct DynamicFreeze
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<0>;

    const index_t low_idx_;

    __host__ __device__ explicit constexpr DynamicFreeze(const index_t& low_idx) : low_idx_{low_idx}
    {
    }

    __host__ __device__ explicit constexpr DynamicFreeze() : low_idx_{0} {}

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 0; }

    __host__ __device__ constexpr auto GetUpperLengths() const { return UpperIndex{}; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(0) = low_idx_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void CalculateLowerIndexDiff(LowIdxDiff& idx_low_diff,
                                                            const UpIdxDiff& idx_up_diff,
                                                            const LowIdx& /* idx_low_old */,
                                                            const UpIdx& /* idx_up_old */)
    {
        idx_low_diff(0) = index_t{0};
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }
};

} // namespace ck
#endif
