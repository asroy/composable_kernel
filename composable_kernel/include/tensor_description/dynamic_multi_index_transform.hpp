#ifndef CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP
#define CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP

#include "common_header.hpp"

namespace ck {

struct DynamicPassThrough
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    const index_t low_length_;

    __host__ __device__ explicit constexpr DynamicPassThrough(const index_t& low_length)
        : low_length_(low_length)
    {
    }

    __host__ __device__ constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr auto GetUpperIndex() { return UpperIndex({low_length_}); }

    __host__ __device__ constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        return idx_up;
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return idx_up_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpperIndex& /* idx_up */)
    {
        return true;
    }
};

template <bool SkipIsValidCheck = false>
struct DynamicLeftPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    const index_t low_length_;
    const index_t left_pad_;

    __host__ __device__ explicit constexpr Pad(const index_t& low_length, const index_t& left_pad)
        : low_length_{low_length}, left_pad_{left_pad}
    {
    }

    __host__ __device__ constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr auto GetUpperIndex()
    {
        return UpperIndex({low_length_ + left_pad_});
    }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        return LowerIndex{idx_up - lef_pad_};
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return idx_up_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpperIndex& idx_up)
    {
        return SkipIsValidCheck || (idx_up[0] >= left_pad_);
    }
};

template <bool SkipIsValidCheck = false>
struct DynamicRightPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    const index_t low_length_;
    const index_t right_pad_;

    __host__ __device__ explicit constexpr Pad(const index_t& low_length, const index_t& right_pad)
        : low_length_{low_length}, right_pad_{right_pad}
    {
    }

    __host__ __device__ constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr auto GetUpperIndex()
    {
        return UpperIndex({low_length_ + right_pad_});
    }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        return idx_up;
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return idx_up_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpperIndex& idx_up)
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

    const index_t low_length_;
    const UpperIndex up_lengths_;
    const Array<index_t, NDimUp + 1> coefficients_;

    __host__ __device__ explicit constexpr Embed(const index_t& low_length,
                                                 const UpperIndex& up_lengths,
                                                 const Array<index_t, NDimUp + 1>& coefficients)
        : low_length_(low_length), up_lengths_(up_lengths), coefficients_(coefficients)
    {
        static_assert(up_lengths.GetSize() == nDimUp && coefficients.GetSize() == nDimUp + 1,
                      "wrong! # of dimensions not consistent");
    }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimUp; }

    __host__ __device__ static constexpr auto GetUpperIndex() { return up_lengths_; }

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        index_t idx_low = coefficients_[NDimUp];

        for(index_t i = 0; i < nDimUp; ++i)
        {
            idx_low += idx_up[i] * coefficients_[i];
        }

        return LowerIndex({idx_low});
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        index_t idx_low_diff = 0;

        for(index_t i = 0; i < nDimUp; ++i)
        {
            idx_low_diff += idx_up_diff[i] * Coefficients{}[i];
        }

        return LowerIndex({idx_low_diff});
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpperIndex& /* idx_up */)
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

    __host__ __device__ explicit constexpr DynamicMerge(const LowerIndex& low_lengths_)
        : low_lengths_(low_lengths), 
          low_lengths_scan_(reverse_inclusive_scan_on_array(low_lengths, multiplies<index_t>()),
          up_length(accumulate_on_array(low_lengths, multiplies<index_t>(), 1))
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() {
        return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() {
        return 1; }

    __host__ __device__ constexpr auto GetUpperIndex() const
    {
        return UpperIndex({up_length_});
    }

    __host__ __device__ constexpr auto CalculateLowerIndex(const UpperIndex& idx_up) const
    {
        LowerIndex idx_low;

        index_t itmp = idx_up[0];

#pragma unroll
        for(index_t i; i < NDimLow - 1; ++i)
        {
            idx_low(i) = itmp / low_lengths_scan_[i];
            itmp -= idx_low[i] * low_lengths_scan_[i];
        }

        idx_low(NDimLow - 1) = itmp;
#pragma unroll

        return idx_low;
    }

    // idx_low_diff depends on idx_low_old, so idx_low need to be up-to-date
    // If idx_up_diff is known at compile-time, many calculations can be optimized
    // away by compiler
    // This function assume idx_low_old is not out-of-bound
    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& idx_low_old)
    {
        LowerIndex idx_low_diff;

        // CalculateLowerIndex(idx_up_diff) has multiple integer divisions.
        //   1) If idx_up_diff is known at compile-time, then idx_low_diff_const
        //   can be calculated at compile-time.
        //   2) If idx_up_diff is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then idx_low_diff_const also
        //   doesn't change during the whole kernel execution. Compiler generated ISA should
        //   only caclculate idx_low_diff once and save it durinng the whole kernel execution
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

            bool do_carry = idx_low_tmp >= idx_low_length_minus_idx_low_diff_const;
#if 0
            bool do_borrow = idx_low_tmp < -idx_low_diff_const[i];
#endif

            idx_low_diff[i] =
                do_carry ? -idx_low_length_minus_idx_low_diff_const : idx_low_diff_const;
#if 0
            idx_low_diff[i] =
                do_borrow ? idx_low_length_plus_idx_low_diff_const : idx_low_diff[i];
#endif

            idx_low_diff[i] += carry;

            carry = do_carry ? 1 : 0;
#if 0
            carry = do_borrow ? -1 : carry;
#endif
        }

        idx_low_diff[0] = idx_low_diff_const[0] + carry;

        return idx_low_diff;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() {
        return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpperIndex& /* idx_up */)
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
        : up_lengths_(up_lengths), 
          up_lengths_scan_(reverse_exclusive_scan_on_array(up_lengths, multiplies<index_t>(), index_t(1))
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() {
        return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() {
        return NDimUp; }

    __host__ __device__ constexpr auto GetUpperIndex() const {
        return up_lengths_; }

    __host__ __device__ constexpr auto CalculateLowerIndex(const UpperIndex& idx_up) const
    {
        index_t idx_low = idx_up[NDimUp];

#pragma unroll
        for(index_t i = 0; i < NDimUp - 1; ++i)
        {
            idx_low += idx_up[i] * up_lengths_scan_[i];
        }

        return LowerIndex{idx_low};
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& idx_up_diff,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return CalculateLowerIndex(idx_up_diff);
    }

    __host__ __device__ static constexpr bool IsLinearTransform() {
        return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpperIndex& /* idx_up */)
    {
        return true;
    }
};

struct DynamicFreeze
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<>;

    const index_t low_idx_;
    const index_t low_length_;

    __host__ __device__ explicit constexpr Freeze(const index_t& low_idx, const index_t& low_length)
        : low_idx_(low_idx), low_length_(low_length)
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 0; }

    __host__ __device__ static constexpr auto GetUpperIndex() { return UpperIndex(); }

    __host__ __device__ constexpr auto CalculateLowerIndex(const UpperIndex& /*idx_up*/) const
    {
        return LowerIndex({low_length_});
    }

    __host__ __device__ static constexpr auto
    CalculateLowerIndexDiff(const UpperIndex& /* idx_up_diff */,
                            const UpperIndex& /* idx_up_old */,
                            const LowerIndex& /* idx_low_old */)
    {
        return LowerIndex({0});
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpperIndex& /* idx_up */)
    {
        return true;
    }
};

} // namespace ck
#endif
