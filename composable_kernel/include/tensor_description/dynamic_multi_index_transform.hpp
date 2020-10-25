#ifndef CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP
#define CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP

#include "common_header.hpp"
#include "multi_index.hpp"

namespace ck {

struct DynamicPassThrough
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    const UpperIndex up_lengths_;

#if 0
    __host__ __device__   constexpr DynamicPassThrough(const DynamicPassThrough&) = default;

    __host__ __device__   constexpr DynamicPassThrough(DynamicPassThrough&&) = default;
#else
    __host__ __device__ constexpr DynamicPassThrough(const DynamicPassThrough& other)
        : up_lengths_{other.up_lengths_}
    {
    }

    __host__ __device__ constexpr DynamicPassThrough(DynamicPassThrough&& other)
        : up_lengths_{other.up_lengths_}
    {
    }
#endif

    __host__ __device__ constexpr DynamicPassThrough(const index_t& low_length)
        : up_lengths_{make_multi_index(low_length)}
    {
    }

    __host__ __device__ constexpr DynamicPassThrough() : up_lengths_{0} {}

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void CalculateLowerIndexDiff(LowIdxDiff& idx_diff_low,
                                                            const UpIdxDiff& idx_diff_up,
                                                            const LowIdx& /* idx_low_old */,
                                                            const UpIdx& /* idx_up_old */)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(Number<0>{}) = idx_diff_up[Number<0>{}];
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

    const UpperIndex up_lengths_;
    const index_t left_pad_;

#if 0
    __host__ __device__   constexpr DynamicLeftPad(const DynamicLeftPad&) = default;

    __host__ __device__   constexpr DynamicLeftPad(DynamicLeftPad&&) = default;
#else
    __host__ __device__ constexpr DynamicLeftPad(const DynamicLeftPad& other)
        : up_lengths_{other.up_lengths_}, left_pad_{other.left_pad_}
    {
    }

    __host__ __device__ constexpr DynamicLeftPad(DynamicLeftPad&& other)
        : up_lengths_{other.up_lengths_}, left_pad_{other.left_pad_}
    {
    }
#endif

    __host__ __device__ constexpr DynamicLeftPad(const index_t& low_length, const index_t& left_pad)
        : up_lengths_{make_multi_index(low_length + left_pad)}, left_pad_{left_pad}
    {
    }

    __host__ __device__ constexpr DynamicLeftPad() : up_lengths_{0}, left_pad_{0} {}

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] - left_pad_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static constexpr void
    CalculateLowerIndexDiff(LowIdxDiff& idx_diff_low,
                            const UpIdxDiff& idx_diff_up,
                            const LowIdx& /* idx_low_old */,
                            const UpIdx& /* idx_up_old */)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(Number<0>{}) = idx_diff_up[Number<0>{}];
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
        return SkipIsValidCheck || (idx_up[Number<0>{}] >= left_pad_);
    }
};

template <bool SkipIsValidCheck = false>
struct DynamicRightPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    const UpperIndex up_lengths_;
    const index_t low_length_;
    const index_t right_pad_;

#if 0
    __host__ __device__   constexpr DynamicRightPad(const DynamicRightPad&) = default;

    __host__ __device__   constexpr DynamicRightPad(DynamicRightPad&&) = default;
#else
    __host__ __device__ constexpr DynamicRightPad(const DynamicRightPad& other)
        : up_lengths_{other.up_lengths_},
          low_length_{other.low_length_},
          right_pad_{other.right_pad_}
    {
    }

    __host__ __device__ constexpr DynamicRightPad(DynamicRightPad&& other)
        : up_lengths_{other.up_lengths_},
          low_length_{other.low_length_},
          right_pad_{other.right_pad_}
    {
    }
#endif

    __host__ __device__ constexpr DynamicRightPad(const index_t& low_length,
                                                  const index_t& right_pad)
        : up_lengths_{make_multi_index(low_length + right_pad)},
          low_length_{low_length},
          right_pad_{right_pad}
    {
    }

    __host__ __device__ constexpr DynamicRightPad() : up_lengths_{0}, low_length_{0}, right_pad_{0}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                                  const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static constexpr void
    CalculateLowerIndexDiff(LowIdxDiff& idx_diff_low,
                            const UpIdxDiff& idx_diff_up,
                            const LowIdx& /* idx_low_old */,
                            const UpIdx& /* idx_up_old */)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(Number<0>{}) = idx_diff_up[Number<0>{}];
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
        return SkipIsValidCheck || (idx_up[Number<0>{}] < low_length_);
    }
};

// idx_low = coefficients[0, ...nDimUp-1] * idx_up[0, ...nDimUp-1]
template <index_t NDimUp>
struct DynamicEmbed
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    const UpperIndex up_lengths_;
    const UpperIndex coefficients_;

#if 0
    __host__ __device__   constexpr DynamicEmbed(const DynamicEmbed&) = default;

    __host__ __device__   constexpr DynamicEmbed(DynamicEmbed&&) = default;
#else
    __host__ __device__ constexpr DynamicEmbed(const DynamicEmbed& other)
        : up_lengths_{other.up_lengths_}, coefficients_{other.coefficients_}
    {
    }

    __host__ __device__ constexpr DynamicEmbed(DynamicEmbed&& other)
        : up_lengths_{other.up_lengths_}, coefficients_{other.coefficients_}
    {
    }
#endif
    __host__ __device__ constexpr DynamicEmbed(const UpperIndex& up_lengths,
                                               const UpperIndex& coefficients)
        : up_lengths_{up_lengths}, coefficients_{coefficients}
    {
        static_assert(UpperIndex::Size() == NDimUp, "wrong! # of dimensions not consistent");
    }

    template <typename UpperLengths, typename Coefficients>
    __host__ __device__ constexpr DynamicEmbed(const UpperLengths& up_lengths,
                                               const Coefficients& coefficients)
        : up_lengths_{up_lengths}, coefficients_{coefficients}
    {
    }

    __host__ __device__ constexpr DynamicEmbed()
        : up_lengths_{make_zero_multi_index<NDimUp>()},
          coefficients_{make_zero_multi_index<NDimUp>()}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}([&idx_low, &idx_up, this](auto i) {
            idx_low(Number<0>{}) += idx_up[i] * this->coefficients_[i];
        });
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndexDiff(LowIdxDiff& idx_diff_low,
                                                               const UpIdxDiff& idx_diff_up,
                                                               const LowIdx& /* idx_low_old */,
                                                               const UpIdx& /* idx_up_old */) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == NDimUp &&
                          LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}(
            [&](auto i) { idx_diff_low(Number<0>{}) += idx_diff_up[i] * coefficients_[i]; });
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
    const UpperIndex up_lengths_;

#if 0
    __host__ __device__   constexpr DynamicMerge(const DynamicMerge&) = default;

    __host__ __device__   constexpr DynamicMerge(DynamicMerge&&) = default;
#else
    __host__ __device__ constexpr DynamicMerge(const DynamicMerge& other)
        : low_lengths_{other.low_lengths_},
          low_lengths_scan_{other.low_lengths_scan_},
          up_lengths_{other.up_lengths_}
    {
    }

    __host__ __device__ constexpr DynamicMerge(DynamicMerge&& other)
        : low_lengths_{other.low_lengths_},
          low_lengths_scan_{other.low_lengths_scan_},
          up_lengths_{other.up_lengths_}
    {
    }
#endif

    __host__ __device__ constexpr DynamicMerge(const LowerIndex& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{container_reverse_exclusive_scan(
              low_lengths, math::multiplies<index_t>{}, index_t{1})},
          up_lengths_{make_multi_index(
              container_reduce(low_lengths, math::multiplies<index_t>(), index_t{1}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ constexpr DynamicMerge()
        : low_lengths_{make_zero_multi_index<NDimLow>()},
          low_lengths_scan_{make_zero_multi_index<NDimLow>()},
          up_lengths_{0}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&idx_low, &tmp, this](auto i) {
            idx_low(i) = tmp / this->low_lengths_scan_[i];
            tmp -= idx_low[i] * this->low_lengths_scan_[i];
        });

        idx_low(Number<NDimLow - 1>{}) = tmp;
    }

    // idx_diff_low depends on idx_low_old, so idx_low need to be up-to-date
    // If idx_diff_up is known at compile-time, many calculations can be optimized
    // away by compiler
    // This function assume idx_low_old is not out-of-bound
    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndexDiff(LowIdxDiff& idx_diff_low,
                                                               const UpIdxDiff& idx_diff_up,
                                                               const LowIdx& idx_low_old,
                                                               const UpIdx& /* idx_up_old */) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

#if 0
        // I only want to do this check, if idx_diff_up is know at compile-time
        if(idx_diff_up[Number<0>{}] == 0)
        {
            static_for<0, NDimLow, 1>{}([&idx_diff_low](auto i){
                idx_diff_low(i) = 0;
            });

            return;
        }
#endif
        // CalculateLowerIndex(idx_diff_low_const) has multiple integer divisions.
        // However,
        //   1) If idx_diff_up is known at compile-time, then idx_diff_low_const
        //   can be calculated at compile-time.
        //   2) If idx_diff_up is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then
        //   idx_diff_low_const also
        //   doesn't change during the whole kernel execution. Compiler generated
        //   ISA should
        //   only caclculate idx_diff_low_const once and save it durinng the whole
        //   kernel execution
        // If neither 1) nor 2) is satisfied, then the calculation will also be
        // computed at
        //   run-time each time this function is called, and can be very expensive.
        LowerIndex idx_diff_low_const;
        CalculateLowerIndex(idx_diff_low_const, idx_diff_up);

        // do carry check on each low dimension in reversed order
        // do not need to check the first dimension
        index_t carry = 0;

        static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
            // this should be saved in SGPR as well
            index_t idx_low_length_minus_idx_diff_low_const =
                low_lengths_[i] - idx_diff_low_const[i];
#if 1
            index_t idx_low_length_plus_idx_diff_low_const =
                low_lengths_[i] + idx_diff_low_const[i];
#endif

            index_t idx_low_tmp = idx_low_old[i] + carry;

            bool do_carry = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const;
#if 1
            bool do_borrow = idx_low_tmp < -idx_diff_low_const[i];
#endif

            idx_diff_low(i) =
                do_carry ? -idx_low_length_minus_idx_diff_low_const : idx_diff_low_const[i];
#if 1
            idx_diff_low(i) = do_borrow ? idx_low_length_plus_idx_diff_low_const : idx_diff_low[i];
#endif

            idx_diff_low(i) += carry;

            carry = do_carry ? 1 : 0;
#if 1
            carry = do_borrow ? -1 : carry;
#endif
        });

        idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;
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

    __host__ __device__ constexpr DynamicUnMerge(const UpperIndex& up_lengths)
        : up_lengths_{up_lengths},
          up_lengths_scan_{
              container_reverse_exclusive_scan(up_lengths, math::multiplies<index_t>(), index_t{1})}
    {
    }

    __host__ __device__ constexpr DynamicUnMerge()
        : up_lengths_{make_zero_multi_index<NDimUp>()},
          up_lengths_scan_{make_zero_multi_index<NDimUp>()}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        idx_low(Number<0>{}) = idx_up[Number<NDimUp - 1>{}];

        static_for<0, NDimUp - 1, 1>{}(
            [&](auto i) { idx_low(Number<0>{}) += idx_up[i] * up_lengths_scan_[i]; });
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndexDiff(LowIdxDiff& idx_diff_low,
                                                               const UpIdxDiff& idx_diff_up,
                                                               const LowIdx& /* idx_low_old */,
                                                               const UpIdx& /* idx_up_old */) const
    {
        CalculateLowerIndex(idx_diff_low, idx_diff_up);
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

    __host__ __device__ constexpr DynamicFreeze(const index_t& low_idx) : low_idx_{low_idx} {}

    __host__ __device__ constexpr DynamicFreeze() : low_idx_{0} {}

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 0; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return UpperIndex{}; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = low_idx_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    __host__ __device__ static void CalculateLowerIndexDiff(LowIdxDiff& idx_diff_low,
                                                            const UpIdxDiff& idx_diff_up,
                                                            const LowIdx& /* idx_low_old */,
                                                            const UpIdx& /* idx_up_old */)
    {
        idx_diff_low(Number<0>{}) = index_t{Number<0>{}};
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
