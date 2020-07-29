#ifndef CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP
#define CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP

#include "common_header.hpp"

namespace ck {

struct DynamicPassThrough
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    index_t low_length_;

    __host__ __device__ constexpr DynamicPassThrough(index_t low_length) : low_length_(low_length)
    {
    }

    __host__ __device__ static constexpr auto GetNumOfLowerDimension() { return Number<1>{}; }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension() { return Number<1>{}; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return Sequence<Length>{}; }

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
        return true;
    }
};

template <index_t NDimLow>
struct DynamicMerge
{
    static constexpr index_t ndim_low_ = NDimLow static constexpr index_t ndim_up_ = 1;

    using LowerIndex = MultiIndex<ndim_low_>;
    using UpperIndex = MultiIndex<ndum_up_>;

    Array<index_t, NDimLow> low_lengths_;
    index_t up_length_;

    __host__ __device__ static constexpr auto GetNumOfLowerDimension()
    {
        return Number<ndim_low_>{};
    }

    __host__ __device__ static constexpr auto GetNumOfUpperDimension()
    {
        return Number<ndim_up_>{};
    }

    __host__ __device__ static constexpr auto GetUpperLengths()
    {
        return Array<index_t, 1> up_length_;
    }

    // emulate constexpr lambda
    template <typename PseudoLowStrides>
    struct lambda_CalculateLowerIndex
    {
        index_t& itmp;
        LowerIndex& idx_low;

        __host__ __device__ explicit constexpr lambda_CalculateLowerIndex(index_t& itmp_,
                                                                          LowerIndex& idx_low_)
            : itmp(itmp_), idx_low(idx_low_)
        {
        }

        template <typename IDim>
        __host__ __device__ constexpr void operator()(IDim idim) const
        {
            constexpr index_t stride = PseudoLowStrides::At(idim);
            idx_low(idim)            = itmp / stride;
            itmp -= idx_low[idim] * stride;
        }
    };

    __host__ __device__ static constexpr auto CalculateLowerIndex(const UpperIndex& idx_up)
    {
        LowerIndex idx_low;

        index_t itmp = idx_up[0];

        constexpr auto pseudo_low_strides =
            reverse_inclusive_scan_sequence(
                LowerLengths::PopFront(), math::multiplies<index_t>{}, Number<1>{})
                .PushBack(Number<1>{});

        static_for<0, nDimLow - 1, 1>{}(
            lambda_CalculateLowerIndex<decltype(pseudo_low_strides)>(itmp, idx_low));

        idx_low(nDimLow - 1) = itmp / pseudo_low_strides[nDimLow - 1];

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
        if(idx_up_diff[0] == 0)
        {
            return make_zero_array<index_t, nDimLow>();
        }
        else
        {
            // CalculateLowerIndex(idx_up_diff) has multiple integer divisions.
            //   If idx_up_diff is known at compile-time, the calculation can
            //   be done at compile-time. However, if idx_up_diff is only known
            //   at run-time, then the calculation will also be computed at
            //   run-time, and can be very expensive.
            LowerIndex idx_low_diff_tmp = CalculateLowerIndex(idx_up_diff);

            // find out the last low dimension that changed
            index_t last_changed_low_dim = 0;

            static_for<0, nDimLow, 1>{}([&](auto i) {
                if(idx_low_diff_tmp[i] != 0)
                {
                    last_changed_low_dim = i;
                }
            });

            LowerIndex idx_low_new = idx_low_old + idx_low_diff_tmp;

            if(idx_up_diff[0] > 0)
            {
                // do carry check on each low dimension in reversed order
                // starting from the first digit that changed
                // don't check the highest dimension
                bool carry = false;

                static_for<nDimLow - 1, 0, -1>{}([&](auto i) {
                    if(i <= last_changed_low_dim)
                    {
                        if(carry)
                        {
                            ++idx_low_new(i);
                        }

                        carry = false;

                        if(idx_low_new[i] >= LowerLengths::At(i))
                        {
                            idx_low_new(i) -= LowerLengths::At(i);
                            carry = true;
                        }
                    }
                });

                // highest dimension, no out-of-bound check
                if(carry)
                {
                    ++idx_low_new(0);
                }
            }
            else
            {
                // do borrow check on each low dimension in reversed order
                // starting from the first digit that changed
                // don't check the highest dimension
                bool borrow = false;

                static_for<nDimLow - 1, 0, -1>{}([&](auto i) {
                    if(i <= last_changed_low_dim)
                    {
                        if(borrow)
                        {
                            --idx_low_new(i);
                        }

                        borrow = false;

                        if(idx_low_new[i] < 0)
                        {
                            idx_low_new(i) += LowerLengths::At(i);
                            borrow = true;
                        }
                    }
                });

                // highest dimension, no out-of-bound check
                if(borrow)
                {
                    --idx_low_new(0);
                }
            }

            return idx_low_new - idx_low_old;
        }
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }
};

} // namespace ck
#endif
