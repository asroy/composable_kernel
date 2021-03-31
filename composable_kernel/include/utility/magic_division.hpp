#ifndef CK_MAGIC_DIVISION_HPP
#define CK_MAGIC_DIVISION_HPP

#include "config.hpp"
#include "integral_constant.hpp"
#include "number.hpp"
#include "type.hpp"
#include "tuple.hpp"

namespace ck {

// magic number division
struct magic_division
{
    // uint32_t
    __host__ __device__ static constexpr auto CalculateMagicNumbers(uint32_t divisor)
    {
        // assert(divisior >= 1 && divisior <= INT32_MAX);
        uint32_t shift = 0;
        for(shift = 0; shift < 32; ++shift)
        {
            if((1U << shift) >= divisor)
            {
                break;
            }
        }

        uint64_t one        = 1;
        uint64_t multiplier = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
        // assert(multiplier <= 0xffffffffUL);

        return make_tuple(uint32_t(multiplier), shift);
    }

    __host__ __device__ static constexpr uint32_t CalculateMagicMultiplier(uint32_t divisor)
    {
        auto tmp = CalculateMagicNumbers(divisor);

        return tmp[Number<0>{}];
    }

    __host__ __device__ static constexpr uint32_t CalculateMagicShift(uint32_t divisor)
    {
        auto tmp = CalculateMagicNumbers(divisor);

        return tmp[Number<1>{}];
    }

    // integral_constant<uint32_t, .>
    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicNumbers(integral_constant<uint32_t, Divisor>)
    {
        constexpr auto tmp = CalculateMagicNumbers(uint32_t{Divisor});

        constexpr uint32_t multiplier = tmp[Number<0>{}];
        constexpr uint32_t shift      = tmp[Number<1>{}];

        return make_tuple(integral_constant<uint32_t, multiplier>{},
                          integral_constant<uint32_t, shift>{});
    }

    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicMultiplier(integral_constant<uint32_t, Divisor>)
    {
        constexpr uint32_t multiplier = CalculateMagicMultiplier(uint32_t{Divisor});

        return integral_constant<uint32_t, multiplier>{};
    }

    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicShift(integral_constant<uint32_t, Divisor>)
    {
        constexpr uint32_t shift = CalculateMagicShift(uint32_t{Divisor});

        return integral_constant<uint32_t, shift>{};
    }

    // integral_constant<int32_t, .>
    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicNumbers(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicNumbers(integral_constant<uint32_t, Divisor>{});
    }

    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicMultiplier(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicMultiplier(integral_constant<uint32_t, Divisor>{});
    }

    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicShift(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicShift(integral_constant<uint32_t, Divisor>{});
    }

    // magic division
    __host__ __device__ static constexpr uint32_t
    DoMagicDivision(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = (uint64_t(dividend) * uint64_t(multiplier)) >> 32;
        return (tmp + dividend) >> shift;
    }
};

} // namespace ck

#endif
