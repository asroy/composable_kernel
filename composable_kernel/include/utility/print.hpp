#ifndef CK_PRINT_HPP
#define CK_PRINT_HPP

#include "array.hpp"
#include "sequence.hpp"

namespace ck {

template <typename T>
__host__ __device__ void print_array(const char* s, T a)
{
    using data_type         = typename decltype(a)::data_type;
    constexpr index_t nsize = a.Size();

    static_assert(nsize >= 0 && nsize <= 10, "wrong!");

    if constexpr(is_same<data_type, uint32_t>{})
    {
        if constexpr(nsize == 0)
        {
            printf("%s size %u\n", s, nsize);
        }
        else if constexpr(nsize == 1)
        {
            printf("%s size %u, {%u}\n", s, nsize, a[0]);
        }
        else if constexpr(nsize == 2)
        {
            printf("%s size %u, {%u %u}\n", s, nsize, a[0], a[1]);
        }
        else if constexpr(nsize == 3)
        {
            printf("%s size %u, {%u %u %u}\n", s, nsize, a[0], a[1], a[2]);
        }
        else if constexpr(nsize == 4)
        {
            printf("%s size %u, {%u %u %u %u}\n", s, nsize, a[0], a[1], a[2], a[3]);
        }
        else if constexpr(nsize == 5)
        {
            printf("%s size %u, {%u %u %u %u %u}\n", s, nsize, a[0], a[1], a[2], a[3], a[4]);
        }
        else if constexpr(nsize == 6)
        {
            printf(
                "%s size %u, {%u %u %u %u %u %u}\n", s, nsize, a[0], a[1], a[2], a[3], a[4], a[5]);
        }
        else if constexpr(nsize == 7)
        {
            printf("%s size %u, {%u %u %u %u %u %u %u}\n",
                   s,
                   nsize,
                   a[0],
                   a[1],
                   a[2],
                   a[3],
                   a[4],
                   a[5],
                   a[6]);
        }
        else if constexpr(nsize == 8)
        {
            printf("%s size %u, {%u %u %u %u %u %u %u %u}\n",
                   s,
                   nsize,
                   a[0],
                   a[1],
                   a[2],
                   a[3],
                   a[4],
                   a[5],
                   a[6],
                   a[7]);
        }
        else if constexpr(nsize == 9)
        {
            printf("%s size %u, {%u %u %u %u %u %u %u %u %u}\n",
                   s,
                   nsize,
                   a[0],
                   a[1],
                   a[2],
                   a[3],
                   a[4],
                   a[5],
                   a[6],
                   a[7],
                   a[8]);
        }
        else if constexpr(nsize == 10)
        {
            printf("%s size %u, {%u %u %u %u %u %u %u %u %u %u}\n",
                   s,
                   nsize,
                   a[0],
                   a[1],
                   a[2],
                   a[3],
                   a[4],
                   a[5],
                   a[6],
                   a[7],
                   a[8],
                   a[9]);
        }
    }
    else if constexpr(is_same<data_type, int32_t>{})
    {
        if constexpr(nsize == 0)
        {
            printf("%s size %d\n", s, nsize);
        }
        else if constexpr(nsize == 1)
        {
            printf("%s size %d, {%d}\n", s, nsize, a[0]);
        }
        else if constexpr(nsize == 2)
        {
            printf("%s size %d, {%d %d}\n", s, nsize, a[0], a[1]);
        }
        else if constexpr(nsize == 3)
        {
            printf("%s size %d, {%d %d %d}\n", s, nsize, a[0], a[1], a[2]);
        }
        else if constexpr(nsize == 4)
        {
            printf("%s size %d, {%d %d %d %d}\n", s, nsize, a[0], a[1], a[2], a[3]);
        }
        else if constexpr(nsize == 5)
        {
            printf("%s size %d, {%d %d %d %d %d}\n", s, nsize, a[0], a[1], a[2], a[3], a[4]);
        }
        else if constexpr(nsize == 6)
        {
            printf(
                "%s size %d, {%d %d %d %d %d %d}\n", s, nsize, a[0], a[1], a[2], a[3], a[4], a[5]);
        }
        else if constexpr(nsize == 7)
        {
            printf("%s size %d, {%d %d %d %d %d %d %d}\n",
                   s,
                   nsize,
                   a[0],
                   a[1],
                   a[2],
                   a[3],
                   a[4],
                   a[5],
                   a[6]);
        }
        else if constexpr(nsize == 8)
        {
            printf("%s size %d, {%d %d %d %d %d %d %d %d}\n",
                   s,
                   nsize,
                   a[0],
                   a[1],
                   a[2],
                   a[3],
                   a[4],
                   a[5],
                   a[6],
                   a[7]);
        }
        else if constexpr(nsize == 9)
        {
            printf("%s size %d, {%d %d %d %d %d %d %d %d %d}\n",
                   s,
                   nsize,
                   a[0],
                   a[1],
                   a[2],
                   a[3],
                   a[4],
                   a[5],
                   a[6],
                   a[7],
                   a[8]);
        }
        else if constexpr(nsize == 10)
        {
            printf("%s size %d, {%d %d %d %d %d %d %d %d %d %d}\n",
                   s,
                   nsize,
                   a[0],
                   a[1],
                   a[2],
                   a[3],
                   a[4],
                   a[5],
                   a[6],
                   a[7],
                   a[8],
                   a[9]);
        }
    }
}

} // namespace ck
#endif
