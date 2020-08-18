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

    static_assert(nsize > 0 && nsize <= 10, "wrong!");

    if
        constexpr(is_same<data_type, uint32_t>{})
        {
            static_if<nsize == 1>{}([&](auto) { printf("%s size %u, {%u}\n", s, nsize, a[0]); });

            static_if<nsize == 2>{}(
                [&](auto) { printf("%s size %u, {%u %u}\n", s, nsize, a[0], a[1]); });

            static_if<nsize == 3>{}(
                [&](auto) { printf("%s size %u, {%u %u %u}\n", s, nsize, a[0], a[1], a[2]); });

            static_if<nsize == 4>{}([&](auto) {
                printf("%s size %u, {%u %u %u %u}\n", s, nsize, a[0], a[1], a[2], a[3]);
            });

            static_if<nsize == 5>{}([&](auto) {
                printf("%s size %u, {%u %u %u %u %u}\n", s, nsize, a[0], a[1], a[2], a[3], a[4]);
            });

            static_if<nsize == 6>{}([&](auto) {
                printf("%s size %u, {%u %u %u %u %u %u}\n",
                       s,
                       nsize,
                       a[0],
                       a[1],
                       a[2],
                       a[3],
                       a[4],
                       a[5]);
            });

            static_if<nsize == 7>{}([&](auto) {
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
            });

            static_if<nsize == 8>{}([&](auto) {
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
            });

            static_if<nsize == 9>{}([&](auto) {
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
            });

            static_if<nsize == 10>{}([&](auto) {
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
            });
        }
    else if
        constexpr(is_same<data_type, int32_t>{})
        {
            static_if<nsize == 1>{}([&](auto) { printf("%s size %d, {%d}\n", s, nsize, a[0]); });

            static_if<nsize == 2>{}(
                [&](auto) { printf("%s size %d, {%d %d}\n", s, nsize, a[0], a[1]); });

            static_if<nsize == 3>{}(
                [&](auto) { printf("%s size %d, {%d %d %d}\n", s, nsize, a[0], a[1], a[2]); });

            static_if<nsize == 4>{}([&](auto) {
                printf("%s size %d, {%d %d %d %d}\n", s, nsize, a[0], a[1], a[2], a[3]);
            });

            static_if<nsize == 5>{}([&](auto) {
                printf("%s size %d, {%d %d %d %d %d}\n", s, nsize, a[0], a[1], a[2], a[3], a[4]);
            });

            static_if<nsize == 6>{}([&](auto) {
                printf("%s size %d, {%d %d %d %d %d %d}\n",
                       s,
                       nsize,
                       a[0],
                       a[1],
                       a[2],
                       a[3],
                       a[4],
                       a[5]);
            });

            static_if<nsize == 7>{}([&](auto) {
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
            });

            static_if<nsize == 8>{}([&](auto) {
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
            });

            static_if<nsize == 9>{}([&](auto) {
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
            });

            static_if<nsize == 10>{}([&](auto) {
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
            });
        }
}

} // namespace ck
#endif
