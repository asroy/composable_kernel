#ifndef CK_PRINT_SEQUENCE_HPP
#define CK_PRINT_SEQUENCE_HPP

#include "sequence.hpp"

namespace ck {

template <index_t... Xs>
__host__ __device__ void print_sequence(const char* s, Sequence<Xs...>)
{
    constexpr index_t nsize = Sequence<Xs...>::Size();

    static_assert(nsize <= 10, "wrong!");

    static_if<nsize == 0>{}([&](auto) { printf("%s size %u, {}\n", s, nsize, Xs...); });

    static_if<nsize == 1>{}([&](auto) { printf("%s size %u, {%u}\n", s, nsize, Xs...); });

    static_if<nsize == 2>{}([&](auto) { printf("%s size %u, {%u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 3>{}([&](auto) { printf("%s size %u, {%u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 4>{}([&](auto) { printf("%s size %u, {%u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 5>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 6>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 7>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 8>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 9>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u %u %u}\n", s, nsize, Xs...); });

    static_if<nsize == 10>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u %u %u %u %u %u %u}\n", s, nsize, Xs...); });
}

template <typename... Xr>
__host__ __device__ void print_sequence(const char* s, const DynamicSequence<Xr...>& dseq)
{
    constexpr index_t nsize = dseq.GetSize();

    static_assert(nsize > 0 && nsize <= 10, "wrong!");

    static_if<nsize == 1>{}([&](auto) { printf("%s size %u, {%u}\n", s, nsize, dseq[0]); });

    static_if<nsize == 2>{}(
        [&](auto) { printf("%s size %u, {%u %u}\n", s, nsize, dseq[0], dseq[1]); });

    static_if<nsize == 3>{}(
        [&](auto) { printf("%s size %u, {%u %u %u}\n", s, nsize, dseq[0], dseq[1], dseq[2]); });

    static_if<nsize == 4>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u}\n", s, nsize, dseq[0], dseq[1], dseq[2], dseq[3]);
    });

    static_if<nsize == 5>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u}\n",
               s,
               nsize,
               dseq[0],
               dseq[1],
               dseq[2],
               dseq[3],
               dseq[4]);
    });

    static_if<nsize == 6>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u}\n",
               s,
               nsize,
               dseq[0],
               dseq[1],
               dseq[2],
               dseq[3],
               dseq[4],
               dseq[5]);
    });

    static_if<nsize == 7>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u}\n",
               s,
               nsize,
               dseq[0],
               dseq[1],
               dseq[2],
               dseq[3],
               dseq[4],
               dseq[5],
               dseq[6]);
    });

    static_if<nsize == 8>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u %u}\n",
               s,
               nsize,
               dseq[0],
               dseq[1],
               dseq[2],
               dseq[3],
               dseq[4],
               dseq[5],
               dseq[6],
               dseq[7]);
    });

    static_if<nsize == 9>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u %u %u}\n",
               s,
               nsize,
               dseq[0],
               dseq[1],
               dseq[2],
               dseq[3],
               dseq[4],
               dseq[5],
               dseq[6],
               dseq[7],
               dseq[8]);
    });

    static_if<nsize == 10>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u %u %u %u}\n",
               s,
               nsize,
               dseq[0],
               dseq[1],
               dseq[2],
               dseq[3],
               dseq[4],
               dseq[5],
               dseq[6],
               dseq[7],
               dseq[8],
               dseq[9]);
    });
}

} // namespace ck

#endif
