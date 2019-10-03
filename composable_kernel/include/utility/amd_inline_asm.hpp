#ifndef CK_AMD_INLINE_ASM_HPP
#define CK_AMD_INLINE_ASM_HPP

#include "vector_type.hpp"

// disable inline asm due to the compiler issue: SWDEV-202749
#define WORKAROUND_SWDEV_202749 1

namespace ck {

// cast a pointer of LDS to its address
extern "C" __attribute__((address_space(3))) __device__ void* __to_local(void* p);

__device__ void __outer_product_1x2(float a, float b0, float b1, float& c0, float& c1)
{
///\to-do: enable the inline asm after the compiler fix
#if WORKAROUND_SWDEV_202749
    c0 += a * b0;
    c1 += a * b1;
#else
    asm volatile("\n \
            v_mac_f32 %0, %2, %3 \n \
            v_mac_f32 %1, %2, %4 \n \
            "
                 : "=v"(c0), "=v"(c1)
                 : "v"(a), "v"(b0), "v"(b1), "0"(c0), "1"(c1));
#endif
}

__device__ void __outer_product_1x4(
    float a, float b0, float b1, float b2, float b3, float& c0, float& c1, float& c2, float& c3)
{
    asm volatile("\n \
            v_mac_f32 %0, %4, %5 \n \
            v_mac_f32 %1, %4, %6 \n \
            v_mac_f32 %2, %4, %7 \n \
            v_mac_f32 %3, %4, %8 \n \
            "
                 : "=v"(c0), "=v"(c1), "=v"(c2), "=v"(c3)
                 : "v"(a), "v"(b0), "v"(b1), "v"(b2), "v"(b3), "0"(c0), "1"(c1), "2"(c2), "3"(c3));
}

} // namespace ck
#endif
