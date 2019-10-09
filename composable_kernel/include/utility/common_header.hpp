#ifndef CK_COMMON_HEADER_HPP
#define CK_COMMON_HEADER_HPP

#define MIOPEN_USE_FP16 0
#define MIOPEN_USE_BFP16 0
#define MIOPEN_USE_FP32 1

#define __HIP_PLATFORM_HCC__ 1

#include "config.hpp"
#include "utility.hpp"
#include "integral_constant.hpp"
#include "math.hpp"
#include "vector_type.hpp"
#include "Sequence.hpp"
#include "Array.hpp"
#include "functional.hpp"
#include "functional2.hpp"
#include "functional3.hpp"

#if CK_USE_AMD_INLINE_ASM
#include "amd_inline_asm.hpp"
#endif

#endif
