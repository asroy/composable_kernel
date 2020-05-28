#ifndef CK_COMMON_HEADER_HPP
#define CK_COMMON_HEADER_HPP

#include "config.hpp"
#include "utility.hpp"
#include "integral_constant.hpp"
#include "number.hpp"
#include "float_type.hpp"
#include "type.hpp"
#include "tuple.hpp"
#include "math.hpp"
#include "sequence.hpp"
#include "array.hpp"
#include "functional.hpp"
#include "functional2.hpp"
#include "functional3.hpp"
#include "functional4.hpp"
#include "in_memory_operation.hpp"

#if CK_USE_AMD_INLINE_ASM
#include "amd_inline_asm.hpp"
#endif

#if CK_USE_AMD_BUFFER_ADDRESSING
#include "amd_buffer_addressing.hpp"
#endif

#if CK_USE_AMD_XDLOPS
#include "amd_xdlops.hpp"
#endif

#define CK_EXTEND_IMAGE_SIZE_PAD_W 1  //fake pad 7x7 into 7x8 to make it dwordx4 
#define CK_EXTEND_IMAGE_SIZE_PAD_HW 0  //fake pad 7x7 into 7x7+3 to make it dwordx4

#endif
