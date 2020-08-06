#ifndef CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP
#define CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP

#include "common_header.hpp"

namespace ck {

class DynamicTransformation
{
};

template<index_t N>
class DynamicEmbed : public DynamicTransformation
{
    const array<idnex_t, N+1> coefficients_;

    __host__ __device__ constexpr DynamicEmbed(coefficients)
        : coefficients_(coefficients)
    {
    }
};

} // namespace ck
#endif
