/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef ONLINE_REDUCE_COMMON_HPP_
#define ONLINE_REDUCE_COMMON_HPP_ 1

#include <half.hpp>

typedef enum
{
    DirectThreadWise = 1,
    DirectWarpWise   = 2,
    BlockWise        = 3,
    MultiBlock       = 4
} ReductionMethod_t;

typedef enum
{
    REDUCE_TENSOR_ADD   = 0,
    REDUCE_TENSOR_MUL   = 1,
    REDUCE_TENSOR_MIN   = 2,
    REDUCE_TENSOR_MAX   = 3,
    REDUCE_TENSOR_AMAX  = 4,
    REDUCE_TENSOR_AVG   = 5,
    REDUCE_TENSOR_NORM1 = 6,
    REDUCE_TENSOR_NORM2 = 7
} ReduceTensorOp_t;

typedef enum
{
    NOT_PROPAGATE_NAN = 0,
    PROPAGATE_NAN     = 1,
} NanPropagation_t;

typedef enum
{
    REDUCE_TENSOR_NO_INDICES        = 0,
    REDUCE_TENSOR_FLATTENED_INDICES = 1,
} ReduceTensorIndices_t;

typedef enum
{
    APP_32BIT_INDICES = 0,
    APP_64BIT_INDICES = 1,
    APP_16BIT_INDICES = 2,
    APP_8BIT_INDICES  = 3,
} IndicesType_t;

namespace reduce {

template <typename Tdst, typename Tsrc>
static inline Tdst convert_type(Tsrc x)
{
    return static_cast<Tdst>(x);
}

} // namespace reduce

// this enumerate should be synchronized with include/miopen.h
typedef enum
{
    appHalf     = 0,
    appFloat    = 1,
    appInt32    = 2,
    appInt8     = 3,
    appInt8x4   = 4,
    appBFloat16 = 5,
    appDouble   = 6,
} appDataType_t;

namespace Driver {

template <appDataType_t typeNum>
struct get_type_from_type_enum
{
    using type = float;
};

template <>
struct get_type_from_type_enum<appHalf>
{
    using type = half_float::half;
};

template <>
struct get_type_from_type_enum<appFloat>
{
    using type = float;
};

template <>
struct get_type_from_type_enum<appDouble>
{
    using type = double;
};

template <>
struct get_type_from_type_enum<appInt32>
{
    using type = int;
};

template <typename T>
static inline appDataType_t get_type_enum_from_type()
{
    throw std::runtime_error("Unsupported typeid conversion for this type!");
};

template <>
inline appDataType_t get_type_enum_from_type<float>()
{
    return (appFloat);
};

template <>
inline appDataType_t get_type_enum_from_type<half_float::half>()
{
    return (appHalf);
};

template <>
inline appDataType_t get_type_enum_from_type<double>()
{
    return (appDouble);
};

static inline float get_effective_average(std::vector<float>& values)
{
    assert(!values.empty());

    if(values.size() == 1)
        return (values[0]);
    else
    {
        float sum    = 0.0f;
        float maxVal = 0.0f;

        for(const auto val : values)
        {
            if(maxVal < val)
                maxVal = val;
            sum += val;
        };

        return ((sum - maxVal) / (values.size() - 1));
    };
};

} // namespace Driver

#endif
