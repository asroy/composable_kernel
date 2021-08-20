/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef CK_GRIDWISE_GENERIC_REDUCTION_WRAPPER_COMMON
#define CK_GRIDWISE_GENERIC_REDUCTION_WRAPPER_COMMON

#include "config.hpp"
#include "number.hpp"
#include "sequence.hpp"
#include "tensor_descriptor_helper.hpp"
#include "reduction_common.hpp"

namespace ck {

template <char tid>
struct get_type_from_type_id
{
    using type = float;
};

template <>
struct get_type_from_type_id<'H'>
{
    using type = half_t;
};

template <>
struct get_type_from_type_id<'F'>
{
    using type = float;
};

template <>
struct get_type_from_type_id<'D'>
{
    using type = double;
};

template <index_t persistentID>
struct get_reduce_op // any other ID
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::ADD;
};

template <>
struct get_reduce_op<656868> // 'A' * 10000 + 'D' * 100 + 'D'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::ADD;
};

template <>
struct get_reduce_op<778576> // 'M' * 10000 + 'U' * 100 + 'L'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MUL;
};

template <>
struct get_reduce_op<777378> // 'M' * 10000 + 'I' * 100 + 'N'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MIN;
};

template <>
struct get_reduce_op<776588> // 'M' * 10000 + 'A' * 100 + 'X'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MAX;
};

template <>
struct get_reduce_op<657788> // 'A' * 10000 + 'M' * 100 + 'X'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::AMAX;
};

template <>
struct get_reduce_op<658671> // 'A' * 10000 + 'V' * 100 + 'G'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::AVG;
};

template <>
struct get_reduce_op<788201> // 'N' * 10000 + 'R' * 100 + '1'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::NORM1;
};

template <>
struct get_reduce_op<788202> // 'N' * 10000 + 'R' * 100 + '2'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::NORM2;
};

}; // end of namespace ck

#endif
