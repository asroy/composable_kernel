#ifndef CK_STATICALLY_INDEXED_ARRAY_HPP
#define CK_STATICALLY_INDEXED_ARRAY_HPP

#include "functional2.hpp"
#include "sequence.hpp"
#include "tuple.hpp"

namespace ck {

template <typename TData, index_t NSize>
struct StaticallyIndexedArray
{
};

template <typename TData>
struct StaticallyIndexedArray<TData, 0> : Tuple<>
{
    using data_type = TData;
    using base      = Tuple<>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys) : base(ys...)
    {
    }
};

template <typename TData>
struct StaticallyIndexedArray<TData, 1> : Tuple<TData>
{
    using data_type = TData;
    using base      = Tuple<TData>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys) : base(ys...)
    {
    }
};

template <typename TData>
struct StaticallyIndexedArray<TData, 2> : Tuple<TData, TData>
{
    using data_type = TData;
    using base      = Tuple<TData, TData>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys) : base(ys...)
    {
    }
};

template <typename TData>
struct StaticallyIndexedArray<TData, 3> : Tuple<TData, TData, TData>
{
    using data_type = TData;
    using base      = Tuple<TData, TData, TData>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys) : base(ys...)
    {
    }
};

template <typename TData>
struct StaticallyIndexedArray<TData, 4> : Tuple<TData, TData, TData, TData>
{
    using data_type = TData;
    using base      = Tuple<TData, TData, TData, TData>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys) : base(ys...)
    {
    }
};

template <typename TData>
struct StaticallyIndexedArray<TData, 5> : Tuple<TData, TData, TData, TData, TData>
{
    using data_type = TData;
    using base      = Tuple<TData, TData, TData, TData, TData>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys) : base(ys...)
    {
    }
};

template <typename TData>
struct StaticallyIndexedArray<TData, 6> : Tuple<TData, TData, TData, TData, TData, TData>
{
    using data_type = TData;
    using base      = Tuple<TData, TData, TData, TData, TData, TData>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys) : base(ys...)
    {
    }
};

template <typename TData>
struct StaticallyIndexedArray<TData, 7> : Tuple<TData, TData, TData, TData, TData, TData, TData>
{
    using data_type = TData;
    using base      = Tuple<TData, TData, TData, TData, TData, TData, TData>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys) : base(ys...)
    {
    }
};

template <typename TData>
struct StaticallyIndexedArray<TData, 8>
    : Tuple<TData, TData, TData, TData, TData, TData, TData, TData>
{
    using data_type = TData;
    using base      = Tuple<TData, TData, TData, TData, TData, TData, TData, TData>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys) : base(ys...)
    {
    }
};

template <typename TData>
struct StaticallyIndexedArray<TData, 9>
    : Tuple<TData, TData, TData, TData, TData, TData, TData, TData, TData>
{
    using data_type = TData;
    using base      = Tuple<TData, TData, TData, TData, TData, TData, TData, TData, TData>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys) : base(ys...)
    {
    }
};

template <typename TData>
struct StaticallyIndexedArray<TData, 10>
    : Tuple<TData, TData, TData, TData, TData, TData, TData, TData, TData, TData>
{
    using data_type = TData;
    using base      = Tuple<TData, TData, TData, TData, TData, TData, TData, TData, TData, TData>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys) : base(ys...)
    {
    }
};

template <typename TData>
struct StaticallyIndexedArray<TData, 11>
    : Tuple<TData, TData, TData, TData, TData, TData, TData, TData, TData, TData, TData>
{
    using data_type = TData;
};

template <typename TData>
struct StaticallyIndexedArray<TData, 12>
    : Tuple<TData, TData, TData, TData, TData, TData, TData, TData, TData, TData, TData, TData>
{
    using data_type = TData;
};

template <typename TData>
struct StaticallyIndexedArray<TData, 13> : Tuple<TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData>
{
    using data_type = TData;
};

template <typename TData>
struct StaticallyIndexedArray<TData, 14> : Tuple<TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData>
{
    using data_type = TData;
};

template <typename TData>
struct StaticallyIndexedArray<TData, 15> : Tuple<TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData>
{
    using data_type = TData;
};

template <typename TData>
struct StaticallyIndexedArray<TData, 16> : Tuple<TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData>
{
    using data_type = TData;
};

template <typename TData>
struct StaticallyIndexedArray<TData, 17> : Tuple<TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData>
{
    using data_type = TData;
};

template <typename TData>
struct StaticallyIndexedArray<TData, 18> : Tuple<TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData>
{
    using data_type = TData;
};

template <typename TData>
struct StaticallyIndexedArray<TData, 19> : Tuple<TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData>
{
    using data_type = TData;
};

template <typename TData>
struct StaticallyIndexedArray<TData, 20> : Tuple<TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData>
{
    using data_type = TData;
};

template <typename TData>
struct StaticallyIndexedArray<TData, 21> : Tuple<TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData>
{
    using data_type = TData;
};

template <typename TData>
struct StaticallyIndexedArray<TData, 22> : Tuple<TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData,
                                                 TData>
{
    using data_type = TData;
};

} // namespace ck
#endif
