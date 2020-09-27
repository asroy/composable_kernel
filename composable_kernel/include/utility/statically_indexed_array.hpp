#ifndef CK_STATICALLY_INDEXED_ARRAY_HPP
#define CK_STATICALLY_INDEXED_ARRAY_HPP

#include "functional2.hpp"
#include "sequence.hpp"
#include "tuple.hpp"

namespace ck {

template <typename T, index_t NSize>
struct StaticallyIndexedArray
{
};

template <typename T>
struct StaticallyIndexedArray<T, 0> : public Tuple<>
{
    using data_type = T;
    using base      = Tuple<>;

    __host__ __device__ explicit constexpr StaticallyIndexedArray() : base() {}
};

template <typename T>
struct StaticallyIndexedArray<T, 1> : public Tuple<T>
{
    using data_type = T;
    using base      = Tuple<T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 2> : public Tuple<T, T>
{
    using data_type = T;
    using base      = Tuple<T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 3> : public Tuple<T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 4> : public Tuple<T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 5> : public Tuple<T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 6> : public Tuple<T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 7> : public Tuple<T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 8> : public Tuple<T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 9> : public Tuple<T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 10> : public Tuple<T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 11> : public Tuple<T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 12> : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 13> : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 14> : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 15> : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 16> : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 17>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 18>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 19>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 20>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 21>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 22>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(static_cast<T&&>(ys)...)
    {
    }
};

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_statically_indexed_array(const X& x, const Xs&... xs)
{
    return StaticallyIndexedArray<X, sizeof...(Xs) + 1>(x, static_cast<X>(xs)...);
}

// make empty StaticallyIndexedArray
template <typename X>
__host__ __device__ constexpr auto make_statically_indexed_array()
{
    return StaticallyIndexedArray<X, 0>();
}

} // namespace ck
#endif
