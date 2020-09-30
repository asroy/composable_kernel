#ifndef CK_STATICALLY_INDEXED_ARRAY_HPP
#define CK_STATICALLY_INDEXED_ARRAY_HPP

#include "functional2.hpp"
#include "sequence.hpp"
#include "tuple.hpp"

namespace ck {

#if 0
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

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 1> : public Tuple<T>
{
    using type      = StaticallyIndexedArray;
    using data_type = T;
    using base      = Tuple<T>;
    static constexpr index_t nsize = base::Size();

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;

    template <typename Y>
    __host__
        __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray<Y, nsize>& y)
        : base(static_cast<const Tuple<Y>&>(y))
    {
    }

    template <typename Y>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray<Y, nsize>&& y)
        : base(static_cast<Tuple<Y>&&>(y))
    {
    }

#if 0
    template <typename... Ys,
              typename std::enable_if<sizeof...(Ys) == base::Size(),
                  bool>::type = false>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
        static_assert(sizeof...(Ys) == nsize, "wrong! inconsistent size");
    }
#else
    template <typename Y>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Y&& y)
        : base(std::forward<Y>(y))
    {
    }
#endif
};

template <typename T>
struct StaticallyIndexedArray<T, 2> : public Tuple<T, T>
{
    using data_type = T;
    using base      = Tuple<T, T>;

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;

    template <typename Y>
    __host__
        __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray<Y, 2>& y)
        : base(static_cast<const Tuple<Y, Y>&>(y))
    {
    }

    template <typename Y>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray<Y, 2>&& y)
        : base(static_cast<Tuple<Y, Y>&&>(y))
    {
    }

    template <typename... Ys,
              typename std::enable_if<sizeof...(Ys) == base::Size(),
                  bool>::type = false>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
        static_assert(sizeof...(Ys) == 2, "wrong! inconsistent size");
    }
};

template <typename T>
struct StaticallyIndexedArray<T, 3> : public Tuple<T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 4> : public Tuple<T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 5> : public Tuple<T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 6> : public Tuple<T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 7> : public Tuple<T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 8> : public Tuple<T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 9> : public Tuple<T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 10> : public Tuple<T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 11> : public Tuple<T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 12> : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 13> : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 14> : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 15> : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 16> : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 17>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 18>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 19>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 20>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 21>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};

template <typename T>
struct StaticallyIndexedArray<T, 22>
    : public Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>
{
    using data_type = T;
    using base      = Tuple<T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;
};
#else
namespace detail {

template <typename T, index_t NSize>
__host__ __device__ constexpr auto generate_same_type_tuple()
{
    return generate_tuple([](auto) -> T { return T{}; }, Number<NSize>{});
}

template <typename T, index_t NSize>
using same_type_tuple = decltype(generate_same_type_tuple<T, NSize>());

} // namespace detail

#if 0
template <typename T, index_t NSize>
struct StaticallyIndexedArray : public detail::same_type_tuple<T, NSize>
{
    using type      = StaticallyIndexedArray;
    using data_type = T;
    using base      = detail::same_type_tuple<T, NSize>;

    __host__ __device__ explicit constexpr StaticallyIndexedArray(const StaticallyIndexedArray&) =
        default;

    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray&&) = default;

    template <typename Y>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(
        const StaticallyIndexedArray<Y, NSize>& y)
        : base(static_cast<const detail::same_type_tuple<Y, NSize>&>(y))
    {
    }

    template <typename Y>
    __host__
        __device__ explicit constexpr StaticallyIndexedArray(StaticallyIndexedArray<Y, NSize>&& y)
        : base(static_cast<detail::same_type_tuple<Y, NSize>&&>(y))
    {
    }

    template <typename... Ys,
              typename std::enable_if<sizeof...(Ys) == base::Size(), bool>::type = false>
    __host__ __device__ explicit constexpr StaticallyIndexedArray(Ys&&... ys)
        : base(std::forward<Ys>(ys)...)
    {
        static_assert(sizeof...(Ys) == NSize, "wrong! inconsistent size");
    }
};
#else
template <typename T, index_t NSize>
using StaticallyIndexedArray = detail::same_type_tuple<T, NSize>;
#endif

#endif

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
