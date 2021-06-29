#ifndef HOST_TENSOR_GENERATOR_HPP
#define HOST_TENSOR_GENERATOR_HPP

#include <cmath>
#include "config.hpp"

struct GeneratorTensor_1
{
    int value = 1;

    template <typename... Is>
    double operator()(Is... is)
    {
        return value;
    }
};

struct GeneratorTensor_2
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    double operator()(Is...)
    {
        return (std::rand() % (max_value - min_value)) + min_value;
    }
};

struct GeneratorTensor_3
{
    template <typename... Is>
    double operator()(Is... is)
    {
        std::array<ck::index_t, sizeof...(Is)> dims = {{static_cast<ck::index_t>(is)...}};

        auto f_acc = [](auto a, auto b) { return 10 * a + b; };

        return std::accumulate(dims.begin(), dims.end(), ck::index_t(0), f_acc);
    }
};

struct GeneratorTensor_4
{
    template <typename A>
    double operator()(A a)
    {
        return std::pow(2, a);
    }

    template <typename A, typename B>
    double operator()(A a, B b)
    {
        return std::pow(2, a);
    }

    template <typename A, typename B, typename C>
    double operator()(A a, B b, C c)
    {
        return std::pow(2, a);
    }

    template <typename A, typename B, typename C, typename D>
    double operator()(A a, B b, C c, D d)
    {
        return std::pow(2, a);
    }
};

struct GeneratorTensor_5
{
    template <typename A>
    double operator()(A a)
    {
        return std::pow(2, a);
    }

    template <typename A, typename B>
    double operator()(A a, B b)
    {
        return std::pow(2, b);
    }

    template <typename A, typename B, typename C>
    double operator()(A a, B b, C c)
    {
        return std::pow(2, c);
    }

    template <typename A, typename B, typename C, typename D>
    double operator()(A a, B b, C c, D d)
    {
        return c * 100 + d;
    }
};

struct GeneratorTensor_Checkboard
{
    template <typename... Ts>
    double operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};
        return std::accumulate(dims.begin(),
                               dims.end(),
                               true,
                               [](bool init, ck::index_t x) -> int { return init != (x % 2); })
                   ? 1
                   : -1;
    }
};

#endif
