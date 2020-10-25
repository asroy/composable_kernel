#pragma once
#include "host_tensor.hpp"

template <typename T,
          typename FilterSizes,
          typename OutputSizes,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads>
void host_col2im(const Tensor<T>& in_eb,
                 Tensor<T>& in_nchw,
                 FilterSizes,
                 OutputSizes,
                 ConvStrides,
                 ConvDilations,
                 LeftPads,
                 RightPads)
{
    using namespace ck;

    int N  = in_nchw.mDesc.GetLengths()[0];
    int C  = in_nchw.mDesc.GetLengths()[1];
    int Hi = in_nchw.mDesc.GetLengths()[2];
    int Wi = in_nchw.mDesc.GetLengths()[3];

    int Y = FilterSizes{}[0];
    int X = FilterSizes{}[1];

    int Ho = OutputSizes{}[0];
    int Wo = OutputSizes{}[1];

    auto f = [&](auto n, auto c, auto hi, auto wi) {
        double v = 0;

        for(int y = 0; y < Y; ++y)
        {
            int h_tmp = hi + LeftPads{}[0] - y * ConvDilations{}[0];

            if(h_tmp % ConvStrides{}[0] == 0)
            {
                int ho = h_tmp / ConvStrides{}[0];

                if(ho >= 0 && ho < Ho)
                {
                    for(int x = 0; x < X; ++x)
                    {
                        int w_tmp = wi + LeftPads{}[1] - x * ConvDilations{}[1];

                        if(w_tmp % ConvStrides{}[1] == 0)
                        {
                            int wo = w_tmp / ConvStrides{}[1];

                            if(wo >= 0 && wo < Wo && w_tmp % ConvStrides{}[1] == 0)
                            {
                                int e = c * (Y * X) + y * X + x;
                                int b = n * (Ho * Wo) + ho * Wo + wo;

                                v += in_eb(e, b);
                            }
                        }
                    }
                }
            }
        }

        in_nchw(n, c, hi, wi) = v;
    };

    auto f_par = make_ParallelTensorFunctor(f,
                                            in_nchw.mDesc.GetLengths()[0],
                                            in_nchw.mDesc.GetLengths()[1],
                                            in_nchw.mDesc.GetLengths()[2],
                                            in_nchw.mDesc.GetLengths()[3]);

    f_par(std::thread::hardware_concurrency());
}
