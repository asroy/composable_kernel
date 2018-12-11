#pragma once
#include "constant_tensor_descriptor.cuh"

template <class TFloat,
          class InThreadDesc,      //{NPerThread, CPerThread, InTileSizeH, WPerThread}
          class InTransThreadDesc, //{NPerThread, CPerThread, InTileSizeH, WPerThread}
          unsigned InTileSizeH,
          unsigned S>
__device__ void
threadwise_winograd_vertical_transform_input(InThreadDesc,
                                             TFloat* const __restrict__ p_in_thread,
                                             InTransThreadDesc,
                                             TFloat* __restrict__ p_in_transform_thread)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_thread_desc       = InThreadDesc{};
    constexpr auto in_trans_thread_desc = InTransThreadDesc{};

    for(unsigned n = 0; n < in_thread_desc.GetLength(I0); ++n)
    {
        for(unsigned c = 0; c < in_thread_desc.GetLength(I1); ++c)
        {
            for(unsigned w = 0; w < in_thread_desc.GetLength(I3); ++w)
            {
                const unsigned s0 = in_thread_desc.Get1dIndex(n, c, 0, w);
                const unsigned s1 = in_thread_desc.Get1dIndex(n, c, 1, w);
                const unsigned s2 = in_thread_desc.Get1dIndex(n, c, 2, w);
                const unsigned s3 = in_thread_desc.Get1dIndex(n, c, 3, w);

                const unsigned d0 = in_transform_thread_desc.Get1dIndex(n, c, 0, w);
                const unsigned d1 = in_transform_thread_desc.Get1dIndex(n, c, 1, w);
                const unsigned d2 = in_transform_thread_desc.Get1dIndex(n, c, 2, w);
                const unsigned d3 = in_transform_thread_desc.Get1dIndex(n, c, 3, w);

                p_in_transform_thread[d0] = p_in_thread[s0] - p_in_thread[s2];
                p_in_transform_thread[d1] = p_in_thread[s1] + p_in_thread[s2];
                p_in_transform_thread[d2] = -p_in_thread[s1] + p_in_thread[s2];
                p_in_transform_thread[d3] = p_in_thread[s1] - p_in_thread[s3];
            }
        }
    }
}

template <class TFloat,
          class InThreadDesc,      //{NPerThread, CPerThread, HPerThread, InTileSizeW}
          class InTransThreadDesc, //{NPerThread, CPerThread, HPerThread, InTileSizeW}
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned S,
          unsigned R,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW>
__device__ void
threadwise_winograd_horizontal_transform_input(InThreadDesc,
                                               TFloat* const __restrict__ p_in_thread,
                                               InTransThreadDesc,
                                               TFloat* __restrict__ p_in_transform_thread)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_thread_desc       = InThreadDesc{};
    constexpr auto in_trans_thread_desc = InTransThreadDesc{};

    for(unsigned n = 0; n < in_thread_desc.GetLength(I0); ++n)
    {
        for(unsigned c = 0; c < in_thread_desc.GetLength(I1); ++c)
        {
            for(unsigned h = 0; h < in_thread_desc.GetLength(I2); ++h)
            {
                const unsigned s0 = in_thread_desc.Get1dIndex(n, c, h, 0);
                const unsigned s1 = in_thread_desc.Get1dIndex(n, c, h, 1);
                const unsigned s2 = in_thread_desc.Get1dIndex(n, c, h, 2);
                const unsigned s3 = in_thread_desc.Get1dIndex(n, c, h, 3);

                const unsigned d0 = in_transform_thread_desc.Get1dIndex(n, c, h, 0);
                const unsigned d1 = in_transform_thread_desc.Get1dIndex(n, c, h, 1);
                const unsigned d2 = in_transform_thread_desc.Get1dIndex(n, c, h, 2);
                const unsigned d3 = in_transform_thread_desc.Get1dIndex(n, c, h, 3);

                p_in_transform_thread[d0] = p_in_thread[s0];
                p_in_transform_thread[d1] = p_in_thread[s1] - p_in_thread[s2] + p_in_thread[s3];
                p_in_transform_thread[d2] = -p_in_thread[s0] + p_in_thread[s1] + p_in_thread[s2];
                p_in_transform_thread[d3] = -p_in_thread[s3];
            }
        }
    }
}

template <class TFloat,
          class InTransThreadDesc,  //{NPerThread, CPerThread, InTileSizeH, InTileSizeW}
          class WeiTransThreadDesc, //{KPerThread, CPerThread, InTileSizeH, InTileSizeW}
          class OutTransThreadDesc, //{NPerThread, KPerThread, InTileSizeH, InTileSizeW}
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned S,
          unsigned R,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW>
__device__ void
threadwise_winograd_calculate_transformed_output(InTransThreadDesc,
                                                 TFloat* const __restrict__ p_in_transform_thread,
                                                 WeiTransThreadDesc,
                                                 TFloat* const __restrict__ p_wei_transform_thread,
                                                 OutTransThreadDesc,
                                                 TFloat* __restrict__ p_out_transform_thread)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_transform_thread_desc  = InTransThreadDesc{};
    constexpr auto wei_transform_thread_desc = WeiTransThreadDesc{};
    constexpr auto out_transform_thread_desc = OutTransThreadDesc{};

    for(unsigned n = 0; n < out_transform_thread_desc.GetLength(I0); ++n)
    {
        for(unsigned k = 0; k < out_transform_thread_desc.GetLength(I1); ++k)
        {
            for(unsigned h = 0; h < out_transform_thread_desc.GetLength(I2); ++h)
            {
                for(unsigned w = 0; w < out_transform_thread_desc.GetLength(I3); ++w)
                {
                    for(unsigned c = 0; c < wei_transform_thread_desc.GetLength(I1); ++c)
                    {
                        const unsigned in_index  = in_transform_thread_desc.Get1dIndex(n, c, h, w);
                        const unsigned wei_index = wei_transform_thread_desc.Get1dIndex(k, c, h, w);
                        const unsigned out_index = out_transform_thread_desc.Get1dIndex(n, k, h, w);

                        p_out_transform_thread[out_index] +=
                            p_wei_transform_thread[wei_index] * p_in_transform_thread[in_index];
                    }
                }
            }
        }
    }
}

template <class TFloat,
          class OutTransThreadDesc, //{NPerThread, KPerThread,  InTileSizeH,  InTileSizeW}
          class OutThreadDesc,      //{NPerThread, CPerThread, OutTileSizeH, OutTileSizeW}
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned S,
          unsigned R,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW>
__device__ void
threadwise_winograd_reverse_transform_output(OutTransThreadDesc,
                                             TFloat* const __restrict__ p_out_transform_thread,
                                             OutThreadDesc,
                                             TFloat* __restrict__ p_out_thread)
{
    static_assert(InTileSizeH == 4, "wrong");
    static_assert(InTileSizeW == 4, "wrong");
    static_assert(S == 3, "wrong");
    static_assert(R == 3, "wrong");
    static_assert(OutTileSizeH == 2, "wrong");
    static_assert(OutTileSizeW == 2, "wrong");

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto out_transform_thread_desc = OutTransThreadDesc{};
    constexpr auto out_thread_desc           = OutThreadDesc{};

    static_assert(InTileSizeH == out_transform_thread_desc.GetLength(I2), "wrong");
    static_assert(InTileSizeW == out_transform_thread_desc.GetLength(I3), "wrong");
    static_assert(OutTileSizeH == out_thread_desc.GetLength(I2), "wrong");
    static_assert(OutTileSizeW == out_thread_desc.GetLength(I3), "wrong");

    for(unsigned n = 0; n < out_thread_desc.GetLength(I0); ++n)
    {
        for(unsigned k = 0; k < out_thread_desc.GetLength(I1); ++k)
        {
            p_out_thread[out_thread_desc.Get1dIndex(n, k, 0, 0)] =
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 0)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 2)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 0)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 2)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 0)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 2)];

            p_out_thread[out_thread_desc.Get1dIndex(n, k, 0, 1)] =
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 0, 3)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 3)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 3)];

            p_out_thread[out_thread_desc.Get1dIndex(n, k, 1, 0)] =
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 0)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 0)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 0)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 2)];

            p_out_thread[out_thread_desc.Get1dIndex(n, k, 1, 1)] =
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 1)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 2)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 1, 3)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 2)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 2, 3)] -
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 1)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 2)] +
                p_out_transform_thread[out_transform_thread_desc.Get1dIndex(n, k, 3, 3)];
        }
    }
}