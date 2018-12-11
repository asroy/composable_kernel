#pragma once
#include "constant_tensor_descriptor.cuh"

template <class TFloat,
          class InBlockDesc, // {NPerBlock, CPerBlock, YPerBlock * (InTileSizeH - (S - 1)) + S - 1,
                             // WPerBlock}
          class InTransBlockDesc, // {NPerBlock, CPerBlock, YPerBlock * InTileSizeH, WPerBlock}
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned S,
          unsigned R,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned NPerBlock,
          unsigned CPerBlock,
          unsigned YPerBlock,
          unsigned WPerBlock,
          unsigned NPerThread,
          unsigned CPerThread,
          unsigned YPerThread,
          unsigned WPerThread,
          unsigned BlockSize>
__device__ void
blockwise_winograd_vertical_transform_input(InBlockDesc,
                                            TFloat* const __restrict__ p_in_block,
                                            InTransBlockDesc,
                                            TFloat* __restrict__ p_in_transform_block)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_block_desc           = InBlockDesc{};
    constexpr auto in_transform_block_desc = InTransBlockDesc{};

    constexpr unsigned NThreadWork = (NPerBlock + NPerThread - 1) / NPerThread;
    constexpr unsigned CThreadWork = (CPerBlock + CPerThread - 1) / CPerThread;
    constexpr unsigned YThreadWork = (YPerBlock + YPerThread - 1) / YPerThread;
    constexpr unsigned WThreadWork = (WPerBlock + WPerThread - 1) / WPerThread;

    for(unsigned thread_work_id = threadIdx.x;
        thread_work_id < NThreadWork * CThreadWork * YThreadWork * WThreadWork;
        thread_work_id += BlockSize)
    {
        unsigned itmp             = thread_work_id;
        unsigned n_thread_work_id = itmp / (CThreadWork * YThreadWork * WThreadWork);
        itmp -= n_thread_work_id * (CThreadWork * YThreadWork * WThreadWork);
        unsigned c_thread_work_id = itmp / (YThreadWork * WThreadWork);
        itmp -= c_thread_work_id * (YThreadWork * WThreadWork);
        unsigned y_thread_work_id = itmp / WThreadWork;
        unsigned w_thread_work_id = itmp - y_thread_work_id * WThreadWork;

        unsigned n_thread_data_begin = n_thread_work_id * NPerThread;
        unsigned c_thread_data_begin = c_thread_work_id * CPerThread;
        unsigned y_thread_data_begin = y_thread_work_id * YPerThread;
        unsigned w_thread_data_begin = w_thread_work_id * WPerThread;

        unsigned h_thread_data_begin = y_thread_data_begin * (InTileSizeH - (S - 1));

        // this contains a whole tile height
        constexpr auto in_thread_desc =
            make_ConstantTensorDescriptor(Sequence<1, 1, InTileSizeH, WPerThread>{});

        constexpr auto in_thread_block_desc =
            make_ConstantTensorDescriptor(in_thread_desc.GetLengths(), in_block_desc.GetStrides{});

        constexpr unsigned in_thread_size = in_thread_desc.GetElementSpace();

        TFloat p_in_thread[in_thread_size];

        // this contains a newly read tile part
        constexpr unsigned InTileReuseH   = S - 1;
        constexpr unsigned InTileNewReadH = InTileSizeH - (S - 1);

        constexpr auto in_newread_thread_desc =
            make_ConstantTensorDescriptor(Sequence<1, 1, InTileNewReadH, WPerThread>{});

        constexpr auto in_newread_thread_block_desc = make_ConstantTensorDescriptor(
            in_newread_thread_desc.GetLengths(), in_block_desc.GetStrides{});

        for(unsigned n = 0; n < NPerThread; ++n)
        {
            unsigned n_thread_data = n_thread_data_begin + n;

            for(unsigned c = 0; c < CPerThread; ++c)
            {
                unsigned c_thread_data = c_thread_data_begin + c;

                // read first tile
                threadwise_tensor_copy(
                    in_thread_block_desc,
                    p_in_block + in_block_desc.Get1dIndex(n_thread_data, c_thread_data, 0, 0),
                    in_thread_desc,
                    p_in_thread);

                // vertically transform first tile
                threadwise_winograd_vertical_transform_input<TFloat,
                                                             decltype(in_thread_desc),
                                                             decltype(in_transform_thread_desc),
                                                             InTileSizeH,
                                                             InTileSizeW,
                                                             S,
                                                             R,
                                                             OutTileSizeH,
                                                             OutTileSizeW>(
                    in_thread_desc, p_in_thread, in_thread_transform_desc, p_in_transform_thread);

                // write first vertically transformed tile
                threadwise_tensor_copy(
                    in_transform_thread_desc,
                    p_in_transform_thread,
                    in_transform_block_desc,
                    p_in_transform_block +
                        in_transform_block_desc.Get1dIndex(
                            n_thread_data, c_thread_data, h_thread_data_begin, 0));

                // next tile
                for(unsigned y = 1; y < YPerBlock; ++y)
                {
                    unsigned h_thread_data = h_thread_data_begin + y * InTileNewReadH;

                    // shift down to reuse data
                    threadwise_4d_tensor_shift_down(
                        in_thread_desc, p_in_thread, I2, InTileNewReadH);

                    // read new data
                    threadwise_4d_tensor_copy(
                        in_thread_block_desc,
                        p_in_block +
                            in_block_desc.Get1dIndex(
                                n_thread_data, c_thread_data, h_thread_data + InTileReuseH, 0),
                        in_newread_thread_desc,
                        p_in_thread + in_thread_desc.Get1dIndex(0, 0, InTileReuseH, 0));

                    // vertical transform tile
                    threadwise_winograd_vertical_transform_input<TFloat,
                                                                 decltype(in_thread_desc),
                                                                 decltype(in_transform_thread_desc),
                                                                 InTileSizeH,
                                                                 InTileSizeW,
                                                                 S,
                                                                 R,
                                                                 OutTileSizeH,
                                                                 OutTileSizeW>(
                        in_thread_desc,
                        p_in_thread,
                        in_thread_transform_desc,
                        p_in_transform_thread);

                    // write vertically transformed tile
                    threadwise_tensor_copy(in_transform_thread_desc,
                                           p_in_transform_thread,
                                           in_transform_block_desc,
                                           p_in_transform_block +
                                               in_transform_block_desc.Get1dIndex(
                                                   n_thread_data, c_thread_data, h_thread_data, 0));
                }
            }
        }
    }
}

template <class TFloat,
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned S,
          unsigned R,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned NPerBlock,
          unsigned CPerBlock,
          unsigned XPerBlock,
          unsigned HPerBlock,
          unsigned NPerThread,
          unsigned CPerThread,
          unsigned XPerThread,
          unsigned HPerThread,
          unsigned BlockSize>
__device__ void blockwise_winograd_horizontal_transform_input(TFloat* const __restrict__ p_in,
                                                              TFloat* __restrict__ p_in_transform)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_block_desc           = InBlockDesc{};
    constexpr auto in_transform_block_desc = InTransBlockDesc{};

    constexpr unsigned NThreadWork = (NPerBlock + NPerThread - 1) / NPerThread;
    constexpr unsigned CThreadWork = (CPerBlock + CPerThread - 1) / CPerThread;
    constexpr unsigned HThreadWork = (YPerBlock + YPerThread - 1) / YPerThread;
    constexpr unsigned WThreadWork = (WPerBlock + WPerThread - 1) / WPerThread;

    for(unsigned thread_work_id = threadIdx.x;
        thread_work_id < NThreadWork * CThreadWork * YThreadWork * WThreadWork;
        thread_work_id += BlockSize)
    {
        unsigned itmp             = thread_work_id;
        unsigned n_thread_work_id = itmp / (CThreadWork * YThreadWork * WThreadWork);
        itmp -= n_thread_work_id * (CThreadWork * YThreadWork * WThreadWork);
        unsigned c_thread_work_id = itmp / (YThreadWork * WThreadWork);
        itmp -= c_thread_work_id * (YThreadWork * WThreadWork);
        unsigned y_thread_work_id = itmp / WThreadWork;
        unsigned w_thread_work_id = itmp - y_thread_work_id * WThreadWork;

        unsigned n_thread_data_begin = n_thread_work_id * NPerThread;
        unsigned c_thread_data_begin = c_thread_work_id * CPerThread;
        unsigned y_thread_data_begin = y_thread_work_id * YPerThread;
        unsigned w_thread_data_begin = w_thread_work_id * WPerThread;

        unsigned h_thread_data_begin = y_thread_data_begin * (InTileSizeH - (S - 1));

        // this contains a whole tile height
        constexpr auto in_thread_desc =
            make_ConstantTensorDescriptor(Sequence<1, 1, InTileSizeH, WPerThread>{});

        constexpr auto in_thread_block_desc =
            make_ConstantTensorDescriptor(in_thread_desc.GetLengths(), in_block_desc.GetStrides{});

        constexpr unsigned in_thread_size = in_thread_desc.GetElementSpace();

        TFloat p_in_thread[in_thread_size];

        // this contains a newly read tile part
        constexpr unsigned InTileReuseH   = S - 1;
        constexpr unsigned InTileNewReadH = InTileSizeH - (S - 1);

        constexpr auto in_newread_thread_desc =
            make_ConstantTensorDescriptor(Sequence<1, 1, InTileNewReadH, WPerThread>{});

        constexpr auto in_newread_thread_block_desc = make_ConstantTensorDescriptor(
            in_newread_thread_desc.GetLengths(), in_block_desc.GetStrides{});

        for(unsigned n = 0; n < NPerThread; ++n)
        {
            unsigned n_thread_data = n_thread_data_begin + n;

            for(unsigned c = 0; c < CPerThread; ++c)
            {
                unsigned c_thread_data = c_thread_data_begin + c;

                // read first tile
                threadwise_tensor_copy(
                    in_thread_block_desc,
                    p_in_block + in_block_desc.Get1dIndex(n_thread_data, c_thread_data, 0, 0),
                    in_thread_desc,
                    p_in_thread);

                // vertically transform first tile
                threadwise_winograd_vertical_transform_input<TFloat,
                                                             decltype(in_thread_desc),
                                                             decltype(in_transform_thread_desc),
                                                             InTileSizeH,
                                                             InTileSizeW,
                                                             S,
                                                             R,
                                                             OutTileSizeH,
                                                             OutTileSizeW>(
                    in_thread_desc, p_in_thread, in_thread_transform_desc, p_in_transform_thread);

                // write first vertically transformed tile
                threadwise_tensor_copy(
                    in_transform_thread_desc,
                    p_in_transform_thread,
                    in_transform_block_desc,
                    p_in_transform_block +
                        in_transform_block_desc.Get1dIndex(
                            n_thread_data, c_thread_data, h_thread_data_begin, 0));

                // next tile
                for(unsigned y = 1; y < YPerBlock; ++y)
                {
                    unsigned h_thread_data = h_thread_data_begin + y * InTileNewReadH;

                    // shift down to reuse data
                    threadwise_4d_tensor_shift_down(
                        in_thread_desc, p_in_thread, I2, InTileNewReadH);

                    // read new data
                    threadwise_4d_tensor_copy(
                        in_thread_block_desc,
                        p_in_block +
                            in_block_desc.Get1dIndex(
                                n_thread_data, c_thread_data, h_thread_data + InTileReuseH, 0),
                        in_newread_thread_desc,
                        p_in_thread + in_thread_desc.Get1dIndex(0, 0, InTileReuseH, 0));

                    // vertical transform tile
                    threadwise_winograd_vertical_transform_input<TFloat,
                                                                 decltype(in_thread_desc),
                                                                 decltype(in_transform_thread_desc),
                                                                 InTileSizeH,
                                                                 InTileSizeW,
                                                                 S,
                                                                 R,
                                                                 OutTileSizeH,
                                                                 OutTileSizeW>(
                        in_thread_desc,
                        p_in_thread,
                        in_thread_transform_desc,
                        p_in_transform_thread);

                    // write vertically transformed tile
                    threadwise_tensor_copy(in_transform_thread_desc,
                                           p_in_transform_thread,
                                           in_transform_block_desc,
                                           p_in_transform_block +
                                               in_transform_block_desc.Get1dIndex(
                                                   n_thread_data, c_thread_data, h_thread_data, 0));
                }
            }
        }
    }
}

template <class TFloat,
          unsigned InTileSizeH,
          unsigned InTileSizeW,
          unsigned S,
          unsigned R,
          unsigned OutTileSizeH,
          unsigned OutTileSizeW,
          unsigned KPerBlock,
          unsigned CPerBlock,
          unsigned BlockSize>
__device__ void blockwise_winograd_transform_weight(TFloat* const __restrict__ p_wei,
                                                    TFloat* __restrict__ p_wei_transform)
{
    p_wei_transform[0] = 1;
}