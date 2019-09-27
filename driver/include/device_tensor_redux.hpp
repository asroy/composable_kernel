#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_redux_kernel_wrapper.hpp"
//#include "gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw.hpp"
#include "gridwise_tensor_redux.hpp"

template <class T,
          class InDesc,
          class OutDesc
          >
void device_tensor_redux(InDesc,
        const Tensor<T>& in_nchw,
        OutDesc,
        Tensor<T>& out_nkhw,
        index_t nrepeat)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc  = InDesc{};
    constexpr auto out_nkhw_desc = OutDesc{};

    constexpr index_t C = in_nchw_desc.GetLength(I1);
    constexpr index_t Hi = in_nchw_desc.GetLength(I2);
    constexpr index_t Wi = in_nchw_desc.GetLength(I3);

    constexpr index_t N  = out_nkhw_desc.GetLength(I0);

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());
    constexpr index_t BlockSize = 256;
    constexpr auto GridSize = (N * C * Hi * Wi) / BlockSize;

    constexpr auto redux_dim = Sequence<I0>{};

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(index_t i = 0; i < nrepeat; ++i)
    {
        constexpr auto gridwise_redux =
            GridwiseTensorRedux_v1
            <GridSize,
             BlockSize,
             T,
             decltype(in_nchw_desc),
             decltype(out_nkhw_desc), decltype(redux_dim)
             >{};

        float time = launch_kernel(run_gridwise_redux_kernel<decltype(gridwise_redux), T>,
                                   dim3(GridSize),
                                   dim3(BlockSize),
                                   0,
                                   static_cast<const T*>(in_nchw_device_buf.GetDeviceBuffer()),
                                   static_cast<T*>(out_nkhw_device_buf.GetDeviceBuffer()));

        printf("Elapsed time : %f ms\n",
               time);
        usleep(std::min(time * 1000, float(10000)));
    }

    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
