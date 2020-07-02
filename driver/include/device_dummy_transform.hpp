#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "dummy_static_transform.hpp"

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_dummy_transform(InDesc,
                            const Tensor<T>& in_nchw,
                            WeiDesc,
                            const Tensor<T>& wei_kcyx,
                            OutDesc,
                            Tensor<T>& out_nkhw,
                            ConvStrides,
                            ConvDilations,
                            InLeftPads,
                            InRightPads,
                            ck::index_t nrepeat)
{
    using namespace ck;

    using TDevice = typename conditional<is_same<half_float::half, T>::value, half_t, T>::type;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc =
        make_native_tensor_descriptor(InDesc::GetLengths(), InDesc::GetStrides());
    constexpr auto wei_kcyx_desc =
        make_native_tensor_descriptor(WeiDesc::GetLengths(), WeiDesc::GetStrides());
    constexpr auto out_nkhw_desc =
        make_native_tensor_descriptor(OutDesc::GetLengths(), OutDesc::GetStrides());

    constexpr index_t N  = out_nkhw_desc.GetLength(I0);
    constexpr index_t K  = out_nkhw_desc.GetLength(I1);
    constexpr index_t Ho = out_nkhw_desc.GetLength(I2);
    constexpr index_t Wo = out_nkhw_desc.GetLength(I3);

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_kcyx_device_buf(data_sz * wei_kcyx.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_kcyx_device_buf.ToDevice(wei_kcyx.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

    constexpr index_t BlockSize = 256;
    constexpr index_t GridSize  = 1;

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    using dummy_transform = DummyStaticTransform<GridSize,
                                                 BlockSize,
                                                 float,
                                                 decltype(in_nchw_desc),
                                                 decltype(wei_kcyx_desc),
                                                 decltype(out_nkhw_desc),
                                                 ConvStrides,
                                                 ConvDilations,
                                                 InLeftPads,
                                                 InRightPads>;

    for(index_t i = 0; i < 5; ++i)
    {
        std::cout << "Start running " << nrepeat << " times..." << std::endl;

        KernelTimer timer;
        timer.Start();

        for(index_t j = 0; j < nrepeat; ++j)
        {
            launch_kernel(run_gridwise_operation<dummy_transform,
                                                 float* const __restrict__,
                                                 float* const __restrict__,
                                                 float* const __restrict__>,
                          dim3(GridSize),
                          dim3(BlockSize),
                          0,
                          0,
                          static_cast<float*>(in_nchw_device_buf.GetDeviceBuffer()),
                          static_cast<float*>(wei_kcyx_device_buf.GetDeviceBuffer()),
                          static_cast<float*>(out_nkhw_device_buf.GetDeviceBuffer()));
        }
    }

    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
