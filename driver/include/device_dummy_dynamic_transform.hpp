#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "dummy_dynamic_transform.hpp"

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_dummy_dynamic_transform(InDesc,
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

    const auto in_nchw_desc  = make_dynamic_native_tensor_descriptor(to_array(InDesc::GetLengths()),
                                                                    to_array(InDesc::GetStrides()));
    const auto wei_kcyx_desc = make_dynamic_native_tensor_descriptor(
        to_array(WeiDesc::GetLengths()), to_array(WeiDesc::GetStrides()));
    const auto out_nkhw_desc = make_dynamic_native_tensor_descriptor(
        to_array(OutDesc::GetLengths()), to_array(OutDesc::GetStrides()));

    const auto conv_strides   = to_array(ConvStrides{});
    const auto conv_dilations = to_array(ConvDilations{});
    const auto in_left_pads   = to_array(InLeftPads{});
    const auto in_right_pads  = to_array(InRightPads{});

    {
        const auto tensor_descs = map_convolution_into_gemm(wei_kcyx_desc,
                                                            in_nchw_desc,
                                                            out_nkhw_desc,
                                                            conv_strides,
                                                            conv_dilations,
                                                            in_left_pads,
                                                            in_right_pads);

        const auto in_gemmk_gemmn_global_desc = tensor_descs.At(Number<0>{});

        auto in_gemmk_gemmn_coord =
            make_dynamic_tensor_coordinate(in_gemmk_gemmn_global_desc, MultiIndex<2>{0, 0});

        for(index_t iter = 0; iter < 100; ++iter)
        {
            constexpr auto gemmk1_gemmn0 = MultiIndex<2>{1, 0};

            printf("iter %d\n", iter);

            print_array("idx0: ", in_gemmk_gemmn_coord.GetIndex());
            print_array("idx1: ", in_gemmk_gemmn_coord.GetLowerCoordinate().GetIndex());
            print_array("idx2: ",
                        in_gemmk_gemmn_coord.GetLowerCoordinate().GetLowerCoordinate().GetIndex());
            print_array("idx3: ",
                        in_gemmk_gemmn_coord.GetLowerCoordinate()
                            .GetLowerCoordinate()
                            .GetLowerCoordinate()
                            .GetIndex());
            print_array("idx4: ",
                        in_gemmk_gemmn_coord.GetLowerCoordinate()
                            .GetLowerCoordinate()
                            .GetLowerCoordinate()
                            .GetLowerCoordinate()
                            .GetIndex());
            printf("offset: %d\n", in_gemmk_gemmn_coord.GetOffset());

            printf("\n");

            in_gemmk_gemmn_coord += gemmk1_gemmn0;
        }
    }

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

    using dummy_transform = DummyDynamicTransform<BlockSize>;

    for(index_t i = 0; i < 5; ++i)
    {
        std::cout << "Start running " << nrepeat << " times..." << std::endl;

        KernelTimer timer;
        timer.Start();

        for(index_t j = 0; j < nrepeat; ++j)
        {
            launch_kernel(run_gridwise_operation<dummy_transform,
                                                 index_t* const,
                                                 float* const,
                                                 float* const,
                                                 const DynamicNativeTensorDescriptor<4>,
                                                 const DynamicNativeTensorDescriptor<4>,
                                                 const DynamicNativeTensorDescriptor<4>,
                                                 const Array<index_t, 2>,
                                                 const Array<index_t, 2>,
                                                 const Array<index_t, 2>,
                                                 const Array<index_t, 2>>,
                          dim3(GridSize),
                          dim3(BlockSize),
                          0,
                          0,
                          static_cast<index_t*>(wei_kcyx_device_buf.GetDeviceBuffer()),
                          static_cast<float*>(in_nchw_device_buf.GetDeviceBuffer()),
                          static_cast<float*>(out_nkhw_device_buf.GetDeviceBuffer()),
                          wei_kcyx_desc,
                          in_nchw_desc,
                          out_nkhw_desc,
                          conv_strides,
                          conv_dilations,
                          in_left_pads,
                          in_right_pads);
        }
    }

    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
