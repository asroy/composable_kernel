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

    const auto in_nchw_desc = make_dynamic_native_tensor_descriptor<4>(
        to_multi_index(InDesc::GetLengths()), to_multi_index(InDesc::GetStrides()));
    const auto wei_kcyx_desc = make_dynamic_native_tensor_descriptor<4>(
        to_multi_index(WeiDesc::GetLengths()), to_multi_index(WeiDesc::GetStrides()));
    const auto out_nkhw_desc = make_dynamic_native_tensor_descriptor<4>(
        to_multi_index(OutDesc::GetLengths()), to_multi_index(OutDesc::GetStrides()));

    const auto conv_strides   = to_multi_index(ConvStrides{});
    const auto conv_dilations = to_multi_index(ConvDilations{});
    const auto in_left_pads   = to_multi_index(InLeftPads{});
    const auto in_right_pads  = to_multi_index(InRightPads{});

    const auto tensor_descs = map_convolution_into_gemm_fwd_v4r4(wei_kcyx_desc,
                                                                 in_nchw_desc,
                                                                 out_nkhw_desc,
                                                                 conv_strides,
                                                                 conv_dilations,
                                                                 in_left_pads,
                                                                 in_right_pads);

    const auto in_gemmk_gemmn_gemmkpack_global_desc = tensor_descs.At(Number<0>{});

    // test on cpu
    {
        auto in_gemmk_gemmn_gemmkpack_coord = make_dynamic_tensor_coordinate(
            in_gemmk_gemmn_gemmkpack_global_desc, make_multi_index(0, 0, 0));

        const auto in_gemmk_gemmn_gemmkpack_coord_step_0_0_1 = make_dynamic_tensor_coordinate_step(
            in_gemmk_gemmn_gemmkpack_global_desc, make_multi_index(0, 0, 1));

        print_array_v2("do_tansforms 0 0 1: ",
                       in_gemmk_gemmn_gemmkpack_coord_step_0_0_1.do_transforms_);

        for(index_t iter = 0; iter < 10; ++iter)
        {
            printf("iter %d\n", iter);
            print_array_v2("idx: ", in_gemmk_gemmn_gemmkpack_coord.GetIndex());
            print_array_v2("hidden idx: ", in_gemmk_gemmn_gemmkpack_coord.GetHiddenIndex());
            printf("offset: %d\n", in_gemmk_gemmn_gemmkpack_coord.GetOffset());
            printf("\n");

            move_dynamic_tensor_coordinate(in_gemmk_gemmn_gemmkpack_global_desc,
                                           in_gemmk_gemmn_gemmkpack_coord,
                                           in_gemmk_gemmn_gemmkpack_coord_step_0_0_1);
        }
    }

    {
        auto in_gemmk_gemmn_gemmkpack_coord = make_dynamic_tensor_coordinate(
            in_gemmk_gemmn_gemmkpack_global_desc, make_multi_index(0, 0, 0));

        const auto in_gemmk_gemmn_gemmkpack_coord_step_0_1_0 = make_dynamic_tensor_coordinate_step(
            in_gemmk_gemmn_gemmkpack_global_desc, make_multi_index(0, 1, 0));

        print_array_v2("do_tansforms 0 1 0: ",
                       in_gemmk_gemmn_gemmkpack_coord_step_0_1_0.do_transforms_);

        for(index_t iter = 0; iter < 10; ++iter)
        {
            printf("iter %d\n", iter);
            print_array_v2("idx: ", in_gemmk_gemmn_gemmkpack_coord.GetIndex());
            print_array_v2("hidden idx: ", in_gemmk_gemmn_gemmkpack_coord.GetHiddenIndex());
            printf("offset: %d\n", in_gemmk_gemmn_gemmkpack_coord.GetOffset());
            printf("\n");

            move_dynamic_tensor_coordinate(in_gemmk_gemmn_gemmkpack_global_desc,
                                           in_gemmk_gemmn_gemmkpack_coord,
                                           in_gemmk_gemmn_gemmkpack_coord_step_0_1_0);
        }
    }

    {
        auto in_gemmk_gemmn_gemmkpack_coord = make_dynamic_tensor_coordinate(
            in_gemmk_gemmn_gemmkpack_global_desc, make_multi_index(0, 0, 0));

        const auto in_gemmk_gemmn_gemmkpack_coord_step_1_0_0 = make_dynamic_tensor_coordinate_step(
            in_gemmk_gemmn_gemmkpack_global_desc, make_multi_index(1, 0, 0));

        print_array_v2("do_tansforms 1 0 0: ",
                       in_gemmk_gemmn_gemmkpack_coord_step_1_0_0.do_transforms_);

        for(index_t iter = 0; iter < 10; ++iter)
        {
            printf("iter %d\n", iter);
            print_array_v2("idx: ", in_gemmk_gemmn_gemmkpack_coord.GetIndex());
            print_array_v2("hidden idx: ", in_gemmk_gemmn_gemmkpack_coord.GetHiddenIndex());
            printf("offset: %d\n", in_gemmk_gemmn_gemmkpack_coord.GetOffset());
            printf("\n");

            move_dynamic_tensor_coordinate(in_gemmk_gemmn_gemmkpack_global_desc,
                                           in_gemmk_gemmn_gemmkpack_coord,
                                           in_gemmk_gemmn_gemmkpack_coord_step_1_0_0);
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

    for(index_t i = 0; i < 5; ++i)
    {
        std::cout << "Start running " << nrepeat << " times..." << std::endl;

        KernelTimer timer;
        timer.Start();

        for(index_t j = 0; j < nrepeat; ++j)
        {
#if 0
            launch_kernel(run_gridwise_operation<DummyDynamicTransform_1<BlockSize>,
                                                 index_t* const,
                                                 float* const,
                                                 float* const,
                                                 const decltype(wei_kcyx_desc),
                                                 const decltype(in_nchw_desc),
                                                 const decltype(out_nkhw_desc),
                                                 const MultiIndex<2>,
                                                 const MultiIndex<2>,
                                                 const MultiIndex<2>,
                                                 const MultiIndex<2>>,
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
#else
            launch_kernel(
                run_gridwise_operation<DummyDynamicTransform_fwd_v4r4<BlockSize>,
                                       index_t* const,
                                       float* const,
                                       float* const,
                                       const decltype(in_gemmk_gemmn_gemmkpack_global_desc)>,
                dim3(GridSize),
                dim3(BlockSize),
                0,
                0,
                static_cast<index_t*>(wei_kcyx_device_buf.GetDeviceBuffer()),
                static_cast<float*>(in_nchw_device_buf.GetDeviceBuffer()),
                static_cast<float*>(out_nkhw_device_buf.GetDeviceBuffer()),
                in_gemmk_gemmn_gemmkpack_global_desc);
#endif
        }
    }

    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
