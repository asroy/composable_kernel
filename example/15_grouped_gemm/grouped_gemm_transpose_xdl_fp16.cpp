#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

#include "check_err.hpp"
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_grouped_gemm_transpose_xdl.hpp"
#include "element_wise_operation.hpp"
#include "reference_gemm_transpose.hpp"
#include "gemm_specialization.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
// static constexpr auto GemmMNPadding =
// ck::tensor_operation::device::GemmSpecialization::MNPadding;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGroupedGemmTransposeXdl
//######| AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|      Num|
//######|  Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|Spacialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar| Prefetch|
//######|      |      |      |        |        |        |        |   Operation|   Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|         |
//######|      |      |      |        |        |        |        |            |            |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |         |
        <   F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   256,   256,   128,     4,  8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1,        1>;
// clang-format on

using ReferenceGemmTransposeInstance = ck::tensor_operation::host::
    ReferenceGemmTranspose<ADataType, BDataType, CDataType, AElementOp, BElementOp, CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        exit(0);
    }

    int group_count = 4;

    // GEMM shape
    std::vector<ck::tensor_operation::device::GemmTransposeDesc> gemm_descs;
    std::vector<const void*> p_a, p_b;
    std::vector<void*> p_c;

    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; i++)
    {
        int B       = 16;
        int S       = 64;
        int NumHead = 16;
        int HeadDim = 64;

        int M0 = B;
        int M1 = S;
        int N0 = NumHead;
        int N1 = HeadDim;

        int M = M0 * N1;
        int N = N0 * N1;
        int K = NumHead * HeadDim;

        int StrideA = K;
        int StrideB = K;

        if(i % 2 == 0)
        {

            int StrideM0 = S * NumHead * HeadDim;
            int StrideM1 = 1;
            int StrideN0 = S * HeadDim;
            int StrideN1 = S;

            gemm_descs.push_back({M,
                                  N,
                                  K,
                                  StrideA,
                                  StrideB,
                                  M0,
                                  M1,
                                  N0,
                                  N1,
                                  StrideM0,
                                  StrideM1,
                                  StrideN0,
                                  StrideN1});
        }
        else
        {

            int StrideM0 = S * NumHead * HeadDim;
            int StrideM1 = HeadDim;
            int StrideN0 = S * HeadDim;
            int StrideN1 = 1;

            gemm_descs.push_back({M,
                                  N,
                                  K,
                                  StrideA,
                                  StrideB,
                                  M0,
                                  M1,
                                  N0,
                                  N1,
                                  StrideM0,
                                  StrideM1,
                                  StrideN0,
                                  StrideN1});
        }
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
        };

    auto f_host_c_tensor_descriptor = [](std::size_t M0,
                                         std::size_t M1,
                                         std::size_t N0,
                                         std::size_t N1,
                                         std::size_t StrideM0,
                                         std::size_t StrideM1,
                                         std::size_t StrideN0,
                                         std::size_t StrideN1) {
        return HostTensorDescriptor(
            std::vector<std::size_t>({M0, M1, N0, N1}),
            std::vector<std::size_t>({StrideM0, StrideM1, StrideN0, StrideN1}));
    };

    std::vector<Tensor<ADataType>> a_tensors;
    std::vector<Tensor<BDataType>> b_tensors;
    std::vector<Tensor<CDataType>> c_host_tensors;
    std::vector<Tensor<CDataType>> c_device_tensors;

    a_tensors.reserve(group_count);
    b_tensors.reserve(group_count);
    c_host_tensors.reserve(group_count);
    c_device_tensors.reserve(group_count);

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a_tensors_device, b_tensors_device, c_tensors_device;

    a_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    c_tensors_device.reserve(group_count);

    std::size_t flop = 0, num_btype = 0;

    for(std::size_t i = 0; i < gemm_descs.size(); i++)
    {
        a_tensors.push_back(Tensor<ADataType>(f_host_tensor_descriptor(
            gemm_descs[i].M, gemm_descs[i].K, gemm_descs[i].StrideA, ALayout{})));
        b_tensors.push_back(Tensor<BDataType>(f_host_tensor_descriptor(
            gemm_descs[i].K, gemm_descs[i].N, gemm_descs[i].StrideB, BLayout{})));
        c_host_tensors.push_back(
            Tensor<CDataType>(f_host_c_tensor_descriptor(gemm_descs[i].M0,
                                                         gemm_descs[i].M1,
                                                         gemm_descs[i].N0,
                                                         gemm_descs[i].N1,
                                                         gemm_descs[i].StrideM0,
                                                         gemm_descs[i].StrideM1,
                                                         gemm_descs[i].StrideN0,
                                                         gemm_descs[i].StrideN1)));
        c_device_tensors.push_back(
            Tensor<CDataType>(f_host_c_tensor_descriptor(gemm_descs[i].M0,
                                                         gemm_descs[i].M1,
                                                         gemm_descs[i].N0,
                                                         gemm_descs[i].N1,
                                                         gemm_descs[i].StrideM0,
                                                         gemm_descs[i].StrideM1,
                                                         gemm_descs[i].StrideN0,
                                                         gemm_descs[i].StrideN1)));

        std::cout << "gemm[" << i << "] a_m_k: " << a_tensors[i].mDesc
                  << " b_k_n: " << b_tensors[i].mDesc << " c_m_n: " << c_device_tensors[i].mDesc
                  << std::endl;

        flop += std::size_t(2) * gemm_descs[i].M * gemm_descs[i].K * gemm_descs[i].N;
        num_btype += sizeof(ADataType) * a_tensors[i].mDesc.GetElementSize() +
                     sizeof(BDataType) * b_tensors[i].mDesc.GetElementSize() +
                     sizeof(CDataType) * c_device_tensors[i].mDesc.GetElementSize();

        switch(init_method)
        {
        case 0: break;
        case 1:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
            break;
        case 2:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
            break;
        default:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<0>{});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<1>{});
        }
    }

    for(std::size_t i = 0; i < gemm_descs.size(); i++)
    {
        a_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(ADataType) * a_tensors[i].mDesc.GetElementSpace()));
        b_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(BDataType) * b_tensors[i].mDesc.GetElementSpace()));
        c_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(CDataType) * c_device_tensors[i].mDesc.GetElementSpace()));

        a_tensors_device[i]->ToDevice(a_tensors[i].mData.data());
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data());

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b.push_back(b_tensors_device[i]->GetDeviceBuffer());
        p_c.push_back(c_tensors_device[i]->GetDeviceBuffer());
    }

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    // do GEMM
    auto argument =
        gemm.MakeArgument(p_a, p_b, p_c, gemm_descs, a_element_op, b_element_op, c_element_op);

    DeviceMem gemm_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument, gemm_desc_workspace.GetDeviceBuffer());

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;
    if(do_verification)
    {
        for(std::size_t i = 0; i < gemm_descs.size(); i++)
        {
            c_tensors_device[i]->FromDevice(c_device_tensors[i].mData.data());
            auto ref_gemm    = ReferenceGemmTransposeInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(a_tensors[i],
                                                      b_tensors[i],
                                                      c_host_tensors[i],
                                                      a_element_op,
                                                      b_element_op,
                                                      c_element_op);

            ref_invoker.Run(ref_argument);
            pass &= ck::utils::check_err(c_device_tensors[i].mData, c_host_tensors[i].mData);
        }
    }

    return pass ? 0 : 1;
}