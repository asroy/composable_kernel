#include <iostream>
#include <fstream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <half.hpp>
#include <getopt.h>

#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
#include "online_device_dynamic_generic_reduction.hpp"
#include "online_driver_common.hpp"
#include "online_reduce_common.hpp"
#include "host_generic_reduction.hpp"

#include "handle.hpp"
#include "hipCheck.hpp"

using namespace std;

static struct option long_options[] = {{"inLengths", required_argument, NULL, 'D'},
                                       {"toReduceDims", required_argument, NULL, 'R'},
                                       {"reduceOp", required_argument, NULL, 'O'},
                                       {"compType", required_argument, NULL, 'C'},
                                       {"nanOpt", required_argument, NULL, 'N'},
                                       {"indicesOpt", required_argument, NULL, 'I'},
                                       {"scales", required_argument, NULL, 'S'},
                                       {"half", no_argument, NULL, '?'},
                                       {"double", no_argument, NULL, '?'},
                                       {"dumpout", required_argument, NULL, 'o'},
                                       {"verify", required_argument, NULL, 'v'},
                                       {"log", required_argument, NULL, 'l'},
                                       {"help", no_argument, NULL, '?'},
                                       {0, 0, 0, 0}};
static int option_index             = 0;

template <typename T>
static T getSingleValueFromString(const string& valueStr);

template <>
int getSingleValueFromString<int>(const string& valueStr)
{
    return (std::stoi(valueStr));
};

template <>
size_t getSingleValueFromString<size_t>(const string& valueStr)
{
    return (std::stol(valueStr));
};

template <>
float getSingleValueFromString<float>(const string& valueStr)
{
    return (std::stof(valueStr));
};

template <typename T>
static std::vector<T> getTypeValuesFromString(const char* cstr_values)
{
    std::string valuesStr(cstr_values);

    std::vector<T> values;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = valuesStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        const std::string sliceStr = valuesStr.substr(pos, new_pos - pos);

        T val = getSingleValueFromString<T>(sliceStr);

        values.push_back(val);

        pos     = new_pos + 1;
        new_pos = valuesStr.find(',', pos);
    };

    std::string sliceStr = valuesStr.substr(pos);
    T val                = getSingleValueFromString<T>(sliceStr);

    values.push_back(val);

    return (values);
}

static void show_usage(const char* cmd)
{
    std::cout << "Usage of " << cmd << std::endl;
    std::cout << "--inLengths or -D, comma separated list of input tensor dimension lengths"
              << std::endl;
    std::cout << "--toReduceDims or -R, comma separated list of to-reduce dimensions" << std::endl;
    std::cout << "--reduceOp or -O, enum value indicating the reduction operations" << std::endl;
    std::cout << "--compType or -C, enum value indicating the type of accumulated values used "
                 "during the reduction"
              << std::endl;
    std::cout << "--nanOpt or -N, enum value indicates the selection for NanOpt" << std::endl;
    std::cout << "--indicesOpt or -I, enum value indicates the selection for IndicesOpt"
              << std::endl;
    std::cout << "--scales or -S, comma separated two float values for alpha and beta" << std::endl;
    std::cout << "--half, use fp16 for the input and output tensor data types" << std::endl;
    std::cout << "--double, use fp64 for the input and output tensor data types" << std::endl;
    std::cout << "--verify or -v, 1/0 to indicate whether to verify the reduction result by "
                 "comparing with the host-based reduction"
              << std::endl;
    std::cout << "--dumpout or -v, 1/0 to indicate wheter to save the reduction result to files "
                 "for further analysis"
              << std::endl;
    std::cout << "--log or -l, 1/0 to indicate whether to log some information" << std::endl;
};

static void check_reduce_dims(const int totalDims, const std::vector<int>& toReduceDims)
{
    for(const auto dim : toReduceDims)
    {
        if(dim < 0 || dim >= totalDims)
            throw std::runtime_error("Invalid dimension index specified for Reducing");
    };
};

static vector<int> get_invariant_dims(int totalDims, const vector<int>& toReduceDims)
{
    vector<int> resDims;
    unsigned int incFlag = 0;

    for(auto dim : toReduceDims)
        incFlag = incFlag | (0x1 << dim);

    for(int dim = 0; dim < totalDims; dim++)
    {
        if(incFlag & (0x1 << dim))
            continue;
        resDims.push_back(dim);
    };

    return (resDims);
};

static bool use_half   = false;
static bool use_double = false;

static vector<size_t> inLengths;
static vector<size_t> outLengths;
static vector<int> toReduceDims;
static vector<int> invariantDims;

static vector<float> scales;

static ReduceTensorOp_t reduceOp        = ReduceTensorOp_t::REDUCE_TENSOR_ADD;
static appDataType_t compTypeId         = appFloat;
static bool compType_assigned           = false;
static NanPropagation_t nanOpt          = NanPropagation_t::NOT_PROPAGATE_NAN;
static ReduceTensorIndices_t indicesOpt = ReduceTensorIndices_t::REDUCE_TENSOR_NO_INDICES;
static bool do_logging                  = false;
static bool do_verification             = false;
static bool do_dumpout                  = false;

static int init_method;
static int nrepeat;

static bool need_indices = false;

static void check_cmdline_arguments(int argc, char* argv[])
{
    unsigned int ch;

    while(1)
    {
        ch = getopt_long(argc, argv, "D:R:O:C:N:I:S:v:o:l:", long_options, &option_index);
        if(ch == -1)
            break;
        switch(ch)
        {
        case 'D':
            if(!optarg)
                throw std::runtime_error("Invalid option format!");

            inLengths = getTypeValuesFromString<size_t>(optarg);
            break;
        case 'R':
            if(!optarg)
                throw std::runtime_error("Invalid option format!");

            toReduceDims = getTypeValuesFromString<int>(optarg);
            break;
        case 'O':
            if(!optarg)
                throw std::runtime_error("Invalid option format!");

            reduceOp = static_cast<ReduceTensorOp_t>(std::atoi(optarg));
            break;
        case 'C':
            if(!optarg)
                throw std::runtime_error("Invalid option format!");

            compTypeId        = static_cast<appDataType_t>(std::atoi(optarg));
            compType_assigned = true;
            break;
        case 'N':
            if(!optarg)
                throw std::runtime_error("Invalid option format!");

            nanOpt = static_cast<NanPropagation_t>(std::atoi(optarg));
            break;
        case 'I':
            if(!optarg)
                throw std::runtime_error("Invalid option format!");

            indicesOpt = static_cast<ReduceTensorIndices_t>(std::atoi(optarg));
            break;
        case 'S':
            if(!optarg)
                throw std::runtime_error("Invalid option format!");

            scales = getTypeValuesFromString<float>(optarg);

            if(scales.size() != 2)
                throw std::runtime_error("Invalid option format!");

            break;
        case 'v':
            if(!optarg)
                throw std::runtime_error("Invalid option format!");

            do_verification = static_cast<bool>(std::atoi(optarg));
            break;
        case 'o':
            if(!optarg)
                throw std::runtime_error("Invalid option format!");

            do_dumpout = static_cast<bool>(std::atoi(optarg));
            break;
        case 'l':
            if(!optarg)
                throw std::runtime_error("Invalid option format!");

            do_logging = static_cast<bool>(std::atoi(optarg));
            break;
        case '?':
            if(std::string(long_options[option_index].name) == "half")
                use_half = true;
            else if(std::string(long_options[option_index].name) == "double")
                use_double = true;
            else if(std::string(long_options[option_index].name) == "help")
            {
                show_usage(argv[0]);
                exit(0);
            };
            break;

        default: show_usage(argv[0]); throw std::runtime_error("Invalid cmd-line options!");
        };
    };

    if(optind + 2 > argc)
        throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

    init_method = std::atoi(argv[optind++]);
    nrepeat     = std::atoi(argv[optind]);

    if(scales.empty())
    {
        scales.push_back(1.0f);
        scales.push_back(0.0f);
    };

    if((reduceOp == ReduceTensorOp_t::REDUCE_TENSOR_MIN ||
        reduceOp == ReduceTensorOp_t::REDUCE_TENSOR_MAX ||
        reduceOp == ReduceTensorOp_t::REDUCE_TENSOR_AMAX) &&
       indicesOpt != ReduceTensorIndices_t::REDUCE_TENSOR_NO_INDICES)
        need_indices = true;
};

template <typename T>
static void dumpBufferToFile(const char* fileName, T* data, size_t dataNumItems)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if(outFile)
    {
        outFile.write(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        outFile.close();
        std::cout << "Write output to file " << fileName << std::endl;
    }
    else
    {
        std::cout << "Could not open file " << fileName << " for writing" << std::endl;
    }
}

static void check_indices(const Tensor<int>& ref, const Tensor<int>& result)
{
    for(int i = 0; i < ref.mData.size(); ++i)
    {
        if(ref.mData[i] != result.mData[i])
        {
            std::cerr << std::endl
                      << "Indices different at position " << i << " (ref: " << ref.mData[i]
                      << ", result: " << result.mData[i] << ")" << std::endl;
            break;
        };
    }

    std::cout << std::endl << "Indices result is completely acccurate!" << std::endl;
}

template <typename dataType, typename compType>
static void do_reduce_testing(olCompile::Handle* handle);

int main(int argc, char* argv[])
{
    using namespace ck;
    using half = half_float::half;

    check_cmdline_arguments(argc, argv);

    hipStream_t stream;
    olCompile::Handle* handle;

    MY_HIP_CHECK(hipStreamCreate(&stream));

    handle = new olCompile::Handle(stream);

    check_reduce_dims(inLengths.size(), toReduceDims);

    invariantDims = get_invariant_dims(inLengths.size(), toReduceDims);

    if(invariantDims.empty())
    {
        outLengths.push_back(1);
    }
    else
    {
        for(auto dim : invariantDims)
            outLengths.push_back(inLengths[dim]);
    };

    if(use_half)
    {
        if(!compType_assigned)
            compTypeId = appHalf;

        if(compTypeId == appHalf)
            do_reduce_testing<half_float::half, half_float::half>(handle);
        else if(compTypeId == appFloat)
            do_reduce_testing<half_float::half, float>(handle);
        else
            throw std::runtime_error("Invalid compType assignment!");
    }
    else if(use_double)
        do_reduce_testing<double, double>(handle);
    else
    {
        if(compTypeId == appFloat)
            do_reduce_testing<float, float>(handle);
        else if(compTypeId == appDouble)
            do_reduce_testing<float, double>(handle);
        else
            throw std::runtime_error("Invalid compType assignment!");
    };

    delete handle;
    MY_HIP_CHECK(hipStreamDestroy(stream));
};

template <typename dataType, typename compType>
static void do_reduce_testing(olCompile::Handle* handle)
{
    Tensor<dataType> in(inLengths);
    Tensor<dataType> out_host(outLengths);
    Tensor<dataType> out_dev(outLengths);
    Tensor<int> out_indices_host(outLengths);
    Tensor<int> out_indices_dev(outLengths);

    // ostream_HostTensorDescriptor(in.mDesc, std::cout << "in: ");
    // ostream_HostTensorDescriptor(out_host.mDesc, std::cout << "out: ");

    std::size_t num_thread = std::thread::hardware_concurrency();

    float alpha = scales[0];
    float beta  = scales[1];

    if(do_verification)
    {
        switch(init_method)
        {
        case 0:
            in.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            if(beta != 0.0f)
                out_host.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            break;
        case 1:
            in.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            if(beta != 0.0f)
                out_host.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            break;
        default:
            in.GenerateTensorValue(GeneratorTensor_2{1, 5}, num_thread);
            if(beta != 0.0f)
                out_host.GenerateTensorValue(GeneratorTensor_2{1, 5}, num_thread);
        }

        if(beta != 0.0f)
            for(size_t i = 0; i < out_host.mDesc.GetElementSpace(); i++)
                out_dev.mData[i] = out_host.mData[i];
    }

    tunable_dyn_generic_reduction* tunable = &default_tunable_dyn_generic_reduction;

    device_dynamic_generic_reduction_olc<dataType, compType, dataType>(handle,
                                                                       invariantDims,
                                                                       toReduceDims,
                                                                       in,
                                                                       out_dev,
                                                                       out_indices_dev,
                                                                       reduceOp,
                                                                       nanOpt,
                                                                       indicesOpt,
                                                                       alpha,
                                                                       beta,
                                                                       tunable,
                                                                       nrepeat);

    if(do_verification)
    {
        ReductionHost<dataType, dataType> hostReduce(reduceOp,
                                                     compTypeId,
                                                     nanOpt,
                                                     indicesOpt,
                                                     in.mDesc,
                                                     out_host.mDesc,
                                                     invariantDims,
                                                     toReduceDims);

        hostReduce.Run(
            alpha, in.mData.data(), beta, out_host.mData.data(), out_indices_host.mData.data());

        check_error(out_host, out_dev);

        if(need_indices)
            check_indices(out_indices_host, out_indices_dev);

        if(do_logging)
        {
            LogRange(std::cout << "in : ", in.mData, ",") << std::endl;
            LogRange(std::cout << "out_host  : ", out_host.mData, ",") << std::endl;
            LogRange(std::cout << "out_device: ", out_dev.mData, ",") << std::endl;
        }
    }

    if(do_dumpout)
    {
        dumpBufferToFile("dump_in.bin", in.mData.data(), in.mDesc.GetElementSize());
        dumpBufferToFile("dump_out.bin", out_dev.mData.data(), out_dev.mDesc.GetElementSize());
        dumpBufferToFile(
            "dump_out_host.bin", out_host.mData.data(), out_host.mDesc.GetElementSize());
        dumpBufferToFile("dump_indices.bin",
                         out_indices_dev.mData.data(),
                         out_indices_dev.mDesc.GetElementSize());
        dumpBufferToFile("dump_indices_host.bin",
                         out_indices_host.mData.data(),
                         out_indices_host.mDesc.GetElementSize());
    };
}
