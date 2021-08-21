#include "device.hpp"
#include "host_tensor.hpp"

#include "online_reduce_common.hpp"
#include "reduce_tunables.hpp"

#include "handle.hpp"

#include <sstream>

namespace detail_dyn_generic_reduction {

template <typename TSrc, typename TComp, typename TDst>
static std::string get_network_config_string_from_types()
{
    std::ostringstream outs;

    outs << Driver::get_type_enum_from_type<TSrc>() << Driver::get_type_enum_from_type<TComp>()
         << Driver::get_type_enum_from_type<TDst>();

    return (outs.str());
};

static std::string get_network_config_string_from_tunable(const tunable_dyn_generic_reduction* pt)
{
    std::ostringstream outs;

    outs << "TUN_" << pt->BlockSize << "_";
    outs << pt->GredThreadBufferLength << "_";
    outs << pt->GredAccessesPerThreadInBlock << "_";
    outs << pt->GredAccessesPerThreadInWarp;

    return (outs.str());
};

template <typename TSrc, typename TComp, typename TDst>
static std::string get_definition_string_from_types()
{
    std::ostringstream outs;

    outs << " -DCK_PARAM_SRC_DATATYPE=" << Driver::get_type_enum_from_type<TSrc>();
    outs << " -DCK_PARAM_DST_DATATYPE=" << Driver::get_type_enum_from_type<TDst>();
    outs << " -DCK_PARAM_REDUCE_COMPTYPE=" << Driver::get_type_enum_from_type<TComp>();

    return (outs.str());
};

static std::string get_definition_string_from_tunable(const tunable_dyn_generic_reduction* pt)
{
    std::ostringstream outs;

    outs << " -DCK_PARAM_BLOCKSIZE=" << pt->BlockSize;
    outs << " -DCK_PARAM_THREAD_BUFFER_LENGTH=" << pt->GredThreadBufferLength;
    outs << " -DCK_PARAM_ACCESSES_PER_THREAD_INBLOCK=" << pt->GredAccessesPerThreadInBlock;
    outs << " -DCK_PARAM_ACCESSES_PER_THREAD_INWARP=" << pt->GredAccessesPerThreadInWarp;

    return (outs.str());
};

struct ReductionKernelConfigurator
{
    ReductionKernelConfigurator() = default;

    ReductionKernelConfigurator(int blockSize, int warpSize)
        : blockSize_(blockSize), warpSize_(warpSize)
    {
        GredDirectThreadWiseUpperReductionLen = warpSize;
        GredDirectWarpWiseUpperReductionLen   = blockSize;
        GredBlockWiseUpperReductionLen        = blockSize * 4;
        GredUpperNumBlocksPerReduction        = 32;

        numWarpsPerBlock = blockSize / warpSize;
    };

    int blockSize_;
    int warpSize_;
    int numWarpsPerBlock;

    std::size_t GredDirectThreadWiseUpperReductionLen;
    std::size_t GredDirectWarpWiseUpperReductionLen;
    std::size_t GredBlockWiseUpperReductionLen;
    std::size_t GredUpperNumBlocksPerReduction;

    std::size_t getGridSize(std::size_t invariantLength, std::size_t toReduceLength) const
    {
        assert(invariantLength > 0 && toReduceLength > 1);

        if(invariantLength == 1)
        {
            if(toReduceLength <=
               GredBlockWiseUpperReductionLen) // let one block to do this only reduction
                return (1);
            else
                return ((toReduceLength + blockSize_ - 1) /
                        blockSize_); // let multiple blocks to do this only reduction
        }
        else
        {
            if(toReduceLength <=
               GredDirectThreadWiseUpperReductionLen) // let one thread to do each reduction
                return ((invariantLength + blockSize_ - 1) / blockSize_);
            else if(toReduceLength <=
                    GredDirectWarpWiseUpperReductionLen) // let one warp to do each reduction
                return ((invariantLength + numWarpsPerBlock - 1) / numWarpsPerBlock);
            else if(toReduceLength <=
                    GredBlockWiseUpperReductionLen) // let one block to do each reduction
                return (invariantLength);
            else
            { // let multiple blocks to do each reduction
                std::size_t expBlocksPerReduction =
                    (toReduceLength + GredBlockWiseUpperReductionLen - 1) /
                    GredBlockWiseUpperReductionLen;

                if(expBlocksPerReduction > GredUpperNumBlocksPerReduction)
                    return (invariantLength * GredUpperNumBlocksPerReduction);
                else
                    return (invariantLength * expBlocksPerReduction);
            };
        };
    };

    ReductionMethod_t getReductionMethod(std::size_t invariantLength,
                                         std::size_t toReduceLength) const
    {
        assert(invariantLength > 0 && toReduceLength > 1);

        if(invariantLength == 1)
        {
            if(toReduceLength <=
               GredBlockWiseUpperReductionLen) // let one block to do this only reduction
                return (ReductionMethod_t::BlockWise);
            else // let multiple blocks to do this only reduction
                return (ReductionMethod_t::MultiBlock);
        }
        else
        {
            if(toReduceLength <=
               GredDirectThreadWiseUpperReductionLen) // let one thread to do each reduction
                return (ReductionMethod_t::DirectThreadWise);
            else if(toReduceLength <=
                    GredDirectWarpWiseUpperReductionLen) // let one warp to do each reduction
                return (ReductionMethod_t::DirectWarpWise);
            else if(toReduceLength <=
                    GredBlockWiseUpperReductionLen) // let one block to do each reduction
                return (ReductionMethod_t::BlockWise);
            else
                return (ReductionMethod_t::MultiBlock); // let multiple blocks to do each reduction
        };
    };

    std::size_t getWorkspaceSize(std::size_t invariantLength, std::size_t toReduceLength) const
    {
        assert(invariantLength > 0 && toReduceLength > 1);

        if(getReductionMethod(invariantLength, toReduceLength) == ReductionMethod_t::MultiBlock)
        {
            auto gridSize = getGridSize(invariantLength, toReduceLength);

            return (gridSize);
        };

        return (0);
    };

    std::size_t getGridSize_2(std::size_t invariantLength, std::size_t toReduceLength) const
    {
        if(toReduceLength <= warpSize_ / 4) // let one thread to do each reduction
            return ((invariantLength + blockSize_ - 1) / blockSize_);
        else if(toReduceLength <= blockSize_) // let one warp to do each reduction
            return ((invariantLength + numWarpsPerBlock - 1) / numWarpsPerBlock);
        else
            return (invariantLength); // let one block to do each reduction
    };

    ReductionMethod_t GetReductionMethod_2(std::size_t invariantLength,
                                           std::size_t toReduceLength) const
    {
        if(toReduceLength <= warpSize_ / 4) // let one thread to do each reduction
            return (ReductionMethod_t::DirectThreadWise);
        else if(toReduceLength <= blockSize_) // let one warp to do each reduction
            return (ReductionMethod_t::DirectWarpWise);
        else
            return (ReductionMethod_t::BlockWise);
    };
};

static std::pair<bool, bool> get_padding_need(ReductionMethod_t reduceImpl,
                                              size_t invariantLen,
                                              size_t toReduceLen,
                                              int GridSize,
                                              int BlockSize,
                                              int BlkGroupSize,
                                              const tunable_dyn_generic_reduction* tunable)
{
    bool src_need_padding = false;
    bool dst_need_padding = false;
    int copySliceLen;
    int reduceSizePerBlock;

    switch(reduceImpl)
    {
    case ReductionMethod_t::DirectThreadWise:
        copySliceLen = tunable->GredThreadBufferLength;
        src_need_padding =
            (invariantLen < GridSize * BlockSize || toReduceLen % copySliceLen > 0) ? true : false;
        dst_need_padding = (invariantLen < GridSize * BlockSize) ? true : false;
        break;
    case ReductionMethod_t::DirectWarpWise:
        copySliceLen = warpSize * tunable->GredAccessesPerThreadInWarp;
        src_need_padding =
            (invariantLen < GridSize * BlockSize / warpSize || toReduceLen % copySliceLen > 0)
                ? true
                : false;
        dst_need_padding = (invariantLen < GridSize * BlockSize / warpSize) ? true : false;
        break;
    case ReductionMethod_t::BlockWise:
        copySliceLen     = BlockSize * tunable->GredAccessesPerThreadInBlock;
        src_need_padding = (toReduceLen % copySliceLen > 0) ? true : false;
        break;
    case ReductionMethod_t::MultiBlock:
        copySliceLen = BlockSize * tunable->GredAccessesPerThreadInBlock;
        reduceSizePerBlock =
            (((toReduceLen + BlkGroupSize - 1) / BlkGroupSize + copySliceLen - 1) / copySliceLen) *
            copySliceLen;
        src_need_padding = (toReduceLen < reduceSizePerBlock * BlkGroupSize) ? true : false;
        break;
    default: throw std::runtime_error("Invalid reduction method ID!"); break;
    };

    return (std::make_pair(src_need_padding, dst_need_padding));
};

static std::string getReductionMethodStr(ReductionMethod_t reduceImpl)
{
    switch(reduceImpl)
    {
    case ReductionMethod_t::DirectThreadWise: return (std::string("threadwise"));
    case ReductionMethod_t::DirectWarpWise: return (std::string("warpwise"));
    case ReductionMethod_t::BlockWise: return (std::string("blockwise"));
    case ReductionMethod_t::MultiBlock: return (std::string("multiblock"));
    default: throw std::runtime_error("Invalid reduction method ID!"); break;
    };
};

static inline std::string get_arch_specific_compiler_flag(const online_compile::Handle* handle)
{
    std::string compiler_flag;

    // GPU target
    static const std::string gpu_target = handle->GetDeviceName();

    if(gpu_target.compare(0, 6, "gfx803") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX803";
    else if(gpu_target.compare(0, 6, "gfx900") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX900";
    else if(gpu_target.compare(0, 6, "gfx906") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX906";
    else if(gpu_target.compare(0, 6, "gfx908") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX908";
    else if(gpu_target.compare(0, 6, "gfx90a") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX90A";
    else if(gpu_target.compare(0, 7, "gfx1030") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX1030";

    return compiler_flag;
};

} // namespace detail_dyn_generic_reduction

template <typename TSrc, typename TComp, typename TDst>
void device_dynamic_generic_reduction_olc(online_compile::Handle* handle,
                                          const std::vector<int> invariantDims,
                                          const std::vector<int> toReduceDims,
                                          const Tensor<TSrc>& in,
                                          Tensor<TDst>& out,
                                          Tensor<int>& out_indices,
                                          ReduceTensorOp_t reduceOp,
                                          NanPropagation_t nanPropaOpt,
                                          ReduceTensorIndices_t reduceIndicesOpt,
                                          float alpha,
                                          float beta,
                                          const tunable_dyn_generic_reduction* tunable,
                                          int nrepeat)
{
    using namespace ck;
    using namespace detail_dyn_generic_reduction;
    using size_t = std::size_t;

    size_t invariantLength = out.mDesc.GetElementSize();
    size_t toReduceLength  = in.mDesc.GetElementSize() / invariantLength;
    int origReduceLen      = toReduceLength;

    ReductionKernelConfigurator configurator(tunable->BlockSize, handle->GetWavefrontWidth());

    // these buffers are usually provided by the user application
    DeviceMem in_dev_buf(sizeof(TSrc) * in.mDesc.GetElementSpace());
    DeviceMem out_dev_buf(sizeof(TDst) * out.mDesc.GetElementSpace());

    in_dev_buf.ToDevice(in.mData.data());

    if(beta != 0.0f)
        out_dev_buf.ToDevice(out.mData.data());

    auto inLengths  = in.mDesc.GetLengths();
    auto inStrides  = in.mDesc.GetStrides();
    auto outLengths = out.mDesc.GetLengths();
    auto outStrides = out.mDesc.GetStrides();

    int p_inLengths[6];
    int p_inStrides[6];
    int p_outLengths[6];
    int p_outStrides[6];

    for(int i = 0; i < inLengths.size(); i++)
        p_inLengths[i] = static_cast<int>(inLengths[i]);

    for(int i = 0; i < inStrides.size(); i++)
        p_inStrides[i] = static_cast<int>(inStrides[i]);

    for(int i = 0; i < outLengths.size(); i++)
        p_outLengths[i] = static_cast<int>(outLengths[i]);

    for(int i = 0; i < outStrides.size(); i++)
        p_outStrides[i] = static_cast<int>(outStrides[i]);

    auto workspace_size = configurator.getWorkspaceSize(invariantLength, toReduceLength);

    bool need_indices = (reduceIndicesOpt == REDUCE_TENSOR_FLATTENED_INDICES) &&
                        (reduceOp == REDUCE_TENSOR_MIN || reduceOp == REDUCE_TENSOR_MAX ||
                         reduceOp == REDUCE_TENSOR_AMAX);

    size_t wsSizeInBytes = !need_indices
                               ? workspace_size * sizeof(TSrc)
                               : workspace_size * (sizeof(TSrc) + sizeof(int)) + 64 + sizeof(int);

    DeviceMem workspace1(wsSizeInBytes);
    DeviceMem workspace2(4096);

    void* p_dev_src2dDesc = (char*)workspace2.GetDeviceBuffer();
    void* p_dev_dst1dDesc = (char*)workspace2.GetDeviceBuffer() + 2048;

    size_t indicesSizeInBytes = need_indices ? out.mDesc.GetElementSize() * sizeof(int) : 0;

    DeviceMem indices_dev_buf(indicesSizeInBytes);

    size_t ws_buf2_bytes_offset = 0;

    if(need_indices && workspace_size > 0)
    {
        size_t byteOffset =
            static_cast<size_t>((wsSizeInBytes / (sizeof(TSrc) + sizeof(int))) * sizeof(TSrc));

        ws_buf2_bytes_offset = ((byteOffset + 63) / 64) * 64;
    };

    ReductionMethod_t reduceImpl = configurator.getReductionMethod(invariantLength, toReduceLength);
    int GridSize = static_cast<int>(configurator.getGridSize(invariantLength, toReduceLength));
    int BlkGroupSize =
        (reduceImpl == ReductionMethod_t::MultiBlock) ? GridSize / invariantLength : 0;

    const std::vector<size_t> vld  = {static_cast<size_t>(tunable->BlockSize), 1, 1};
    const std::vector<size_t> vgd1 = {static_cast<size_t>(tunable->BlockSize), 1, 1};
    const std::vector<size_t> vgd2 = {static_cast<size_t>(GridSize) * tunable->BlockSize, 1, 1};

    std::string algo_name = "dynamic_generic_reduction";

    std::string param = " -std=c++17 ";
    std::string network_config;

    param += get_arch_specific_compiler_flag(handle);

    param += get_definition_string_from_types<TSrc, TComp, TDst>() + " " +
             get_definition_string_from_tunable(tunable);

    param += " -DCK_PARAM_TOREDUCE_DIMS=";
    for(int i = 0; i < toReduceDims.size(); i++)
    {
        param += std::to_string(toReduceDims[i]);
        if(i < toReduceDims.size() - 1)
            param += ",";
    };

    if(!invariantDims.empty())
    {
        param += " -DCK_PARAM_INVARIANT_DIMS=";
        for(int i = 0; i < invariantDims.size(); i++)
        {
            param += std::to_string(invariantDims[i]);
            if(i < invariantDims.size() - 1)
                param += ",";
        };
    }
    else
    {
        param += " -DCK_PARAM_INVARIANT_DIMS= ";
        param += " -DCK_REDUCE_ALL_DIMS=1";
    };

    param += " -DCK_PARAM_REDUCE_OP=" + std::to_string(reduceOp);
    param += " -DCK_PARAM_NAN_PROPAGATE=" + std::to_string(nanPropaOpt == PROPAGATE_NAN ? 1 : 0);
    param += " -DCK_PARAM_REDUCE_INDICES=" +
             std::to_string(reduceIndicesOpt == REDUCE_TENSOR_FLATTENED_INDICES ? 1 : 0);
    param += " -DCK_PARAM_IN_DIMS=" + std::to_string(inLengths.size());
    param += " -DCK_PARAM_OUT_DIMS=" + std::to_string(outLengths.size());

    // disable AMD Buffer Addressing for double data transfering
    if(std::is_same<TSrc, double>::value || std::is_same<TDst, double>::value)
        param += " -DCK_USE_AMD_BUFFER_ADDRESSING=0";

    network_config = get_network_config_string_from_types<TSrc, TComp, TDst>() + "_" +
                     get_network_config_string_from_tunable(tunable) + "_";

    network_config += "I" + std::to_string(inLengths.size()) + "_";

    network_config += "RED";
    for(auto dim : toReduceDims)
        network_config += std::to_string(dim) + "_";
    network_config += "BSIZE_" + std::to_string(tunable->BlockSize);

    const bool reduceAllDims = invariantDims.empty();

    std::vector<float> kernel1_times;
    std::vector<float> kernel2_times;
    std::vector<float> kernel3_times;
    std::vector<float> kernel4_times;

    size_t total_transfer_bytes = 0;
    float total_transfer_time   = 0;

    for(int i = 0; i < nrepeat; ++i)
    {
        KernelTimer timer1, timer2;
        auto use_padding = get_padding_need(reduceImpl,
                                            invariantLength,
                                            toReduceLength,
                                            GridSize,
                                            tunable->BlockSize,
                                            BlkGroupSize,
                                            tunable);

        std::string param1 = param +
                             " -DCK_PARAM_SRC2D_PADDING=" + std::to_string(use_padding.first) +
                             " -DCK_PARAM_DST1D_PADDING=" + std::to_string(use_padding.second);

        std::string program_name1 =
            "gridwise_generic_reduction_first_call_" + getReductionMethodStr(reduceImpl) +
            (reduceAllDims ? "_reduce_all_dims.cpp" : "_reduce_partial_dims.cpp");
        std::string kernel_name1     = "gridwise_generic_reduce_1_prepare";
        std::string network_config_1 = network_config + "_1_P" + std::to_string(reduceImpl) +
                                       std::to_string(use_padding.first) +
                                       std::to_string(use_padding.second);

        timer1.Start();
        handle->AddKernel(
            algo_name, network_config_1, program_name1, kernel_name1, vld, vgd1, param1)(
            GridSize,
            BlkGroupSize,
            p_inLengths[0],
            p_inLengths[1],
            p_inLengths[2],
            p_inLengths[3],
            p_inLengths[4],
            p_inLengths[5],
            p_inStrides[0],
            p_inStrides[1],
            p_inStrides[2],
            p_inStrides[3],
            p_inStrides[4],
            p_inStrides[5],
            p_outLengths[0],
            p_outLengths[1],
            p_outLengths[2],
            p_outLengths[3],
            p_outLengths[4],
            p_outLengths[5],
            p_outStrides[0],
            p_outStrides[1],
            p_outStrides[2],
            p_outStrides[3],
            p_outStrides[4],
            p_outStrides[5],
            p_dev_src2dDesc,
            p_dev_dst1dDesc);
        timer1.End();

        kernel_name1     = "gridwise_generic_reduce_1";
        network_config_1 = network_config + "_1" + std::to_string(reduceImpl) +
                           std::to_string(use_padding.first) + std::to_string(use_padding.second);

        timer2.Start();
        handle->AddKernel(
            algo_name, network_config_1, program_name1, kernel_name1, vld, vgd2, param1)(
            origReduceLen,
            BlkGroupSize,
            p_dev_src2dDesc,
            p_dev_dst1dDesc,
            alpha,
            in_dev_buf.GetDeviceBuffer(),
            beta,
            out_dev_buf.GetDeviceBuffer(),
            workspace1.GetDeviceBuffer(),
            ws_buf2_bytes_offset,
            indices_dev_buf.GetDeviceBuffer());
        timer2.End();

        kernel1_times.push_back(timer1.GetElapsedTime());
        kernel2_times.push_back(timer2.GetElapsedTime());

        total_transfer_bytes = (size_t)invariantLength * toReduceLength * sizeof(TSrc);

        if(reduceImpl == ReductionMethod_t::MultiBlock)
        {
            auto toReduceLength_2 = BlkGroupSize;
            int GridSize_2 =
                static_cast<int>(configurator.getGridSize_2(invariantLength, toReduceLength_2));
            const std::vector<size_t> vgd2_2 = {
                static_cast<size_t>(GridSize_2) * tunable->BlockSize, size_t{1}, size_t{1}};
            auto reduceImpl2 = configurator.GetReductionMethod_2(invariantLength, toReduceLength_2);
            auto use_padding2 = get_padding_need(reduceImpl2,
                                                 invariantLength,
                                                 toReduceLength_2,
                                                 GridSize_2,
                                                 tunable->BlockSize,
                                                 BlkGroupSize,
                                                 tunable);

            std::string param2 = param +
                                 " -DCK_PARAM_SRC2D_PADDING=" + std::to_string(use_padding2.first) +
                                 " -DCK_PARAM_DST1D_PADDING=" + std::to_string(use_padding2.second);

            std::string program_name2 = "gridwise_generic_reduction_second_call_" +
                                        getReductionMethodStr(reduceImpl2) + ".cpp";
            std::string kernel_name2     = "gridwise_generic_reduce_2_prepare";
            std::string network_config_2 = network_config + "_2_P" + std::to_string(reduceImpl2) +
                                           std::to_string(use_padding2.first) +
                                           std::to_string(use_padding2.second);

            timer1.Start();
            handle->AddKernel(
                algo_name, network_config_2, program_name2, kernel_name2, vld, vgd1, param2)(
                GridSize_2,
                BlkGroupSize,
                p_outLengths[0],
                p_outLengths[1],
                p_outLengths[2],
                p_outLengths[3],
                p_outLengths[4],
                p_outLengths[5],
                p_outStrides[0],
                p_outStrides[1],
                p_outStrides[2],
                p_outStrides[3],
                p_outStrides[4],
                p_outStrides[5],
                p_dev_src2dDesc,
                p_dev_dst1dDesc);
            timer1.End();

            kernel_name2     = "gridwise_generic_reduce_2";
            network_config_2 = network_config + "_2" + std::to_string(reduceImpl2) +
                               std::to_string(use_padding2.first) +
                               std::to_string(use_padding2.second);

            timer2.Start();
            handle->AddKernel(
                algo_name, network_config_2, program_name2, kernel_name2, vld, vgd2_2, param2)(
                origReduceLen,
                p_dev_src2dDesc,
                p_dev_dst1dDesc,
                alpha,
                in_dev_buf.GetDeviceBuffer(),
                beta,
                out_dev_buf.GetDeviceBuffer(),
                workspace1.GetDeviceBuffer(),
                ws_buf2_bytes_offset,
                indices_dev_buf.GetDeviceBuffer());
            timer2.End();

            kernel3_times.push_back(timer1.GetElapsedTime());
            kernel4_times.push_back(timer2.GetElapsedTime());

            total_transfer_bytes += (size_t)invariantLength * toReduceLength_2 *
                                    (need_indices ? (sizeof(TSrc) + sizeof(int)) : sizeof(TSrc)) *
                                    2;
        };

        total_transfer_bytes +=
            (size_t)invariantLength * (need_indices ? (sizeof(TDst) + sizeof(int)) : sizeof(TDst));
    }

    {
        auto ave_time1 = Driver::get_effective_average(kernel1_times);
        auto ave_time2 = Driver::get_effective_average(kernel2_times);

        total_transfer_time += ave_time2;

        if(reduceImpl == ReductionMethod_t::MultiBlock)
        {
            auto ave_time3 = Driver::get_effective_average(kernel3_times);
            auto ave_time4 = Driver::get_effective_average(kernel4_times);

            total_transfer_time += ave_time4;

            std::cout << "Average time : " << ave_time1 + ave_time2 + ave_time3 + ave_time3
                      << " ms(" << ave_time1 + ave_time3 << ", " << ave_time2 + ave_time4 << ")"
                      << std::endl;
            std::cout << "Average transfer rate : "
                      << total_transfer_bytes * 0.000001f / total_transfer_time << " GBytes/second"
                      << std::endl;
        }
        else
        {
            std::cout << "Average time : " << ave_time1 + ave_time2 << " ms(" << ave_time1 << ", "
                      << ave_time2 << ")" << std::endl;
            std::cout << "Average transfer rate : "
                      << total_transfer_bytes * 0.000001f / total_transfer_time << " GBytes/second"
                      << std::endl;
        };
    };

    // copy result back to host
    out_dev_buf.FromDevice(out.mData.data());

    if(need_indices)
        indices_dev_buf.FromDevice(out_indices.mData.data());
}
