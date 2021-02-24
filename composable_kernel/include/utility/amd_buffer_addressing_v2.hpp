#ifndef CK_AMD_BUFFER_ADDRESSING_V2_HPP
#define CK_AMD_BUFFER_ADDRESSING_V2_HPP

#include "float_type.hpp"

namespace ck {

// For 128 bit SGPRs to supply resource constant in buffer instructions
// https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
template <typename T>
union BufferResourceConstant
{
    int32x4_t data;
    T* address[2];
    int32_t range[4];
    int32_t config[4];
};

__device__ float
__llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ float2_t
__llvm_amdgcn_raw_buffer_load_fp32x2(int32x4_t srsrc,
                                     index_t voffset,
                                     index_t soffset,
                                     index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f32");

__device__ float4_t
__llvm_amdgcn_raw_buffer_load_fp32x4(int32x4_t srsrc,
                                     index_t voffset,
                                     index_t soffset,
                                     index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

__device__ void
__llvm_amdgcn_raw_buffer_store_fp32(float vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f32");

__device__ void
__llvm_amdgcn_raw_buffer_store_fp32x2(float2_t vdata,
                                      int32x4_t rsrc,
                                      index_t voffset,
                                      index_t soffset,
                                      index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f32");

__device__ void
__llvm_amdgcn_raw_buffer_store_fp32x4(float4_t vdata,
                                      int32x4_t rsrc,
                                      index_t voffset,
                                      index_t soffset,
                                      index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f32");

// buffer_load requires:
//   1) p_src_wave must be in global memory space
//   2) p_src_wave to be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t VectorSize>
__device__ typename vector_type<T, VectorSize>::MemoryType
amd_buffer_load_v2(const T* p_src_wave,
                   index_t src_thread_data_offset,
                   bool src_thread_data_valid,
                   index_t src_elemenst_space);

// buffer_store requires:
//   1) p_dst_wave must be global memory
//   2) p_dst_wave to be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t VectorSize>
__device__ void
amd_buffer_store_v2(const typename vector_type<T, VectorSize>::MemoryType src_thread_data,
                    T* p_dst_wave,
                    const index_t dst_thread_data_offset,
                    const bool dst_thread_data_valid,
                    const index_t dst_data_range);

template <>
__device__ float amd_buffer_load_v2<float, 1>(const float* p_src_wave,
                                              index_t src_thread_data_offset,
                                              bool src_thread_data_valid,
                                              index_t src_data_range)
{
    BufferResourceConstant<float> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<float*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(float);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    return __llvm_amdgcn_raw_buffer_load_fp32(
        src_wave_buffer_resource.data, src_addr_shift + src_thread_addr_offset, 0, 0);
#else
    float tmp = __llvm_amdgcn_raw_buffer_load_fp32(
        src_wave_buffer_resource.data, src_thread_addr_offset, 0, 0);

    return src_thread_data_valid ? tmp : float(0);
#endif
}

template <>
__device__ float2_t amd_buffer_load_v2<float, 2>(const float* p_src_wave,
                                                 index_t src_thread_data_offset,
                                                 bool src_thread_data_valid,
                                                 index_t src_data_range)
{
    BufferResourceConstant<float> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<float*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(float);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    return __llvm_amdgcn_raw_buffer_load_fp32x2(
        src_wave_buffer_resource.data, src_addr_shift + src_thread_addr_offset, 0, 0);
#else
    float2_t tmp = __llvm_amdgcn_raw_buffer_load_fp32x2(
        src_wave_buffer_resource.data, src_thread_addr_offset, 0, 0);

    return src_thread_data_valid ? tmp : float2_t(0);
#endif
}

template <>
__device__ float4_t amd_buffer_load_v2<float, 4>(const float* p_src_wave,
                                                 index_t src_thread_data_offset,
                                                 bool src_thread_data_valid,
                                                 index_t src_data_range)
{
    BufferResourceConstant<float> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<float*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(float);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    return __llvm_amdgcn_raw_buffer_load_fp32x4(
        src_wave_buffer_resource.data, src_addr_shift + src_thread_addr_offset, 0, 0);
#else
    float4_t tmp = __llvm_amdgcn_raw_buffer_load_fp32x4(
        src_wave_buffer_resource.data, src_thread_addr_offset, 0, 0);

    return src_thread_data_valid ? tmp : float4_t(0);
#endif
}

template <>
__device__ float8_t amd_buffer_load_v2<float, 8>(const float* p_src_wave,
                                                 index_t src_thread_data_offset,
                                                 bool src_thread_data_valid,
                                                 index_t src_data_range)
{
    BufferResourceConstant<float> src_wave_buffer_resource;

    // wavewise base address (64 bit)
    src_wave_buffer_resource.address[0] = const_cast<float*>(p_src_wave);
    // wavewise range (32 bit)
    src_wave_buffer_resource.range[2] = src_data_range * sizeof(float);
    // wavewise setting (32 bit)
    src_wave_buffer_resource.config[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_data_valid ? 0 : 0x7fffffff;

    vector_type<float, 8> tmp;

    tmp.Vectors(Number<4>{})(Number<0>{}) = __llvm_amdgcn_raw_buffer_load_fp32x4(
        src_wave_buffer_resource.data, src_addr_shift + src_thread_addr_offset, 0, 0);

    tmp.Vectors(Number<4>{})(Number<1>{}) = __llvm_amdgcn_raw_buffer_load_fp32x4(
        src_wave_buffer_resource.data,
        src_addr_shift + src_thread_addr_offset + 4 * sizeof(float),
        0,
        0);

    return tmp.Vector();
#else
    vector_type<float, 8> tmp;

    tmp.Vectors(Number<4>{})(Number<0>{}) = __llvm_amdgcn_raw_buffer_load_fp32x4(
        src_wave_buffer_resource.data, src_thread_addr_offset, 0, 0);

    tmp.Vectors(Number<4>{})(Number<1>{}) = __llvm_amdgcn_raw_buffer_load_fp32x4(
        src_wave_buffer_resource.data, src_thread_addr_offset + 4 * sizeof(float), 0, 0);

    return src_thread_data_valid ? tmp.Vector() : float8_t(0);
#endif
}

template <>
__device__ void amd_buffer_store_v2<float, 1>(const float src_thread_data,
                                              float* p_dst_wave,
                                              const index_t dst_thread_data_offset,
                                              const bool dst_thread_data_valid,
                                              const index_t dst_data_range)
{
    BufferResourceConstant<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_raw_buffer_store_fp32(src_thread_data,
                                        dst_wave_buffer_resource.data,
                                        dst_addr_shift + dst_thread_addr_offset,
                                        0,
                                        0);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_buffer_store_fp32(
            src_thread_data, dst_wave_buffer_resource.data, dst_thread_addr_offset, 0, 0);
    }
#endif
}

template <>
__device__ void amd_buffer_store_v2<float, 2>(const float2_t src_thread_data,
                                              float* p_dst_wave,
                                              const index_t dst_thread_data_offset,
                                              const bool dst_thread_data_valid,
                                              const index_t dst_data_range)
{
    BufferResourceConstant<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_raw_buffer_store_fp32x2(src_thread_data,
                                          dst_wave_buffer_resource.data,
                                          dst_addr_shift + dst_thread_addr_offset,
                                          0,
                                          0);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_raw_buffer_store_fp32x2(
            src_thread_data, dst_wave_buffer_resource.data, dst_thread_addr_offset, 0, 0);
    }
#endif
}

template <>
__device__ void amd_buffer_store_v2<float, 4>(const float4_t src_thread_data,
                                              float* p_dst_wave,
                                              const index_t dst_thread_data_offset,
                                              const bool dst_thread_data_valid,
                                              const index_t dst_data_range)
{
    BufferResourceConstant<float> dst_wave_buffer_resource;

    // wavewise base address (64 bit)
    dst_wave_buffer_resource.address[0] = p_dst_wave;
    // wavewise range (32 bit)
    dst_wave_buffer_resource.range[2] = dst_data_range * sizeof(float);
    // wavewise setting (32 bit)
    dst_wave_buffer_resource.config[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_data_valid ? 0 : 0x7fffffff;

    __llvm_amdgcn_raw_buffer_store_fp32x4(src_thread_data,
                                          dst_wave_buffer_resource.data,
                                          dst_addr_shift + dst_thread_addr_offset,
                                          0,
                                          0);
#else
    if(dst_thread_data_valid)
    {
        __llvm_amdgcn_raw_buffer_store_fp32x4(
            src_thread_data, dst_wave_buffer_resource.data, dst_thread_addr_offset, 0, 0);
    }
#endif
}

} // namespace ck
#endif
