#pragma once
#include "ConstantTensorDescriptor.hip.hpp"

// starting point need to be aligned to float4 or float2 or float
// stride3 need to be 1 for both source and destination
template <unsigned BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class CopyLengths,
          class ThreadPerDims,
          unsigned DataPerRead>
struct Blockwise4dTensorCopy3
{
    using vector_t = typename vector_type<Float, DataPerRead>::type;

    unsigned mSrcMyThreadOffset;
    unsigned mDstMyThreadOffset;

    __device__ Blockwise4dTensorCopy3()
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        static_assert(SrcDesc{}.GetStride(I3) == 1 && DstDesc{}.GetStride(I3) == 1,
                      "wrong! only support stride3 == 1!\n");

        static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                      "wrong! only support DataPerRead == 1, 2 or 4!\n");

        static_assert(
            SrcDesc{}.GetStride(I2) % DataPerRead == 0 &&
                DstDesc{}.GetStride(I2) % DataPerRead == 0,
            "wrong! src and dst stride should be multiple of DataPerRead to keep alignment");

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);
        constexpr unsigned L2 = CopyLengths{}.Get(I2);
        constexpr unsigned L3 = CopyLengths{}.Get(I3);

        constexpr unsigned thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr unsigned thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr unsigned thread_per_d2 = ThreadPerDims{}.Get(I2);
        constexpr unsigned thread_per_d3 = ThreadPerDims{}.Get(I3);

        // we allow out-of-bound read from src in D3 dimension,
        //   but we need to make sure dst stride is big enough,
        //   so that the out-of-bound write won't contaminate next line in dst
        constexpr unsigned nloop_d3 = integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

        static_assert(nloop_d3 * thread_per_d3 * DataPerRead <= DstDesc{}.GetStride(I2),
                      "wrong! out-of-bound write will contaminate next line!\n");

        static_assert(L0 % thread_per_d0 == 0 && L1 % thread_per_d1 == 0 && L2 % thread_per_d2 == 0,
                      "wrong! L0, L1, L2 should be divided evenly!\n");

        static_assert(BlockSize >= thread_per_d0 * thread_per_d1 * thread_per_d2 * thread_per_d3,
                      "wrrong! BlockSize is not big enough for ThreadPerDims!");

        constexpr unsigned num_active_thread =
            thread_per_d0 * thread_per_d1 * thread_per_d2 * thread_per_d3;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        const unsigned thread_id_d0 =
            get_thread_local_1d_id() / (thread_per_d1 * thread_per_d2 * thread_per_d3);
        unsigned itmp = get_thread_local_1d_id() -
                        thread_id_d0 * (thread_per_d1 * thread_per_d2 * thread_per_d3);
        const unsigned thread_id_d1 = itmp / (thread_per_d2 * thread_per_d3);
        itmp -= thread_id_d1 * (thread_per_d2 * thread_per_d3);
        const unsigned thread_id_d2 = itmp / thread_per_d3;
        const unsigned thread_id_d3 = itmp - thread_id_d2 * thread_per_d3;

        mSrcMyThreadOffset = SrcDesc{}.Get1dIndex(
            thread_id_d0, thread_id_d1, thread_id_d2, thread_id_d3 * DataPerRead);
        mDstMyThreadOffset = DstDesc{}.Get1dIndex(
            thread_id_d0, thread_id_d1, thread_id_d2, thread_id_d3 * DataPerRead);
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);
        constexpr unsigned L2 = CopyLengths{}.Get(I2);
        constexpr unsigned L3 = CopyLengths{}.Get(I3);

        constexpr unsigned thread_per_d0 = ThreadPerDims{}.Get(I0);
        constexpr unsigned thread_per_d1 = ThreadPerDims{}.Get(I1);
        constexpr unsigned thread_per_d2 = ThreadPerDims{}.Get(I2);
        constexpr unsigned thread_per_d3 = ThreadPerDims{}.Get(I3);

        constexpr unsigned num_active_thread =
            thread_per_d0 * thread_per_d1 * thread_per_d2 * thread_per_d3;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr unsigned nloop_d0 = L0 / thread_per_d0;
        constexpr unsigned nloop_d1 = L1 / thread_per_d1;
        constexpr unsigned nloop_d2 = L2 / thread_per_d2;
        constexpr unsigned nloop_d3 = integer_divide_ceil(L3, thread_per_d3 * DataPerRead);

#pragma unroll
        for(unsigned iloop_d0 = 0; iloop_d0 < nloop_d0; ++iloop_d0)
        {
#pragma unroll
            for(unsigned iloop_d1 = 0; iloop_d1 < nloop_d1; ++iloop_d1)
            {
#pragma unroll
                for(unsigned iloop_d2 = 0; iloop_d2 < nloop_d2; ++iloop_d2)
                {
#pragma unroll
                    for(unsigned iloop_d3 = 0; iloop_d3 < nloop_d3; ++iloop_d3)
                    {
                        const unsigned src_offset =
                            SrcDesc{}.Get1dIndex(iloop_d0 * thread_per_d0,
                                                 iloop_d1 * thread_per_d1,
                                                 iloop_d2 * thread_per_d2,
                                                 iloop_d3 * thread_per_d3 * DataPerRead);

                        const unsigned dst_offset =
                            DstDesc{}.Get1dIndex(iloop_d0 * thread_per_d0,
                                                 iloop_d1 * thread_per_d1,
                                                 iloop_d2 * thread_per_d2,
                                                 iloop_d3 * thread_per_d3 * DataPerRead);

                        *(reinterpret_cast<vector_t*>(p_dst + dst_offset + mDstMyThreadOffset)) =
                            *(reinterpret_cast<const vector_t*>(p_src + src_offset +
                                                                mSrcMyThreadOffset));
                    }
                }
            }
        }
    }
};
