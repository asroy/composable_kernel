#pragma once
#include "ConstantTensorDescriptor.hip.hpp"

// starting point need to be aligned to float4 or float2 or float
// stride1 need to be 1 for both source and destination
template <unsigned BlockSize,
          class Float,
          class SrcDesc,
          class DstDesc,
          class CopyLengths,
          unsigned DataPerRead>
struct Blockwise2dTensorCopy3
{
    using vector_t = typename vector_type<Float, DataPerRead>::type;

    unsigned mSrcMyThreadOffset;
    unsigned mDstMyThreadOffset;

    __device__ Blockwise2dTensorCopy3()
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        static_assert(SrcDesc{}.GetStride(I1) == 1 && DstDesc{}.GetStride(I1) == 1,
                      "wrong! only support stride1 == 1!\n");

        static_assert(DataPerRead == 1 || DataPerRead == 2 || DataPerRead == 4,
                      "wrong! only support DataPerRead == 1, 2 or 4!\n");

        static_assert(SrcDesc{}.GetStride(I0) % DataPerRead == 0 &&
                          DstDesc{}.GetStride(I0) % DataPerRead == 0,
                      "src and dst stride should be multiple of DataPerRead to keep alignment");

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);

        constexpr unsigned thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr unsigned thread_per_d0 = BlockSize / thread_per_d1;

        // we allow out-of-bound read from src in D1 dimension,
        //   but we need to make sure dst stride is big enough,
        //   so that the out-of-bound write won't contaminate next line in dst
        static_assert(thread_per_d1 * DataPerRead <= DstDesc{}.GetStride(I0),
                      "wrong! out-of-bound write will contaminate next line!\n");

        static_assert(thread_per_d0 >= 1, "wrong! not enough threads to cover one line\n");

        constexpr unsigned num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        const unsigned thread_id_d0 = get_thread_local_1d_id() / thread_per_d1;
        const unsigned thread_id_d1 = get_thread_local_1d_id() - thread_id_d0 * thread_per_d1;

        mSrcMyThreadOffset = SrcDesc{}.Get1dIndex(thread_id_d0, thread_id_d1 * DataPerRead);
        mDstMyThreadOffset = DstDesc{}.Get1dIndex(thread_id_d0, thread_id_d1 * DataPerRead);
    }

    __device__ void Run(const Float* __restrict__ p_src, Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);

        constexpr unsigned thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr unsigned thread_per_d0 = BlockSize / thread_per_d1;

        constexpr unsigned num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr unsigned nloop_d0 = L0 / thread_per_d0;

        constexpr unsigned src_loop_stride = SrcDesc{}.GetStride(I0) * thread_per_d0;
        constexpr unsigned dst_loop_stride = DstDesc{}.GetStride(I0) * thread_per_d0;

        auto f_copy = [&](unsigned iloop) {
            *(reinterpret_cast<vector_t*>(p_dst + mDstMyThreadOffset + iloop * dst_loop_stride)) =
                *(reinterpret_cast<const vector_t*>(p_src + mSrcMyThreadOffset +
                                                    iloop * src_loop_stride));
        };

        for(unsigned iloop = 0; iloop < nloop_d0; ++iloop)
        {
            f_copy(iloop);
        }

        constexpr bool has_tail_d0 = (L0 > nloop_d0 * thread_per_d0);

        if(has_tail_d0)
        {
            constexpr unsigned tail_d0 = L0 - nloop_d0 * thread_per_d0;

            if(get_thread_local_1d_id() < tail_d0 * thread_per_d1)
            {
                f_copy(nloop_d0);
            }
        }
    }

    __device__ constexpr unsigned GetRegisterClipboardSize() const
    {
        static_assert(is_same<Float, float>::value, "wrong! only support float!\n");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);

        constexpr unsigned thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr unsigned thread_per_d0 = BlockSize / thread_per_d1;

        return DataPerRead * (L0 + thread_per_d0 - 1) / thread_per_d0;
    }

    __device__ void RunLoadRegisterClipboard(const Float* __restrict__ p_src,
                                             Float* p_clipboard) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);

        constexpr unsigned thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr unsigned thread_per_d0 = BlockSize / thread_per_d1;

        constexpr unsigned num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr unsigned nloop_d0 = L0 / thread_per_d0;

        constexpr unsigned src_loop_stride = SrcDesc{}.GetStride(I0) * thread_per_d0;
        constexpr unsigned dst_loop_stride = DstDesc{}.GetStride(I0) * thread_per_d0;

        auto f_copy = [&](unsigned iloop) {
            *(reinterpret_cast<vector_t*>(p_clipboard + iloop * 4)) =
                *(reinterpret_cast<const vector_t*>(p_src + mSrcMyThreadOffset +
                                                    iloop * src_loop_stride));
        };

        for(unsigned iloop = 0; iloop < nloop_d0; ++iloop)
        {
            f_copy(iloop);
        }

        constexpr bool has_tail_d0 = (L0 > nloop_d0 * thread_per_d0);

        if(has_tail_d0)
        {
            constexpr unsigned tail_d0 = L0 - nloop_d0 * thread_per_d0;

            if(get_thread_local_1d_id() < tail_d0 * thread_per_d1)
            {
                f_copy(nloop_d0);
            }
        }
    }

    __device__ void RunStoreRegisterClipboard(const Float* __restrict__ p_clipboard,
                                              Float* __restrict__ p_dst) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr unsigned L0 = CopyLengths{}.Get(I0);
        constexpr unsigned L1 = CopyLengths{}.Get(I1);

        constexpr unsigned thread_per_d1 = (L1 + DataPerRead - 1) / DataPerRead;
        constexpr unsigned thread_per_d0 = BlockSize / thread_per_d1;

        constexpr unsigned num_active_thread = thread_per_d0 * thread_per_d1;

        if(BlockSize > num_active_thread)
        {
            if(get_thread_local_1d_id() >= num_active_thread)
            {
                return;
            }
        }

        constexpr unsigned nloop_d0 = L0 / thread_per_d0;

        constexpr unsigned src_loop_stride = SrcDesc{}.GetStride(I0) * thread_per_d0;
        constexpr unsigned dst_loop_stride = DstDesc{}.GetStride(I0) * thread_per_d0;

        auto f_copy = [&](unsigned iloop) {
            *(reinterpret_cast<vector_t*>(p_dst + mDstMyThreadOffset + iloop * dst_loop_stride)) =
                *(reinterpret_cast<const vector_t*>(p_clipboard + iloop * 4));
        };

        for(unsigned iloop = 0; iloop < nloop_d0; ++iloop)
        {
            f_copy(iloop);
        }

        constexpr bool has_tail_d0 = (L0 > nloop_d0 * thread_per_d0);

        if(has_tail_d0)
        {
            constexpr unsigned tail_d0 = L0 - nloop_d0 * thread_per_d0;

            if(get_thread_local_1d_id() < tail_d0 * thread_per_d1)
            {
                f_copy(nloop_d0);
            }
        }
    }
};
