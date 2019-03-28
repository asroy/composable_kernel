#pragma once

extern "C" __attribute__((address_space(3))) void* __to_local(void* p) [[hc]];

template <class Float, class SrcMatrix, class DstMatrix, unsigned NRow, unsigned NCol>
__device__ void threadwise_matrix_copy(SrcMatrix,
                                       const Float* __restrict__ p_src,
                                       DstMatrix,
                                       Float* __restrict__ p_dst,
                                       Sequence<NRow, NCol>)
{
    constexpr auto src_mtx = SrcMatrix{};
    constexpr auto dst_mtx = DstMatrix{};

    for(unsigned i = 0; i < NRow; ++i)
    {
#if 1
        assert(NCol == 8);
        {
            const unsigned src_index = src_mtx.Get1dIndex(i, 0);
            const unsigned dst_index = dst_mtx.Get1dIndex(i, 0);

            const float4* loc = (const float4 *)(p_src + src_index);
            float4* reg = (float4 *)(p_dst + dst_index); 

            //reg[0] = loc[0];
            //reg[1] = loc[1];

            asm volatile("\n \
                    ds_read2_b64 %0, %2 offset1:1 \n \
                    ds_read2_b64 %1, %2 offset0:16 offset1:17 \n \
                    s_waitcnt lgkmcnt(0)" : "=v"(reg[0]), "=v"(reg[1]) : "v"(__to_local((void *)&p_src[src_index])));
        }

#else
        for(unsigned j = 0; j < NCol; ++j)
        {
            const unsigned src_index = src_mtx.Get1dIndex(i, j);
            const unsigned dst_index = dst_mtx.Get1dIndex(i, j);
            
            //p_dst[dst_index] = p_src[src_index];
            asm volatile("ds_read_b32 %0, %1 \ns_waitcnt lgkmcnt(0)" : "=v"(p_dst[dst_index]) : "v"(__to_local((void *)&p_src[src_index])));
        }
#endif
    }
}

template <class MatrixA,
          class MatrixB,
          class MatrixC,
          bool TransA,
          bool TransB,
          bool TransC,
          class FloatA,
          class FloatB,
          class FloatC,
          class Accumulator>
__device__ void threadwise_gemm(MatrixA,
                                integral_constant<bool, TransA>,
                                const FloatA* __restrict__ p_a_thread,
                                MatrixB,
                                integral_constant<bool, TransB>,
                                const FloatB* __restrict__ p_b_thread,
                                MatrixC,
                                integral_constant<bool, TransC>,
                                FloatC* __restrict__ p_c_thread,
                                Accumulator f_accum)
{
    if(TransA && (!TransB) && (!TransC))
    {
        constexpr auto a_mtx = MatrixA{};
        constexpr auto b_mtx = MatrixB{};
        constexpr auto c_mtx = MatrixC{};

        constexpr unsigned M = c_mtx.NRow();
        constexpr unsigned N = c_mtx.NCol();
        constexpr unsigned K = a_mtx.NRow(); // A is transposed

        assert(M == 8);
        assert(N == 8);
        assert(K == 1);

        for(unsigned k = 0; k < K; ++k)
        {
            const unsigned bindex = b_mtx.Get1dIndex(k, 0);
            for(unsigned i = 0; i < M; ++i)
            {
                const unsigned aindex = a_mtx.Get1dIndex(k, i); // A is transposed
                const unsigned cindex = c_mtx.Get1dIndex(i, 0);

                //N = 8
                //for(unsigned j = 0; j < N; ++j)
                {
                    //const unsigned bindex = b_mtx.Get1dIndex(k, j);
                    //const unsigned cindex = c_mtx.Get1dIndex(i, j);
                    //f_accum(p_c_thread[cindex], p_a_thread[aindex] * p_b_thread[bindex]);

                    asm volatile("\n \
                            v_mac_f32 %0, %8, %9 \n \
                            v_mac_f32 %1, %8, %10 \n \
                            v_mac_f32 %2, %8, %11 \n \
                            v_mac_f32 %3, %8, %12 \n \
                            v_mac_f32 %4, %8, %13 \n \
                            v_mac_f32 %5, %8, %14 \n \
                            v_mac_f32 %6, %8, %15 \n \
                            v_mac_f32 %7, %8, %16 \n \
                            "
                            : "=v"(p_c_thread[cindex + 0]),"=v"(p_c_thread[cindex + 1]),"=v"(p_c_thread[cindex + 2]),"=v"(p_c_thread[cindex + 3]),"=v"(p_c_thread[cindex + 4]),"=v"(p_c_thread[cindex + 5]),"=v"(p_c_thread[cindex + 6]),"=v"(p_c_thread[cindex + 7])
                            : "v"(p_a_thread[aindex]), "v"(p_b_thread[bindex + 0]), "v"(p_b_thread[bindex + 1]),"v"(p_b_thread[bindex + 2]),"v"(p_b_thread[bindex + 3]),"v"(p_b_thread[bindex + 4]),"v"(p_b_thread[bindex + 5]),"v"(p_b_thread[bindex + 6]),"v"(p_b_thread[bindex + 7]),
                            "0"(p_c_thread[cindex + 0]),"1"(p_c_thread[cindex + 1]),"2"(p_c_thread[cindex + 2]),"3"(p_c_thread[cindex + 3]),"4"(p_c_thread[cindex + 4]),"5"(p_c_thread[cindex + 5]),"6"(p_c_thread[cindex + 6]),"7"(p_c_thread[cindex + 7])
                            );
                }
            }
        }
    }
    else
    {
        // not implemented
        assert(false);
    }
}
