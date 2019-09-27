#ifndef CK_GRIDWISE_TENSOR_REDUX_v1
#define CK_GRIDWISE_TENSOR_REDUX_v1

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMergedTensorDescriptor.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"

namespace ck {

// define B = merge(N0, Ho, Wo)
template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class InGlobalDesc,
          class OutGlobalDesc,
          class ReduxDims
#if 0
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class InBlockCopySubLengths_E_N1_B_N2,
          class InBlockCopyClusterLengths_E_N1_B_N2,
          class InBlockCopyThreadClusterArrangeOrder,
          class InBlockCopySrcAccessOrder,
          class InBlockCopyDstAccessOrder,
          index_t InBlockCopySrcDataPerRead_B,
          index_t InBlockCopyDstDataPerWrite_N2,
          class WeiBlockCopySubLengths_E_K,
          class WeiBlockCopyClusterLengths_E_K,
          class WeiBlockCopyThreadClusterArrangeOrder,
          class WeiBlockCopySrcAccessOrder,
          class WeiBlockCopyDstAccessOrder,
          index_t WeiBlockCopySrcDataPerRead_E,
          index_t WeiBlockCopyDstDataPerWrite_K
#endif
              >
struct GridwiseTensorRedux_v1
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        Float* const __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};


        constexpr auto in_n_c_h_w_global_desc  = InGlobalDesc{};
        constexpr auto out_k_h_w_global_desc = OutGlobalDesc{};

        constexpr auto N = in_n_c_h_w_global_desc.GetLength(I0);
        constexpr auto C = in_n_c_h_w_global_desc.GetLength(I1);
        constexpr auto H = in_n_c_h_w_global_desc.GetLength(I2);
        constexpr auto W = in_n_c_h_w_global_desc.GetLength(I3);
        constexpr auto total_elems = N * C * H * W;

        // constexpr auto out_k0_k1_k2_n1_h_w_thread_mem_desc =
        //     make_ConstantTensorDescriptor_packed(
        //              Sequence<1>{});
        // Float p_out_thread[out_k0_k1_k2_n1_h_w_thread_mem_desc.GetElementSpace()];
        Float p_out_thread[1];
        // TODO: assert that except the reduced dimension all sizes are the same
        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor_packed(Sequence<1, 1, 1, BlockSize>{});
        const auto thread_cluster_id = thread_cluster_desc.GetMultiIndexFrom1dIndex(get_thread_local_1d_id());

        constexpr auto block_cluster_desc = make_ConstantTensorDescriptor_packed(Sequence<1,1,1,total_elems / BlockSize>{});
        const auto block_cluster_id = block_cluster_desc.GetMultiIndexFrom1dIndex(get_block_1d_id());
        {

            const Float* p_in_thread_on_global =
                p_in_global +
                 in_n_c_h_w_global_desc.GetOffsetFromMultiIndex(block_cluster_id + thread_cluster_id);

            // constexpr auto threadwise_in_copy = ThreadwiseGenericTensorSliceCopy_v2<
            //     decltype(in_n_c_h_w_global_desc), //source
            //     decltype(out_k0_k1_k2_n1_h_w_thread_mem_desc),
            //     NormalTensorCoordinate<decltype(in_n_c_h_w_global_desc)>, //source
            //     NormalTensorCoordinate<decltype(out_k0_k1_k2_n1_h_w_thread_mem_desc)>,
            //     decltype(in_n_c_h_w_global_desc.GetLengths())>(); //source
            // threadwise_in_copy.Run(p_in_thread_on_global, p_out_thread);
            printf("block: (%d, %d), thread: (%d, %d), input: %f\n", block_cluster_id[2], block_cluster_id[3],  thread_cluster_id[2], thread_cluster_id[3], *p_in_thread_on_global);
            p_out_thread[0] = p_in_thread_on_global[0];
        }
        {
            Float* p_out_thread_on_global = 
                p_out_global ; //+ 
                 //out_k_h_w_global_desc.GetOffsetFromMultiIndex(
                 //        get_thread_local_1d_id());
            // constexpr auto threadwise_out_copy = ThreadwiseGenericTensorSliceCopy_v2<
            //     decltype(out_k0_k1_k2_n1_h_w_thread_mem_desc),
            //     decltype(out_k_h_w_global_desc),
            //     NormalTensorCoordinate<decltype(out_k0_k1_k2_n1_h_w_thread_mem_desc)>,
            //     NormalTensorCoordinate<decltype(out_k_h_w_global_desc)>,
            //     decltype(out_k0_k1_k2_n1_h_w_thread_mem_desc.GetLengths())>();
            // threadwise_out_copy.Run(p_out_thread, p_out_thread_on_global);
            auto idx = get_thread_local_1d_id();
            p_out_thread_on_global[idx] = p_out_thread[0];
        }
    }
};

} // namespace ck
#endif
