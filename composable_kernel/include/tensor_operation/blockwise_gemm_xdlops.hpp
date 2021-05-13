#ifndef CK_BLOCKWISE_GEMM_XDLOPS_HPP
#define CK_BLOCKWISE_GEMM_XDLOPS_HPP

#include "common_header.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "xdlops_gemm.hpp"
#include "threadwise_gemm.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          class ABlockDesc,
          class BBlockDesc,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmKPerWave,
          index_t GemmMWaves,
          index_t GemmNWaves,
          index_t GemmDataPerReadA, // \todo unused parameter, remove
          index_t GemmDataPerReadB  // \todo unused parameter, remove
          >
struct BlockwiseGemmXdlops_km_kn_m0m1m2n_v1
{
    struct MatrixIndex
    {
        index_t row;
        index_t col;
    };

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto XdlopsGemm =
        XdlopsGemm_t<float, GemmMPerWave, GemmNPerWave, GemmDataPerReadA, GemmDataPerReadB>{};

    index_t mMyWaveOffsetA;
    index_t mMyWaveOffsetB;

    static constexpr index_t WaveSize = 64;

    __device__ constexpr auto GetOutputLayout() const { return XdlopsGemm.GetOutputLayout(); }

    __device__ constexpr auto GetNumBlks() const
    {
        return XdlopsGemm.GetOutputLayout().GetNumBlks();
    }

    __device__ constexpr auto GetBlkSize() const
    {
        return XdlopsGemm.GetOutputLayout().GetBlkSize();
    }

    __device__ BlockwiseGemmXdlops_km_kn_m0m1m2n_v1()
    {
        static_assert(ABlockDesc::IsKnownAtCompileTime() && BBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ABlockDesc{}.GetLength(I0) == BBlockDesc{}.GetLength(I0),
                      "wrong! K dimension not consistent");

        constexpr index_t M = ABlockDesc{}.GetLength(I1); // A is transposed
        constexpr index_t N = BBlockDesc{}.GetLength(I1);

        static_assert(GemmMPerWave * GemmMWaves == M, "GemmMWaves * GemmMPerWave != M");
        static_assert(GemmNPerWave * GemmNWaves == N, "GemmNWaves * GemmNPerWave != N");

        static_assert(BlockSize == GemmMWaves * GemmNWaves * WaveSize,
                      "BlockSize != GemmMWaves * GemmNWaves * WaveSize\n");

        const index_t waveId   = get_thread_local_1d_id() / WaveSize;
        const index_t waveId_m = waveId / GemmNWaves;
        const index_t waveId_n = waveId % GemmNWaves;

        mMyWaveOffsetA = waveId_m * GemmMPerWave;
        mMyWaveOffsetB = waveId_n * GemmNPerWave;
    }

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf =
            make_static_buffer<AddressSpace::Vgpr, FloatA>(a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf =
            make_static_buffer<AddressSpace::Vgpr, FloatB>(b_thread_desc_.GetElementSpaceSize());

#if 0
        constexpr auto threadwise_gemm =
            ThreadwiseGemm_km0m1_kn0n1_m0m1n0n1<FloatA,
                                                FloatB,
                                                FloatC,
                                                decltype(a_thread_desc_),
                                                decltype(b_thread_desc_),
                                                CThreadDesc,
                                                Sequence<GemmKPerWave>,
                                                Sequence<M0_, M1PerThread>,
                                                Sequence<N0_, N1PerThread>>{};

        constexpr index_t K = ABlockDesc{}.GetLength(I0);

        static_for<0, K, GemmKPerWave>{}([&](auto k) {
            a_thread_copy_.Run(ABlockDesc{},
                               make_tuple(k, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I0, I0),
                               a_thread_buf);

            b_thread_copy_.Run(BBlockDesc{},
                               make_tuple(k, I0, I0),
                               b_block_buf,
                               b_thread_desc_,
                               make_tuple(I0, I0, I0),
                               b_thread_buf);

            threadwise_gemm.Run(a_thread_buf,
                                make_tuple(I0, I0, I0),
                                b_thread_buf,
                                make_tuple(I0, I0, I0),
                                c_thread_buf,
                                make_tuple(I0, I0, I0, I0));
        });
#endif
    }

    template <index_t AStride = GemmMPerWave, index_t BStride = GemmNPerWave>
    __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t i)
    {

        const index_t waveId = get_thread_local_1d_id() / WaveSize;

        const auto thread_mtx_on_blk = XdlopsGemm.GetBeginOfThreadBlk(i);

        const index_t col = (waveId % GemmNWaves) * BStride + thread_mtx_on_blk.col;
        const index_t row = (waveId / GemmNWaves) * AStride + thread_mtx_on_blk.row;

        return MatrixIndex{row, col};
    }

    private:
    // A[K, M]
    static constexpr auto a_thread_desc_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<GemmKPerWave>{}, Number<1>{}));

    // B[K, N]
    static constexpr auto b_thread_desc_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<GemmKPerWave>{}, Number<1>{}));

    using AThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4<FloatA,
                                                                FloatA,
                                                                ABlockDesc,
                                                                decltype(a_thread_desc_),
                                                                Sequence<GemmKPerWave, 1>,
                                                                Sequence<0, 1>,
                                                                1,
                                                                1,
                                                                1>;

    using BThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4<FloatB,
                                                                FloatB,
                                                                BBlockDesc,
                                                                decltype(b_thread_desc_),
                                                                Sequence<GemmKPerWave, 1>,
                                                                Sequence<0, 1>,
                                                                1,
                                                                1,
                                                                1>;

    // AThreadCopy a_thread_copy_;
    // BThreadCopy b_thread_copy_;
};

} // namespace ck
#endif
