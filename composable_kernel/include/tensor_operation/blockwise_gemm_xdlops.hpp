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
          class ABlockDesc,
          class BBlockDesc,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          index_t MWaves,
          index_t NWaves>
struct BlockwiseGemmXdlops_km_kn_m0m1m2n_v1
{

    using CIndex = MultiIndex<2>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto XdlopsGemm = XdlopsGemm_t<float, MPerWave, NPerWave, KPerWave>{};

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

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const index_t thread_id = get_thread_local_1d_id();
        const index_t waveId    = thread_id / WaveSize;
        const index_t laneId    = thread_id % WaveSize;
        const index_t waveId_m  = waveId / NWaves;
        const index_t waveId_n  = waveId % NWaves;

        return make_tuple(0, waveId_m * MPerWave + laneId);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const index_t thread_id = get_thread_local_1d_id();
        const index_t waveId    = thread_id / WaveSize;
        const index_t laneId    = thread_id % WaveSize;
        const index_t waveId_m  = waveId / NWaves;
        const index_t waveId_n  = waveId % NWaves;

        return make_tuple(0, waveId_n * NPerWave + laneId);
    }

    template <index_t AStride = MPerWave, index_t BStride = NPerWave>
    __device__ static CIndex CalculateCThreadOriginDataIndex(index_t blk_i)
    {

        const index_t waveId = get_thread_local_1d_id() / WaveSize;

        const auto thread_mtx_on_blk = XdlopsGemm.GetBeginOfThreadBlk(blk_i);

        const index_t row = (waveId / NWaves) * AStride + thread_mtx_on_blk.row;
        const index_t col = (waveId % NWaves) * BStride + thread_mtx_on_blk.col;

        return CIndex{row, col};
    }

    __device__ BlockwiseGemmXdlops_km_kn_m0m1m2n_v1()
        : a_thread_copy_{CalculateAThreadOriginDataIndex()},
          b_thread_copy_{CalculateBThreadOriginDataIndex()}
    {
        static_assert(ABlockDesc::IsKnownAtCompileTime() && BBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ABlockDesc{}.GetLength(I0) == BBlockDesc{}.GetLength(I0),
                      "wrong! K dimension not consistent");

        constexpr index_t M = ABlockDesc{}.GetLength(I1); // A is transposed
        constexpr index_t N = BBlockDesc{}.GetLength(I1);

        static_assert(MPerWave * MWaves == M, "GemmMWaves * MPerWave != M");
        static_assert(NPerWave * NWaves == N, "GemmNWaves * NPerWave != N");

        static_assert(BlockSize == MWaves * NWaves * WaveSize,
                      "BlockSize != MWaves * NWaves * WaveSize\n");
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

        constexpr index_t KPerBlock = ABlockDesc{}.GetLength(I0);

        static_for<0, KPerBlock, KPerWave>{}([&](auto k) {
            a_thread_copy_.Run(ABlockDesc{},
                               make_tuple(k, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I0),
                               a_thread_buf);

            b_thread_copy_.Run(BBlockDesc{},
                               make_tuple(k, I0),
                               b_block_buf,
                               b_thread_desc_,
                               make_tuple(I0, I0),
                               b_thread_buf);

            XdlopsGemm.template Run(a_thread_buf, b_thread_buf, c_thread_buf);
        });
    }

    private:
    // A[K, M]
    static constexpr auto a_thread_desc_ =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(Number<KPerWave>{}, Number<1>{}));

    // B[K, N]
    static constexpr auto b_thread_desc_ =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(Number<KPerWave>{}, Number<1>{}));

    using AThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4<FloatA,
                                                                FloatA,
                                                                ABlockDesc,
                                                                decltype(a_thread_desc_),
                                                                Sequence<KPerWave, 1>,
                                                                Sequence<0, 1>,
                                                                1,
                                                                1,
                                                                1>;

    using BThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4<FloatB,
                                                                FloatB,
                                                                BBlockDesc,
                                                                decltype(b_thread_desc_),
                                                                Sequence<KPerWave, 1>,
                                                                Sequence<0, 1>,
                                                                1,
                                                                1,
                                                                1>;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
#endif
