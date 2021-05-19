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
          index_t KPerWave>
struct BlockwiseGemmXdlops_km_kn_m0m1m2n_v1
{

    using CIndex = MultiIndex<2>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto xdlops_gemm = XdlopsGemm<float, MPerWave, NPerWave, KPerWave>{};

    static constexpr index_t WaveSize = 64;

    static constexpr index_t M0 = ABlockDesc{}.GetLength(I1);
    static constexpr index_t M1 = ABlockDesc{}.GetLength(I2);

    static constexpr index_t N0 = BBlockDesc{}.GetLength(I1);
    static constexpr index_t N1 = BBlockDesc{}.GetLength(I2);

    static constexpr index_t MWaves = M1 / MPerWave;
    static constexpr index_t NWaves = N1 / NPerWave;

    static constexpr index_t MRepeat = M0;
    static constexpr index_t NRepeat = N0;

    __device__ constexpr auto GetOutputLayout() const { return xdlops_gemm.GetOutputLayout(); }

    __device__ constexpr auto GetNumBlks() const
    {
        return xdlops_gemm.GetOutputLayout().GetNumBlks();
    }

    __device__ constexpr auto GetBlkSize() const
    {
        return xdlops_gemm.GetOutputLayout().GetBlkSize();
    }

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const index_t thread_id = get_thread_local_1d_id();
        const index_t waveId    = thread_id / WaveSize;
        const index_t laneId    = thread_id % WaveSize;
        const index_t waveId_m  = waveId / NWaves;
        const index_t waveId_n  = waveId % NWaves;

        if constexpr(xdlops_gemm.IsKReduction)
        {
            const index_t m_offset = waveId_m * MPerWave + xdlops_gemm.GetBlkTd(laneId);
            const index_t k_offset = xdlops_gemm.GetBlkId(laneId) * xdlops_gemm.mfma_type.k_base;
            return make_tuple(k_offset, 0, m_offset);
        }
        else
        {
            const index_t m_offset = waveId_m * MPerWave + laneId;
            const index_t k_offset = 0;
            return make_tuple(k_offset, 0, m_offset);
        }
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const index_t thread_id = get_thread_local_1d_id();
        const index_t waveId    = thread_id / WaveSize;
        const index_t laneId    = thread_id % WaveSize;
        const index_t waveId_m  = waveId / NWaves;
        const index_t waveId_n  = waveId % NWaves;

        if constexpr(xdlops_gemm.IsKReduction)
        {
            const index_t n_offset = waveId_n * NPerWave + xdlops_gemm.GetBlkTd(laneId);
            const index_t k_offset = xdlops_gemm.GetBlkId(laneId) * xdlops_gemm.mfma_type.k_base;
            return make_tuple(k_offset, 0, n_offset);
        }
        else
        {
            const index_t n_offset = waveId_n * NPerWave + laneId;
            const index_t k_offset = 0;
            return make_tuple(k_offset, 0, n_offset);
        }
    }

    __device__ static CIndex
    CalculateCThreadOriginDataIndex(const index_t m0, const index_t n0, const index_t blk_i)
    {

        const index_t waveId = get_thread_local_1d_id() / WaveSize;

        const auto thread_mtx_on_blk = xdlops_gemm.GetBeginOfThreadBlk(blk_i);

        const index_t waveId_m = waveId / NWaves;
        const index_t waveId_n = waveId % NWaves;

        const index_t row = m0 * M1 + waveId_m * MPerWave + thread_mtx_on_blk.row;
        const index_t col = n0 * N1 + waveId_n * NPerWave + thread_mtx_on_blk.col;

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
            // read A
            a_thread_copy_.Run(ABlockDesc{},
                               make_tuple(k, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I0, I0),
                               a_thread_buf);

            // read B
            b_thread_copy_.Run(BBlockDesc{},
                               make_tuple(k, I0, I0),
                               b_block_buf,
                               b_thread_desc_,
                               make_tuple(I0, I0, I0),
                               b_thread_buf);

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    xdlops_gemm.template Run<decltype(a_thread_desc_),
                                             decltype(b_thread_desc_),
                                             decltype(c_thread_desc_),
                                             m0,
                                             n0>(a_thread_buf, b_thread_buf, c_thread_buf);
                });
            });
        });
    }

    private:
    // A[K, M]
    static constexpr auto a_thread_desc_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<KPerWave>{}, Number<MRepeat>{}, I1));

    // B[K, N]
    static constexpr auto b_thread_desc_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<KPerWave>{}, Number<NRepeat>{}, I1));

    static constexpr auto c_thread_desc_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}));

    using AThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4<FloatA,
                                                                FloatA,
                                                                ABlockDesc,
                                                                decltype(a_thread_desc_),
                                                                Sequence<KPerWave, MRepeat, 1>,
                                                                Sequence<0, 1, 2>,
                                                                2,
                                                                1,
                                                                1>;

    using BThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4<FloatB,
                                                                FloatB,
                                                                BBlockDesc,
                                                                decltype(b_thread_desc_),
                                                                Sequence<KPerWave, NRepeat, 1>,
                                                                Sequence<0, 1, 2>,
                                                                2,
                                                                1,
                                                                1>;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          class ABlockDesc,
          class BBlockDesc,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave>
struct BlockwiseGemmXdlops_km_kn_m0m1m2n_v1_2x2pipeline
{

    using CIndex = MultiIndex<2>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto xdlops_gemm = XdlopsGemm<float, MPerWave, NPerWave, KPerWave>{};

    static constexpr index_t WaveSize = 64;

    static constexpr index_t M0 = ABlockDesc{}.GetLength(I1);
    static constexpr index_t M1 = ABlockDesc{}.GetLength(I2);

    static constexpr index_t N0 = BBlockDesc{}.GetLength(I1);
    static constexpr index_t N1 = BBlockDesc{}.GetLength(I2);

    static constexpr index_t MWaves = M1 / MPerWave;
    static constexpr index_t NWaves = N1 / NPerWave;

    static constexpr index_t MRepeat = M0;
    static constexpr index_t NRepeat = N0;

    __device__ constexpr auto GetOutputLayout() const { return xdlops_gemm.GetOutputLayout(); }

    __device__ constexpr auto GetNumBlks() const
    {
        return xdlops_gemm.GetOutputLayout().GetNumBlks();
    }

    __device__ constexpr auto GetBlkSize() const
    {
        return xdlops_gemm.GetOutputLayout().GetBlkSize();
    }

    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const index_t thread_id = get_thread_local_1d_id();
        const index_t waveId    = thread_id / WaveSize;
        const index_t laneId    = thread_id % WaveSize;
        const index_t waveId_m  = waveId / NWaves;
        const index_t waveId_n  = waveId % NWaves;

        if constexpr(xdlops_gemm.IsKReduction)
        {
            const index_t m_offset = waveId_m * MPerWave + xdlops_gemm.GetBlkTd(laneId);
            const index_t k_offset = xdlops_gemm.GetBlkId(laneId) * xdlops_gemm.mfma_type.k_base;
            return make_tuple(k_offset, 0, m_offset);
        }
        else
        {
            const index_t m_offset = waveId_m * MPerWave + laneId;
            const index_t k_offset = 0;
            return make_tuple(k_offset, 0, m_offset);
        }
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const index_t thread_id = get_thread_local_1d_id();
        const index_t waveId    = thread_id / WaveSize;
        const index_t laneId    = thread_id % WaveSize;
        const index_t waveId_m  = waveId / NWaves;
        const index_t waveId_n  = waveId % NWaves;

        if constexpr(xdlops_gemm.IsKReduction)
        {
            const index_t n_offset = waveId_n * NPerWave + xdlops_gemm.GetBlkTd(laneId);
            const index_t k_offset = xdlops_gemm.GetBlkId(laneId) * xdlops_gemm.mfma_type.k_base;
            return make_tuple(k_offset, 0, n_offset);
        }
        else
        {
            const index_t n_offset = waveId_n * NPerWave + laneId;
            const index_t k_offset = 0;
            return make_tuple(k_offset, 0, n_offset);
        }
    }

    __device__ static CIndex
    CalculateCThreadOriginDataIndex(const index_t m0, const index_t n0, const index_t blk_i)
    {

        const index_t waveId = get_thread_local_1d_id() / WaveSize;

        const auto thread_mtx_on_blk = xdlops_gemm.GetBeginOfThreadBlk(blk_i);

        const index_t waveId_m = waveId / NWaves;
        const index_t waveId_n = waveId % NWaves;

        const index_t row = m0 * M1 + waveId_m * MPerWave + thread_mtx_on_blk.row;
        const index_t col = n0 * N1 + waveId_n * NPerWave + thread_mtx_on_blk.col;

        return CIndex{row, col};
    }

    __device__ BlockwiseGemmXdlops_km_kn_m0m1m2n_v1_2x2pipeline()
        : a_thread_copy_{CalculateAThreadOriginDataIndex()},
          b_thread_copy_{CalculateBThreadOriginDataIndex()}
    {
        static_assert(ABlockDesc::IsKnownAtCompileTime() && BBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ABlockDesc{}.GetLength(I0) == BBlockDesc{}.GetLength(I0),
                      "wrong! K dimension not consistent");

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

        // read A_sub_0
        a_thread_copy_.Run(ABlockDesc{},
                           make_tuple(I0, I0, I0),
                           a_block_buf,
                           a_thread_desc_,
                           make_tuple(I0, I0, I0),
                           a_thread_buf);

        // read B_sub_0
        b_thread_copy_.Run(BBlockDesc{},
                           make_tuple(I0, I0, I0),
                           b_block_buf,
                           b_thread_desc_,
                           make_tuple(I0, I0, I0),
                           b_thread_buf);

        // read B_sub_1
        b_thread_copy_.Run(BBlockDesc{},
                           make_tuple(I0, I1, I0),
                           b_block_buf,
                           b_thread_desc_,
                           make_tuple(I0, I1, I0),
                           b_thread_buf);

        // read A_sub_1
        a_thread_copy_.Run(ABlockDesc{},
                           make_tuple(I0, I1, I0),
                           a_block_buf,
                           a_thread_desc_,
                           make_tuple(I0, I1, I0),
                           a_thread_buf);

        // C_sub_00 += transpose(A_sub_0) * B_sub_0
        xdlops_gemm.template Run<decltype(a_thread_desc_),
                                 decltype(b_thread_desc_),
                                 decltype(c_thread_desc_),
                                 0,
                                 0>(a_thread_buf, b_thread_buf, c_thread_buf);

        // C_sub_01 += transpose(A_sub_0) * B_sub_1
        xdlops_gemm.template Run<decltype(a_thread_desc_),
                                 decltype(b_thread_desc_),
                                 decltype(c_thread_desc_),
                                 0,
                                 1>(a_thread_buf, b_thread_buf, c_thread_buf);

        static_for<KPerWave, KPerBlock, KPerWave>{}([&](auto k) {
            // read A_sub_0
            a_thread_copy_.Run(ABlockDesc{},
                               make_tuple(k, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I0, I0),
                               a_thread_buf);

            // C_sub_10 += transpose(A_sub_1) * B_sub_0
            xdlops_gemm.template Run<decltype(a_thread_desc_),
                                     decltype(b_thread_desc_),
                                     decltype(c_thread_desc_),
                                     1,
                                     0>(a_thread_buf, b_thread_buf, c_thread_buf);

            // read B_sub_0
            b_thread_copy_.Run(BBlockDesc{},
                               make_tuple(k, I0, I0),
                               b_block_buf,
                               b_thread_desc_,
                               make_tuple(I0, I0, I0),
                               b_thread_buf);

            // C_sub_11 += transpose(A_sub_1) * B_sub_1
            xdlops_gemm.template Run<decltype(a_thread_desc_),
                                     decltype(b_thread_desc_),
                                     decltype(c_thread_desc_),
                                     1,
                                     1>(a_thread_buf, b_thread_buf, c_thread_buf);

            // read B_sub_1
            b_thread_copy_.Run(BBlockDesc{},
                               make_tuple(k, I1, I0),
                               b_block_buf,
                               b_thread_desc_,
                               make_tuple(I0, I1, I0),
                               b_thread_buf);

            // read A_sub_1
            a_thread_copy_.Run(ABlockDesc{},
                               make_tuple(k, I1, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I1, I0),
                               a_thread_buf);

            // C_sub_00 += transpose(A_sub_0) * B_sub_0
            xdlops_gemm.template Run<decltype(a_thread_desc_),
                                     decltype(b_thread_desc_),
                                     decltype(c_thread_desc_),
                                     0,
                                     0>(a_thread_buf, b_thread_buf, c_thread_buf);

            // C_sub_01 += transpose(A_sub_0) * B_sub_1
            xdlops_gemm.template Run<decltype(a_thread_desc_),
                                     decltype(b_thread_desc_),
                                     decltype(c_thread_desc_),
                                     0,
                                     1>(a_thread_buf, b_thread_buf, c_thread_buf);
        });

        // C_sub_10 += transpose(A_sub_1) * B_sub_0
        xdlops_gemm.template Run<decltype(a_thread_desc_),
                                 decltype(b_thread_desc_),
                                 decltype(c_thread_desc_),
                                 1,
                                 0>(a_thread_buf, b_thread_buf, c_thread_buf);

        // C_sub_11 += transpose(A_sub_1) * B_sub_1
        xdlops_gemm.template Run<decltype(a_thread_desc_),
                                 decltype(b_thread_desc_),
                                 decltype(c_thread_desc_),
                                 1,
                                 1>(a_thread_buf, b_thread_buf, c_thread_buf);
    }

    private:
    // A[K, M]
    static constexpr auto a_thread_desc_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<KPerWave>{}, Number<MRepeat>{}, I1));

    // B[K, N]
    static constexpr auto b_thread_desc_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<KPerWave>{}, Number<NRepeat>{}, I1));

    static constexpr auto c_thread_desc_ = make_dynamic_naive_tensor_descriptor_packed_v2(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}));

    using AThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4<FloatA,
                                                                FloatA,
                                                                ABlockDesc,
                                                                decltype(a_thread_desc_),
                                                                Sequence<KPerWave, 1, 1>,
                                                                Sequence<0, 1, 2>,
                                                                2,
                                                                1,
                                                                1>;

    using BThreadCopy = ThreadwiseDynamicTensorSliceTransfer_v4<FloatB,
                                                                FloatB,
                                                                BBlockDesc,
                                                                decltype(b_thread_desc_),
                                                                Sequence<KPerWave, 1, 1>,
                                                                Sequence<0, 1, 2>,
                                                                2,
                                                                1,
                                                                1>;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
#endif
