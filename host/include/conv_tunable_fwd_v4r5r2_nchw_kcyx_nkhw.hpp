#ifndef CONV_TUNABLE_FWD_V4R5R2_NCHW_KCYX_NKHW_HPP
#define CONV_TUNABLE_FWD_V4R5R2_NCHW_KCYX_NKHW_HPP

struct tunable_dyn_conv_fwd_v4r5r2_nchw_kcyx_nkhw
{
    int32_t BlockSize = 256;

    int32_t GM1PerBlockGM11 = 128;
    int32_t GN1PerBlockGN11 = 32;
    int32_t GK0PerBlock     = 8;

    int32_t BM1PerThreadBM11 = 4;
    int32_t BN1PerThreadBN11 = 4;
    int32_t BK0PerThread     = 1;

    int32_t BM10BN10ThreadClusterBM100 = 2;
    int32_t BM10BN10ThreadClusterBN100 = 2;
    int32_t BM10BN10ThreadClusterBM101 = 8;
    int32_t BM10BN10ThreadClusterBN101 = 8;

    std::array<int32_t, 4> ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11   = {4, 1, 1, 1};
    std::array<int32_t, 4> ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11 = {2, 1, 1, 128};
    std::array<int32_t, 4> ABlockTransferThreadClusterArrangeOrder             = {3, 2, 1, 0};
    std::array<int32_t, 4> ABlockTransferSrcAccessOrder                        = {3, 2, 1, 0};
    int32_t ABlockTransferSrcVectorDim                                         = 0;
    int32_t ABlockTransferSrcScalarPerVector                                   = 4;
    int32_t ABlockTransferDstScalarPerVector_GM11                              = 1;
    bool AThreadTransferSrcResetCoordinateAfterRun                             = false;

    std::array<int32_t, 4> BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11   = {1, 4, 1, 1};
    std::array<int32_t, 4> BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11 = {8, 1, 1, 32};
    std::array<int32_t, 4> BBlockTransferThreadClusterArrangeOrder             = {0, 3, 2, 1};
    std::array<int32_t, 4> BBlockTransferSrcAccessOrder                        = {0, 3, 2, 1};
    int32_t BBlockTransferSrcVectorDim                                         = 3;
    int32_t BBlockTransferSrcScalarPerVector                                   = 1;
    int32_t BBlockTransferDstScalarPerVector_GN11                              = 1;
    bool BThreadTransferSrcResetCoordinateAfterRun                             = false;

    std::array<int32_t, 6> CThreadTransferSrcDstAccessOrder = {3, 4, 5, 0, 1, 2};
    int32_t CThreadTransferSrcDstVectorDim                  = 5;
    int32_t CThreadTransferDstScalarPerVector               = 1;
};
#endif
