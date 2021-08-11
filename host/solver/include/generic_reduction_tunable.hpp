#ifndef GENERIC_REDUCTION_TUNABLES_HPP
#define GENERIC_REDUCTION_TUNABLES_HPP

struct tunable_dyn_generic_reduction
{
    ck::index_t BlockSize;
    ck::index_t GredThreadBufferLength;
    ck::index_t GredAccessesPerThreadInBlock;
    ck::index_t GredAccessesPerThreadInWarp;
};

static struct tunable_dyn_generic_reduction default_tunable_dyn_generic_reduction = {256, 8, 2, 2};

#endif
