#ifndef CK_CONSTANT_TENSOR_COORDINATE_HPP
#define CK_CONSTANT_TENSOR_COORDINATE_HPP

#include "constant_tensor_descriptor.hpp"
#include "constant_merged_tensor_descriptor.hpp"

namespace ck {

template <class TDesc>
struct TensorCoordinate;

template <class... Ts>
struct TensorCoordinate<ConstantTensorDescriptor<Ts...>>
{
    using TensorDescriptor = ConstantTensorDescriptor<Ts...>;
    using nDim             = TensorDescriptor::GetNumOfDimension();

    __host__ __device__ constexpr TensorCoordinate(Array<nDim, index_t> multi_id) {}

    template <class IDim>
    __host__ __device__ void March(IDim, index_t step_size, bool positive_direction)
    {
    }

    private:
    // multi-index
    // offset
};

template <class... Ts>
struct TensorCoordinate<ConstantMergedTensorDescriptor<Ts...>>
{
    using TensorDescriptor = ConstantMergedTensorDescriptor<Ts...>;

    template <class IDim>
    __host__ __device__ void March(IDim, index_t step_size, bool positive_direction)
    {
    }

    private:
    // multi-index
    // offset
    // original multi-index
    // partial offset
};

} // namespace ck
#endif
