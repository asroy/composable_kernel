#ifndef CK_DYNAMIC_TENSOR_COORDINATE_V1_HPP
#define CK_DYNAMIC_TENSOR_COORDINATE_V1_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor_v1.hpp"

namespace ck {

// A "tensor cooridnate" is an opaque object that represents a "point of location" inside a tensor
// At the bare minimun, user should be able to query the following information from a tensor
// coordinate:
//   1. Tensor descriptor
//   2. Location, represented in the form of multi-index
//   3. Location, represented in the form of the offset to the origin of the tensor
//   4. If the location is inside invalid area or not, e.g. the padding area of an implicitly padded
//      tensor is considered invalid, because the padding area doesn't have any physical memory
//      allocation
// A tensor cooridnate also provides following functionality:
//   1. Given step size in each dimension, update itself, or return a new tensor cooridnate, so user
//      can freely move the "point of location" inside the tensor

// wrapper class for DynamicNativeTensorCoordinate_v1 and DynamicTransformedTensorCoordinate_v1
template <typename TensorDesc>
struct DynamicTensorCoordinate_v1;

// tensor coordinate for native tensor
template <typename TensorDesc>
struct DynamicNativeTensorCoordinate_v1
{
    using type                    = DynamicNativeTensorCoordinate_v1;
    using tensor_desc_type        = TensorDesc;
    static constexpr index_t NDim = tensor_desc_type::GetNumOfDimension();
    using Index                   = MultiIndex<NDim>;

    __host__ __device__ explicit constexpr DynamicNativeTensorCoordinate_v1(
        const tensor_desc_type& tensor_desc, const Index& idx)
        : tensor_desc_{tensor_desc}, idx_{idx}, offset_{tensor_desc.CalculateOffset(idx)}
    {
    }

    __host__ __device__ constexpr auto GetTensorDescriptor() const { return tensor_desc_; }

    __host__ __device__ constexpr const auto& GetUpperIndex() const { return idx_; }

    __host__ __device__ constexpr const auto& GetIndex() const { return idx_; }

    __host__ __device__ constexpr const index_t& GetOffset() const { return offset_; }

    __host__ __device__ constexpr type operator+=(const Index& idx_diff)
    {
        // idx_ is updated here, but some (or all) of its entries may never be used
        // compiler should remove those entries as dead code
        idx_ += idx_diff;

        offset_ += tensor_desc_.CalculateOffsetDiff(idx_diff);

        return *this;
    }

    __host__ __device__ constexpr type operator-=(const Index& idx_diff)
    {
        // idx_ is updated here, but some (or all) of its entries may never be used
        // compiler should remove those entries as dead code
        idx_ -= idx_diff;

        offset_ -= tensor_desc_.CalculateOffsetDiff(idx_diff);

        return *this;
    }

    __host__ __device__ constexpr type operator+(const Index& idx_diff) const
    {
        type coord = *this;
        coord += idx_diff;
        return coord;
    }

    __host__ __device__ constexpr type operator-(const Index& idx_diff) const
    {
        type coord = *this;
        coord -= idx_diff;
        return coord;
    }

    __host__ __device__ constexpr index_t CalculateOffsetDiff(const Index& idx_diff) const
    {
        return tensor_desc_.CalculateOffsetDiff(idx_diff);
    }

    // evaluated at run-time
    __host__ __device__ constexpr bool IsUpperIndexValid() const
    {
        return tensor_desc_.IsUpperIndexValid(idx_);
    }

    // evaluated at run-time
    __host__ __device__ constexpr bool IsOffsetValid() const
    {
        // For native tensor, offset is valid if upper-index is valid
        return IsUpperIndexValid();
    }

    // evaluated at compile-time
    __host__ __device__ static constexpr bool IsOffsetValidAssumingUpperIndexIsValid()
    {
        return true;
    }

    private:
    const tensor_desc_type tensor_desc_;
    // idx_ may be saved and updated, however, the value of some (or all) of its entries may
    //   never be used. Compiler should be able to remove these entries as well as its calculation
    //   as dead code.
    // TODO: make sure compiler indeed remove these dead code
    Index idx_;
    index_t offset_;
};

// tensor coordinate for transformed tensor
template <typename TensorDesc>
struct DynamicTransformedTensorCoordinate_v1
{
    static constexpr index_t NDimUp = TensorDesc::GetNumOfDimension();
    using UpperDesc                 = TensorDesc;
    using UpperCoord                = DynamicTransformedTensorCoordinate_v1;
    using UpperIndex                = MultiIndex<NDimUp>;

    using LowerDesc  = typename UpperDesc::LowerDesc;
    using LowerCoord = typename DynamicTensorCoordinate_v1<LowerDesc>::type;

    __host__ __device__ explicit constexpr DynamicTransformedTensorCoordinate_v1(
        const UpperDesc& tensor_desc_up, const UpperIndex& idx_up)
        : tensor_desc_up_{tensor_desc_up},
          idx_up_{idx_up},
          coord_low_{tensor_desc_up.GetLowerTensorDescriptor(),
                     tensor_desc_up.CalculateLowerIndex(idx_up)}
    {
    }

    __host__ __device__ constexpr auto GetTensorDescriptor() const { return tensor_desc_up_; }

    __host__ __device__ constexpr const LowerCoord& GetLowerCoordinate() const
    {
        return coord_low_;
    }

    __host__ __device__ constexpr const UpperIndex& GetUpperIndex() const { return idx_up_; }

    __host__ __device__ constexpr const UpperIndex& GetIndex() const { return idx_up_; }

    __host__ __device__ constexpr const index_t& GetOffset() const
    {
        return GetLowerCoordinate().GetOffset();
    }

    __host__ __device__ constexpr UpperCoord operator+=(const UpperIndex& idx_up_diff)
    {
        // For transformation of multi-index difference, not all transformation functions need to
        //   know the old lower-index or the old upper-index. We pass both of them to the
        //   transformation function. The transformation function itself decides to use them or not.
        coord_low_ += tensor_desc_up_.CalculateLowerIndexDiff(
            idx_up_diff, GetLowerCoordinate().GetIndex(), GetIndex());

        // idx_up_ is updated here, but some (or all) of its entries may never be used
        // compiler should remove those entries as dead code
        idx_up_ += idx_up_diff;

        return *this;
    }

    __host__ __device__ constexpr UpperCoord operator-=(const UpperIndex& idx_up_diff)
    {
        coord_low_ -= tensor_desc_up_.CalculateLowerIndexDiff(
            idx_up_diff, GetIndex(), GetLowerCoordinate().GetIndex());

        // mIndex is updated here, but some (or all) of its entries may never be used
        // compiler should remove those entries as dead code
        idx_up_ -= idx_up_diff;

        return *this;
    }

    __host__ __device__ constexpr UpperCoord operator+(const UpperIndex& idx_up_diff) const
    {
        UpperCoord coord_up = *this;
        coord_up += idx_up_diff;
        return coord_up;
    }

    __host__ __device__ constexpr UpperCoord operator-(const UpperIndex& idx_up_diff) const
    {
        UpperCoord coord_up = *this;
        coord_up -= idx_up_diff;
        return coord_up;
    }

    // Calculate offset diff without updating tensor-coordinate
    // If idx_up_diff is know at compile time, and has only non-zero entries on linear dimensions,
    //   then all calculation can be done at compile-time.
    // TODO: this function is not compiled to expected ISA
    __host__ __device__ constexpr index_t CalculateOffsetDiff(const UpperIndex& idx_up_diff) const
    {
        // For transformation of multi-index difference, not all transformation functions need to
        //   know the old lower-index or the old upper-index. We pass both of them to the
        //   transformation function. The transformation function itself decides to use them or not.
        const auto idx_low_diff =
            tensor_desc_up_.CalculateLowerIndexDiff(idx_up_diff, coord_low_.GetIndex(), idx_up_);

        return coord_low_.CalculateOffsetDiff(idx_low_diff);
    }

    // evaluated at run-time
    __host__ __device__ constexpr bool IsUpperIndexValid() const
    {
        return tensor_desc_up_.IsUpperIndexValid(idx_up_);
    }

    // evaluted at run-time
    __host__ __device__ constexpr bool IsOffsetValid() const
    {
        return IsUpperIndexValid() && coord_low_.IsOffsetValidAssumingUpperIndexIsValid();
    }

    // most evaluatation is done at comile-time
    __host__ __device__ constexpr bool IsOffsetValidAssumingUpperIndexIsValid() const
    {
        return tensor_desc_up_.IsValidUpperIndexMappedToValidLowerIndex(idx_up_) &&
               coord_low_.IsOffsetValidAssumingUpperIndexIsValid();
    }

    private:
    const UpperDesc tensor_desc_up_;
    // idx_up_ may be calculated and updated, however, the value of some (or all) of its entries
    // may never be used. Compiler should be able to remove these entries as well as its calculation
    // as dead code.
    // TODO: make sure compiler indeed remove these dead code
    UpperIndex idx_up_;
    LowerCoord coord_low_;
};

template <index_t NDim>
__host__ __device__ constexpr auto
make_dynamic_tensor_coordinate_v1(const DynamicNativeTensorDescriptor_v1<NDim>& tensor_desc,
                                  const MultiIndex<NDim>& idx)
{
    return DynamicNativeTensorCoordinate_v1<DynamicNativeTensorDescriptor_v1<NDim>>{tensor_desc,
                                                                                    idx};
}

template <index_t NDim, typename... Ts>
__host__ __device__ constexpr auto
make_dynamic_tensor_coordinate_v1(const DynamicTransformedTensorDescriptor_v1<Ts...>& tensor_desc,
                                  const MultiIndex<NDim>& idx)
{
    static_assert(DynamicTransformedTensorDescriptor_v1<Ts...>::GetNumOfDimension() == NDim,
                  "wrong! inconsistent # of dimensions");

    return DynamicTransformedTensorCoordinate_v1<DynamicTransformedTensorDescriptor_v1<Ts...>>{
        tensor_desc, idx};
}

template <typename TensorDesc>
struct DynamicTensorCoordinate_v1
{
    static constexpr index_t NDim = TensorDesc::GetNumOfDimension();

    using type =
        decltype(make_dynamic_tensor_coordinate_v1<NDim>(TensorDesc{}, MultiIndex<NDim>{}));
};

} // namespace ck
#endif
