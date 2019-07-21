#ifndef CK_TENSOR_VIEW_HPP
#define CK_TENSOR_VIEW_HPP

namespace ck {

template <class TData, class Desc>
struct TensorViewForNormalTensor
{
    using DataType   = TData;
    using TensorDesc = Desc;
    using Coordinate = typename TensorCoordinate<TDesc>::Coordinate;

    constexpr index_t nDim = TensorDesc::GetNumOfDimensions();

    __host__ __device__ constexpr TensorView(
        TData* p_data, Coordinate origin = Coordinate(make_zero_array<index_t, nDim>()))
        : mpData{p_data}, mOrigin{origin}
    {
    }

    // data access method
    __host__ __device__ const TData& operator[](Coordinate coord) const {}

    __host__ __device__ TData& operator()(Coordinate coord) {}

    template <class IDim, class DataPerAccess>
    __host__ __device__ static constexpr bool IsVectorAccessAllowed(IDim, DataPerAccess)
    {
        constexpr index_t length = TensorDescriptor::GetLength(IDim);
        constexpr index_t stride = TensorDescriptor::GetLength(IDim);

        return (length % DataPerAccess == 0) && (stride == 1 || DataPerAccess == 1);
    }

    private:
    DataType* mpData;      // raw data
    index_t mOriginOffset; // offset of the point of origin from pointer
};

template <class TData, class MergedDesc, class Lengths>
struct TensorViewForMergedTensor
{
};

} // namespace ck
#endif
