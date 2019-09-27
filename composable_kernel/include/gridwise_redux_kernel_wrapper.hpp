#ifndef CK_GRIDWISE_REDUX_KERNEL_WRAPPER
#define CK_GRIDWISE_REDUX_KERNEL_WRAPPER

template <class GridwiseRedux, class T>
__global__ void run_gridwise_redux_kernel(const T* const __restrict__ p_in_global,
                                                T* const __restrict__ p_out_global)
{
    GridwiseRedux{}.Run(p_in_global, p_out_global);
}

#endif
