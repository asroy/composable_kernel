#ifndef CK_GRIDWISE_OPERATION_KERNEL_WRAPPER
#define CK_GRIDWISE_OPERATION_KERNEL_WRAPPER

template <typename GridwiseOp, typename... Xs>
__global__ void
#if 1
    __launch_bounds__(256, 2)
#endif
        run_gridwise_operation(Xs... xs)
{
    GridwiseOp{}.Run(xs...);
}

#endif
