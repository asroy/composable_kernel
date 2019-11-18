#ifndef CK_GRIDWISE_OPERATION_KERNEL_WRAPPER
#define CK_GRIDWISE_OPERATION_KERNEL_WRAPPER

template <typename GridwiseOp, typename... Xs>
__global__ void run_gridwise_operation(GridwiseOp& gridwise_op, Xs... xs)
{
    gridwise_op.Run(xs...);
}

#endif
