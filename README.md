# How to build and run

# docker
```
docker run                                                                   \
-it                                                                          \
--rm                                                                         \
--privileged                                                                 \
--group-add sudo                                                             \
-w /root/workspace                                                           \
-v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace                                \
rocm/tensorflow:rocm4.2-tf2.4-dev                                            \
/bin/bash
```

# Install Boost for online compilation
https://www.boost.org/doc/libs/1_66_0/more/getting_started/unix-variants.html#easy-build-and-install


# Build
Change target ID in source code, example below is gfx908
https://github.com/asroy/modular_convolution/blob/aafb5eb18781f1ac9e06a17c3e53d968dd53dcc0/composable_kernel/include/utility/config.amd.hpp.in#L16-L23

Add path of Boost
```
 export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

```
mkdir build && cd build

# need to manually set target ID, example below is gfx908
cmake                                                                                                                              \
-D CMAKE_BUILD_TYPE=Release                                                                                                        \
-D DEVICE_BACKEND=AMD                                                                                                              \
-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx908 -mllvm --amdgpu-spill-vgpr-to-agpr=0 -gline-tables-only -save-temps=$CWD"           \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                                                          \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                                                     \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                                                                  \
..
```

Build drivers:   \
``conv_driver_v2`` is (offline compilation) driver for forward convolution,  \
``conv_bwd_data_driver_v2`` is (offline compilation) driver for backward-data convolution  \
``conv_driver_v2_olc`` is (online compilation) driver for forward convolution
```
 make -j conv_driver_v2
 make -j conv_bwd_data_driver_v2
 make -j conv_driver_v2_olc
```

# Run
* layout: 0 = NCHW; 1 = NHWC
* algo:
   * Forward convolution: https://github.com/asroy/modular_convolution/blob/aafb5eb18781f1ac9e06a17c3e53d968dd53dcc0/driver/conv_driver_v2.cpp#L38
   * Backward data convolution: https://github.com/asroy/modular_convolution/blob/aafb5eb18781f1ac9e06a17c3e53d968dd53dcc0/driver/conv_bwd_data_driver_v2.cpp#L22
* verify: 0 = no verification; 1 = do verification
* init: 0 ~ 3. initialization method
* log: 0 = no log; 1 = do log
* repeat: number of time kernel being launched
```
########################### layout  algo  verify  init  log  repeat  N__ K___ C___ Y X Hi_ Wi__ Strides Dilations LeftPads RightPads
 ./conv_driver_v2                0     6       0     3    0       1  128  256  192 3 3  71   71     2 2       1 1      1 1       1 1
 ./conv_driver_v2                1     9       0     3    0       1  128  256  192 3 3  71   71     2 2       1 1      1 1       1 1
 ./conv_driver_v2                1     9       0     3    0       1  256 1024  256 3 3  14   14     1 1       1 1      1 1       1 1
 ./conv_driver_v2                0     6       0     3    0       1  256 1024  256 3 3  14   14     1 1       1 1      1 1       1 1
 ./conv_bwd_data_driver_v2       1     1       0     3    0       1  256  256 1024 3 3  14   14     1 1       1 1      1 1       1 1
```
