#!/bin/bash


export KMOPTLLC="-mattr=+enable-ds128 -amdgpu-enable-global-sgpr-addr"
export KMDUMPISA=1
export KMDUMPLLVM=1
export KMDUMPDIR=$PWD/build/

CONV_DRIVER_TEST=0
CONV_DRIVER_KERNEL=0

MY_PROJECT_SOURCE=../
MY_PROJECT_INSTALL=../install.dir

rm -rf build && mkdir build && cd build

rm -rf $MY_PROJECT_INSTALL && mkdir  $MY_PROJECT_INSTALL

cmake                                                                                       \
-D CMAKE_INSTALL_PREFIX=${MY_PROJECT_INSTALL}                                               \
-D CMAKE_BUILD_TYPE=Release                                                                 \
-D DEVICE_BACKEND="AMD"                                                                     \
-D HIP_HIPCC_FLAGS="${HIP_HIPCC_FLAGS} -gline-tables-only -v"                               \
-D CMAKE_CXX_FLAGS="-gline-tables-only --amdgpu-target=gfx906 -DCONV_DRIVER_TEST=$CONV_DRIVER_TEST -DCONV_DRIVER_KERNEL=$CONV_DRIVER_KERNEL" \
-D CMAKE_CXX_COMPILER=/opt/rocm/hip/bin/hipcc                                               \
-D CMAKE_PREFIX_PATH="/opt/rocm"                                                            \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                           \
${MY_PROJECT_SOURCE} || exit 1


 make -j`nproc`

cd -
ASM_KERNEL=igemm_v4r1_generic_1x1.s
ASM_HSACO=igemm_v4r1_generic_1x1.co
/opt/rocm/hcc/bin/clang -x assembler -target amdgcn--amdhsa -mcpu=gfx906 -mno-code-object-v3 \
    composable_kernel/asm/$ASM_KERNEL -o build/driver/$ASM_HSACO
/opt/rocm/hcc/bin/llvm-objdump -disassemble -mcpu=gfx906 build/driver/$ASM_HSACO > build/driver/$ASM_KERNEL.dump.s