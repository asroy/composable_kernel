#!/bin/bash

rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=../../../
MY_PROJECT_INSTALL=../install.dir

cmake                                                                                       \
-D CMAKE_INSTALL_PREFIX=${MY_PROJECT_INSTALL}                                               \
-D CMAKE_BUILD_TYPE=Release                                                                 \
-D DEVICE_BACKEND="HIP"                                                                     \
-D CMAKE_CXX_COMPILER=hcc                                                                   \
-D CMAKE_CXX_FLAGS="-gline-tables-only ${HCC_FLAGS}"                                        \
-D CMAKE_EXE_LINKER_FLAGS="${HCC_FLAGS} -L/opt/rocm/lib --hip-link -lhip_hcc  "    \
-D CMAKE_PREFIX_PATH="/opt/rocm;/home/package/build/mlopen_dep"                             \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                           \
${MY_PROJECT_SOURCE}

