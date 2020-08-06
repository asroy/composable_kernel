#!/bin/bash
 export KMOPTLLC="-mattr=+enable-ds128 -amdgpu-enable-global-sgpr-addr"
 export KMDUMPISA=1
 export KMDUMPLLVM=1
 export KMDUMPDIR=$PWD

 make -j $1
#/opt/rocm/hcc/bin/llvm-objdump -mcpu=gfx906 -source -line-numbers driver/dump-gfx906.isabin > driver/dump-gfx906.isabin.asm
