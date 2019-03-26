#!/bin/bash
export KMDUMPISA=1

make -j driver
/opt/rocm/hcc/bin/llvm-objdump -mcpu=gfx900 -source -line-numbers driver/dump-gfx900.isabin > driver/dump-gfx900.isabin.isa
