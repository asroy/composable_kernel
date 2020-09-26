rm *.ll *.s

/opt/rocm/llvm/bin/llvm-dis driver/conv_driver-hip-amdgcn-amd-amdhsa-gfx906-optimized.bc -o tmp.ll
/opt/rocm/llvm/bin/opt -S -inline -inline-threshold=104857 tmp.ll > inline.ll
/opt/rocm/llvm/bin/opt -S -O3 -sroa inline.ll > o3.ll
/opt/rocm/llvm/bin/opt -S -O3 -sroa o3.ll > o3_2.ll
/opt/rocm/llvm/bin/opt -S -O3 -sroa o3_2.ll > o3_3.ll
/opt/rocm/llvm/bin/opt -S -O3 -sroa o3_3.ll > o3_4.ll

/opt/rocm/llvm/bin/llc -mcpu=gfx908 o3.ll
/opt/rocm/llvm/bin/llc -mcpu=gfx908 o3_2.ll
/opt/rocm/llvm/bin/llc -mcpu=gfx908 o3_3.ll
/opt/rocm/llvm/bin/llc -mcpu=gfx908 o3_4.ll
