/*
Inspired by :
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}
*/

#include "gemm_cuda_quick.h"
#include "dequantize_quick.cuh"

#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>


__device__ void compute_gemm(half const* A_shared, int const* B_ptr_local, float* C_warp,
                              half* B_loaded_zeros, half* B_loaded_scales)
{
  half A_warp[8];

  // Load B
  uint4 B_loaded = *(uint4*)(B_ptr_local);
  uint4 B_loaded_zero = *(uint4*)(B_loaded_zeros);
  uint4 B_loaded_scale = *(uint4*)(B_loaded_scales);
  uint4 B_loaded_zero_2 = *(uint4*)(B_loaded_zeros+8);
  uint4 B_loaded_scale_2 = *(uint4*)(B_loaded_scales+8);

  // Copy A
  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((&(A_shared[0]) + ((threadIdx.x & 15) * 32) + (threadIdx.x >> 4) * 8))
    );

    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_warp))[0]),
        "=r"(((unsigned *)(A_warp))[1]),
        "=r"(((unsigned *)(A_warp))[2]),
        "=r"(((unsigned *)(A_warp))[3])
      : "r"(addr)
    );
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.x);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.y));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp))[0]),
        "=f"(((float *)(C_warp))[1]),
        "=f"(((float *)(C_warp))[2]),
        "=f"(((float *)(C_warp))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp))[0]),
        "f"(((float *)(C_warp))[1]),
        "f"(((float *)(C_warp))[2]),
        "f"(((float *)(C_warp))[3]));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 4))[0]),
        "=f"(((float *)(C_warp + 4))[1]),
        "=f"(((float *)(C_warp + 4))[2]),
        "=f"(((float *)(C_warp + 4))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 4))[0]),
        "f"(((float *)(C_warp + 4))[1]),
        "f"(((float *)(C_warp + 4))[2]),
        "f"(((float *)(C_warp + 4))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.y);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 8))[0]),
        "=f"(((float *)(C_warp + 8))[1]),
        "=f"(((float *)(C_warp + 8))[2]),
        "=f"(((float *)(C_warp + 8))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 8))[0]),
        "f"(((float *)(C_warp + 8))[1]),
        "f"(((float *)(C_warp + 8))[2]),
        "f"(((float *)(C_warp + 8))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 12))[0]),
        "=f"(((float *)(C_warp + 12))[1]),
        "=f"(((float *)(C_warp + 12))[2]),
        "=f"(((float *)(C_warp + 12))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 12))[0]),
        "f"(((float *)(C_warp + 12))[1]),
        "f"(((float *)(C_warp + 12))[2]),
        "f"(((float *)(C_warp + 12))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.z);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.y));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 16))[0]),
        "=f"(((float *)(C_warp + 16))[1]),
        "=f"(((float *)(C_warp + 16))[2]),
        "=f"(((float *)(C_warp + 16))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 16))[0]),
        "f"(((float *)(C_warp + 16))[1]),
        "f"(((float *)(C_warp + 16))[2]),
        "f"(((float *)(C_warp + 16))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 20))[0]),
        "=f"(((float *)(C_warp + 20))[1]),
        "=f"(((float *)(C_warp + 20))[2]),
        "=f"(((float *)(C_warp + 20))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 20))[0]),
        "f"(((float *)(C_warp + 20))[1]),
        "f"(((float *)(C_warp + 20))[2]),
        "f"(((float *)(C_warp + 20))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.w);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.w));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 24))[0]),
        "=f"(((float *)(C_warp + 24))[1]),
        "=f"(((float *)(C_warp + 24))[2]),
        "=f"(((float *)(C_warp + 24))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 24))[0]),
        "f"(((float *)(C_warp + 24))[1]),
        "f"(((float *)(C_warp + 24))[2]),
        "f"(((float *)(C_warp + 24))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 28))[0]),
        "=f"(((float *)(C_warp + 28))[1]),
        "=f"(((float *)(C_warp + 28))[2]),
        "=f"(((float *)(C_warp + 28))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 28))[0]),
        "f"(((float *)(C_warp + 28))[1]),
        "f"(((float *)(C_warp + 28))[2]),
        "f"(((float *)(C_warp + 28))[3]));
  }

  // Load next B
  uint4 B_loaded_2 = *(uint4*)(B_ptr_local + 4);

  // Load A
  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((&(A_shared[16]) + ((threadIdx.x & 15) * 32) + (threadIdx.x >> 4) * 8))
    );

    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_warp))[0]),
        "=r"(((unsigned *)(A_warp))[1]),
        "=r"(((unsigned *)(A_warp))[2]),
        "=r"(((unsigned *)(A_warp))[3])
      : "r"(addr)
    );
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.x);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.y));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp))[0]),
        "=f"(((float *)(C_warp))[1]),
        "=f"(((float *)(C_warp))[2]),
        "=f"(((float *)(C_warp))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp))[0]),
        "f"(((float *)(C_warp))[1]),
        "f"(((float *)(C_warp))[2]),
        "f"(((float *)(C_warp))[3]));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 4))[0]),
        "=f"(((float *)(C_warp + 4))[1]),
        "=f"(((float *)(C_warp + 4))[2]),
        "=f"(((float *)(C_warp + 4))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 4))[0]),
        "f"(((float *)(C_warp + 4))[1]),
        "f"(((float *)(C_warp + 4))[2]),
        "f"(((float *)(C_warp + 4))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.y);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 8))[0]),
        "=f"(((float *)(C_warp + 8))[1]),
        "=f"(((float *)(C_warp + 8))[2]),
        "=f"(((float *)(C_warp + 8))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 8))[0]),
        "f"(((float *)(C_warp + 8))[1]),
        "f"(((float *)(C_warp + 8))[2]),
        "f"(((float *)(C_warp + 8))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 12))[0]),
        "=f"(((float *)(C_warp + 12))[1]),
        "=f"(((float *)(C_warp + 12))[2]),
        "=f"(((float *)(C_warp + 12))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 12))[0]),
        "f"(((float *)(C_warp + 12))[1]),
        "f"(((float *)(C_warp + 12))[2]),
        "f"(((float *)(C_warp + 12))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.z);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.y));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 16))[0]),
        "=f"(((float *)(C_warp + 16))[1]),
        "=f"(((float *)(C_warp + 16))[2]),
        "=f"(((float *)(C_warp + 16))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 16))[0]),
        "f"(((float *)(C_warp + 16))[1]),
        "f"(((float *)(C_warp + 16))[2]),
        "f"(((float *)(C_warp + 16))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 20))[0]),
        "=f"(((float *)(C_warp + 20))[1]),
        "=f"(((float *)(C_warp + 20))[2]),
        "=f"(((float *)(C_warp + 20))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 20))[0]),
        "f"(((float *)(C_warp + 20))[1]),
        "f"(((float *)(C_warp + 20))[2]),
        "f"(((float *)(C_warp + 20))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.w);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.w));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 24))[0]),
        "=f"(((float *)(C_warp + 24))[1]),
        "=f"(((float *)(C_warp + 24))[2]),
        "=f"(((float *)(C_warp + 24))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 24))[0]),
        "f"(((float *)(C_warp + 24))[1]),
        "f"(((float *)(C_warp + 24))[2]),
        "f"(((float *)(C_warp + 24))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 28))[0]),
        "=f"(((float *)(C_warp + 28))[1]),
        "=f"(((float *)(C_warp + 28))[2]),
        "=f"(((float *)(C_warp + 28))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 28))[0]),
        "f"(((float *)(C_warp + 28))[1]),
        "f"(((float *)(C_warp + 28))[2]),
        "f"(((float *)(C_warp + 28))[3]));
  }
}


__device__ void compute_gemm_x2(half const* A_shared, int const* B_ptr_local, float* C_warp,
                                half* B_loaded_zeros, half* B_loaded_scales)
{
  half A_warp[16];

  // Load B
  uint4 B_loaded = *(uint4*)(B_ptr_local);
  uint4 B_loaded_zero = *(uint4*)(B_loaded_zeros);
  uint4 B_loaded_scale = *(uint4*)(B_loaded_scales);
  uint4 B_loaded_zero_2 = *(uint4*)(B_loaded_zeros+8);
  uint4 B_loaded_scale_2 = *(uint4*)(B_loaded_scales+8);

  // Copy A
  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((&(A_shared[0]) + ((threadIdx.x & 15) * 32) + (threadIdx.x >> 4) * 8))
    );

    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_warp))[0]),
        "=r"(((unsigned *)(A_warp))[1]),
        "=r"(((unsigned *)(A_warp))[2]),
        "=r"(((unsigned *)(A_warp))[3])
      : "r"(addr)
    );

    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.x);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.y));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp))[0]),
        "=f"(((float *)(C_warp))[1]),
        "=f"(((float *)(C_warp))[2]),
        "=f"(((float *)(C_warp))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp))[0]),
        "f"(((float *)(C_warp))[1]),
        "f"(((float *)(C_warp))[2]),
        "f"(((float *)(C_warp))[3]));

    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((&(A_shared[16 * 32]) + ((threadIdx.x & 15) * 32) + (threadIdx.x >> 4) * 8))
    );

    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_warp))[4]),
        "=r"(((unsigned *)(A_warp))[5]),
        "=r"(((unsigned *)(A_warp))[6]),
        "=r"(((unsigned *)(A_warp))[7])
      : "r"(addr)
    );

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 4))[0]),
        "=f"(((float *)(C_warp + 4))[1]),
        "=f"(((float *)(C_warp + 4))[2]),
        "=f"(((float *)(C_warp + 4))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 4))[0]),
        "f"(((float *)(C_warp + 4))[1]),
        "f"(((float *)(C_warp + 4))[2]),
        "f"(((float *)(C_warp + 4))[3]));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 32))[0]),
        "=f"(((float *)(C_warp + 32))[1]),
        "=f"(((float *)(C_warp + 32))[2]),
        "=f"(((float *)(C_warp + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 32))[0]),
        "f"(((float *)(C_warp + 32))[1]),
        "f"(((float *)(C_warp + 32))[2]),
        "f"(((float *)(C_warp + 32))[3]));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 4 + 32))[0]),
        "=f"(((float *)(C_warp + 4 + 32))[1]),
        "=f"(((float *)(C_warp + 4 + 32))[2]),
        "=f"(((float *)(C_warp + 4 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 4 + 32))[0]),
        "f"(((float *)(C_warp + 4 + 32))[1]),
        "f"(((float *)(C_warp + 4 + 32))[2]),
        "f"(((float *)(C_warp + 4 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.y);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 8))[0]),
        "=f"(((float *)(C_warp + 8))[1]),
        "=f"(((float *)(C_warp + 8))[2]),
        "=f"(((float *)(C_warp + 8))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 8))[0]),
        "f"(((float *)(C_warp + 8))[1]),
        "f"(((float *)(C_warp + 8))[2]),
        "f"(((float *)(C_warp + 8))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 12))[0]),
        "=f"(((float *)(C_warp + 12))[1]),
        "=f"(((float *)(C_warp + 12))[2]),
        "=f"(((float *)(C_warp + 12))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 12))[0]),
        "f"(((float *)(C_warp + 12))[1]),
        "f"(((float *)(C_warp + 12))[2]),
        "f"(((float *)(C_warp + 12))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 8 + 32))[0]),
        "=f"(((float *)(C_warp + 8 + 32))[1]),
        "=f"(((float *)(C_warp + 8 + 32))[2]),
        "=f"(((float *)(C_warp + 8 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 8 + 32))[0]),
        "f"(((float *)(C_warp + 8 + 32))[1]),
        "f"(((float *)(C_warp + 8 + 32))[2]),
        "f"(((float *)(C_warp + 8 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 12 + 32))[0]),
        "=f"(((float *)(C_warp + 12 + 32))[1]),
        "=f"(((float *)(C_warp + 12 + 32))[2]),
        "=f"(((float *)(C_warp + 12 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 12 + 32))[0]),
        "f"(((float *)(C_warp + 12 + 32))[1]),
        "f"(((float *)(C_warp + 12 + 32))[2]),
        "f"(((float *)(C_warp + 12 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.z);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.y));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 16))[0]),
        "=f"(((float *)(C_warp + 16))[1]),
        "=f"(((float *)(C_warp + 16))[2]),
        "=f"(((float *)(C_warp + 16))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 16))[0]),
        "f"(((float *)(C_warp + 16))[1]),
        "f"(((float *)(C_warp + 16))[2]),
        "f"(((float *)(C_warp + 16))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 20))[0]),
        "=f"(((float *)(C_warp + 20))[1]),
        "=f"(((float *)(C_warp + 20))[2]),
        "=f"(((float *)(C_warp + 20))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 20))[0]),
        "f"(((float *)(C_warp + 20))[1]),
        "f"(((float *)(C_warp + 20))[2]),
        "f"(((float *)(C_warp + 20))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 16 + 32))[0]),
        "=f"(((float *)(C_warp + 16 + 32))[1]),
        "=f"(((float *)(C_warp + 16 + 32))[2]),
        "=f"(((float *)(C_warp + 16 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 16 + 32))[0]),
        "f"(((float *)(C_warp + 16 + 32))[1]),
        "f"(((float *)(C_warp + 16 + 32))[2]),
        "f"(((float *)(C_warp + 16 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 20 + 32))[0]),
        "=f"(((float *)(C_warp + 20 + 32))[1]),
        "=f"(((float *)(C_warp + 20 + 32))[2]),
        "=f"(((float *)(C_warp + 20 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 20 + 32))[0]),
        "f"(((float *)(C_warp + 20 + 32))[1]),
        "f"(((float *)(C_warp + 20 + 32))[2]),
        "f"(((float *)(C_warp + 20 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.w);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.w));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 24))[0]),
        "=f"(((float *)(C_warp + 24))[1]),
        "=f"(((float *)(C_warp + 24))[2]),
        "=f"(((float *)(C_warp + 24))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 24))[0]),
        "f"(((float *)(C_warp + 24))[1]),
        "f"(((float *)(C_warp + 24))[2]),
        "f"(((float *)(C_warp + 24))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 28))[0]),
        "=f"(((float *)(C_warp + 28))[1]),
        "=f"(((float *)(C_warp + 28))[2]),
        "=f"(((float *)(C_warp + 28))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 28))[0]),
        "f"(((float *)(C_warp + 28))[1]),
        "f"(((float *)(C_warp + 28))[2]),
        "f"(((float *)(C_warp + 28))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 24 + 32))[0]),
        "=f"(((float *)(C_warp + 24 + 32))[1]),
        "=f"(((float *)(C_warp + 24 + 32))[2]),
        "=f"(((float *)(C_warp + 24 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 24 + 32))[0]),
        "f"(((float *)(C_warp + 24 + 32))[1]),
        "f"(((float *)(C_warp + 24 + 32))[2]),
        "f"(((float *)(C_warp + 24 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 28 + 32))[0]),
        "=f"(((float *)(C_warp + 28 + 32))[1]),
        "=f"(((float *)(C_warp + 28 + 32))[2]),
        "=f"(((float *)(C_warp + 28 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 28 + 32))[0]),
        "f"(((float *)(C_warp + 28 + 32))[1]),
        "f"(((float *)(C_warp + 28 + 32))[2]),
        "f"(((float *)(C_warp + 28 + 32))[3]));
  }

  // Load next B
  uint4 B_loaded_2 = *(uint4*)(B_ptr_local + 4);

  // Copy A
  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((&(A_shared[16]) + ((threadIdx.x & 15) * 32) + (threadIdx.x >> 4) * 8))
    );

    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_warp))[0]),
        "=r"(((unsigned *)(A_warp))[1]),
        "=r"(((unsigned *)(A_warp))[2]),
        "=r"(((unsigned *)(A_warp))[3])
      : "r"(addr)
    );

    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.x);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.y));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp))[0]),
        "=f"(((float *)(C_warp))[1]),
        "=f"(((float *)(C_warp))[2]),
        "=f"(((float *)(C_warp))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp))[0]),
        "f"(((float *)(C_warp))[1]),
        "f"(((float *)(C_warp))[2]),
        "f"(((float *)(C_warp))[3]));

    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((&(A_shared[16 + 16 * 32]) + ((threadIdx.x & 15) * 32) + (threadIdx.x >> 4) * 8))
    );

    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_warp))[4]),
        "=r"(((unsigned *)(A_warp))[5]),
        "=r"(((unsigned *)(A_warp))[6]),
        "=r"(((unsigned *)(A_warp))[7])
      : "r"(addr)
    );

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 4))[0]),
        "=f"(((float *)(C_warp + 4))[1]),
        "=f"(((float *)(C_warp + 4))[2]),
        "=f"(((float *)(C_warp + 4))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 4))[0]),
        "f"(((float *)(C_warp + 4))[1]),
        "f"(((float *)(C_warp + 4))[2]),
        "f"(((float *)(C_warp + 4))[3]));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 32))[0]),
        "=f"(((float *)(C_warp + 32))[1]),
        "=f"(((float *)(C_warp + 32))[2]),
        "=f"(((float *)(C_warp + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 32))[0]),
        "f"(((float *)(C_warp + 32))[1]),
        "f"(((float *)(C_warp + 32))[2]),
        "f"(((float *)(C_warp + 32))[3]));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 4 + 32))[0]),
        "=f"(((float *)(C_warp + 4 + 32))[1]),
        "=f"(((float *)(C_warp + 4 + 32))[2]),
        "=f"(((float *)(C_warp + 4 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 4 + 32))[0]),
        "f"(((float *)(C_warp + 4 + 32))[1]),
        "f"(((float *)(C_warp + 4 + 32))[2]),
        "f"(((float *)(C_warp + 4 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.y);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 8))[0]),
        "=f"(((float *)(C_warp + 8))[1]),
        "=f"(((float *)(C_warp + 8))[2]),
        "=f"(((float *)(C_warp + 8))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 8))[0]),
        "f"(((float *)(C_warp + 8))[1]),
        "f"(((float *)(C_warp + 8))[2]),
        "f"(((float *)(C_warp + 8))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 12))[0]),
        "=f"(((float *)(C_warp + 12))[1]),
        "=f"(((float *)(C_warp + 12))[2]),
        "=f"(((float *)(C_warp + 12))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 12))[0]),
        "f"(((float *)(C_warp + 12))[1]),
        "f"(((float *)(C_warp + 12))[2]),
        "f"(((float *)(C_warp + 12))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 8 + 32))[0]),
        "=f"(((float *)(C_warp + 8 + 32))[1]),
        "=f"(((float *)(C_warp + 8 + 32))[2]),
        "=f"(((float *)(C_warp + 8 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 8 + 32))[0]),
        "f"(((float *)(C_warp + 8 + 32))[1]),
        "f"(((float *)(C_warp + 8 + 32))[2]),
        "f"(((float *)(C_warp + 8 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 12 + 32))[0]),
        "=f"(((float *)(C_warp + 12 + 32))[1]),
        "=f"(((float *)(C_warp + 12 + 32))[2]),
        "=f"(((float *)(C_warp + 12 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 12 + 32))[0]),
        "f"(((float *)(C_warp + 12 + 32))[1]),
        "f"(((float *)(C_warp + 12 + 32))[2]),
        "f"(((float *)(C_warp + 12 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.z);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.y));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 16))[0]),
        "=f"(((float *)(C_warp + 16))[1]),
        "=f"(((float *)(C_warp + 16))[2]),
        "=f"(((float *)(C_warp + 16))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 16))[0]),
        "f"(((float *)(C_warp + 16))[1]),
        "f"(((float *)(C_warp + 16))[2]),
        "f"(((float *)(C_warp + 16))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 20))[0]),
        "=f"(((float *)(C_warp + 20))[1]),
        "=f"(((float *)(C_warp + 20))[2]),
        "=f"(((float *)(C_warp + 20))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 20))[0]),
        "f"(((float *)(C_warp + 20))[1]),
        "f"(((float *)(C_warp + 20))[2]),
        "f"(((float *)(C_warp + 20))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 16 + 32))[0]),
        "=f"(((float *)(C_warp + 16 + 32))[1]),
        "=f"(((float *)(C_warp + 16 + 32))[2]),
        "=f"(((float *)(C_warp + 16 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 16 + 32))[0]),
        "f"(((float *)(C_warp + 16 + 32))[1]),
        "f"(((float *)(C_warp + 16 + 32))[2]),
        "f"(((float *)(C_warp + 16 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 20 + 32))[0]),
        "=f"(((float *)(C_warp + 20 + 32))[1]),
        "=f"(((float *)(C_warp + 20 + 32))[2]),
        "=f"(((float *)(C_warp + 20 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 20 + 32))[0]),
        "f"(((float *)(C_warp + 20 + 32))[1]),
        "f"(((float *)(C_warp + 20 + 32))[2]),
        "f"(((float *)(C_warp + 20 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.w);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.w));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 24))[0]),
        "=f"(((float *)(C_warp + 24))[1]),
        "=f"(((float *)(C_warp + 24))[2]),
        "=f"(((float *)(C_warp + 24))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 24))[0]),
        "f"(((float *)(C_warp + 24))[1]),
        "f"(((float *)(C_warp + 24))[2]),
        "f"(((float *)(C_warp + 24))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 28))[0]),
        "=f"(((float *)(C_warp + 28))[1]),
        "=f"(((float *)(C_warp + 28))[2]),
        "=f"(((float *)(C_warp + 28))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 28))[0]),
        "f"(((float *)(C_warp + 28))[1]),
        "f"(((float *)(C_warp + 28))[2]),
        "f"(((float *)(C_warp + 28))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 24 + 32))[0]),
        "=f"(((float *)(C_warp + 24 + 32))[1]),
        "=f"(((float *)(C_warp + 24 + 32))[2]),
        "=f"(((float *)(C_warp + 24 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 24 + 32))[0]),
        "f"(((float *)(C_warp + 24 + 32))[1]),
        "f"(((float *)(C_warp + 24 + 32))[2]),
        "f"(((float *)(C_warp + 24 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 28 + 32))[0]),
        "=f"(((float *)(C_warp + 28 + 32))[1]),
        "=f"(((float *)(C_warp + 28 + 32))[2]),
        "=f"(((float *)(C_warp + 28 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 28 + 32))[0]),
        "f"(((float *)(C_warp + 28 + 32))[1]),
        "f"(((float *)(C_warp + 28 + 32))[2]),
        "f"(((float *)(C_warp + 28 + 32))[3]));
  }
}


__device__ void extend_scale_bias_h8_h16(half* tensor) {
  #pragma unroll
  for (int i = 15; i > 0; --i) tensor[i] = tensor[i/2];
}


__global__ void gemm_forward_4bit_cuda_quick_m1n128k32(int G, int split_k_iters, half* __restrict__ A, int* __restrict__ B, half* __restrict__ scaling_factors, int* __restrict__ zeros, int M, int IC, int OC, half* __restrict__ C)
{ // Implement GEMV for B <= 8
  float C_warp[32];
  __shared__ half A_shared[2 * 32];

  for (int i = 0; i < 32; ++i) C_warp[i] = 0;

  half* A_ptr = A + (threadIdx.x % 4) * 8;
  int* B_ptr = B + (threadIdx.y * (OC / 8) * 2 + (threadIdx.x / (128 / 8)) * (OC / 8) + blockIdx.x * (128 / 8) + (threadIdx.x % (128 / 8))) * 8;
  half* A_shared_ptr = A_shared + threadIdx.y * 32 + (threadIdx.x % 4) * 8;
  int channel = threadIdx.y * (OC / 8) * 2 + (threadIdx.x / (128 / 8)) * (OC / 8) + blockIdx.x * (128 / 8) + (threadIdx.x % (128 / 8));
  int* zeros_ptr = zeros + (channel / 4);
  half* scaling_factors_ptr = scaling_factors + (channel / 4) * 8;
  half* C_ptr = C + blockIdx.y * M * OC + blockIdx.x * 128 + threadIdx.y * 64 + (threadIdx.x % 4) * 2;

  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * 32 + blockIdx.y >= IC) k_bound -= 1;

  half B_loaded_zero[16];
  half B_loaded_scale[16];
  #pragma unroll
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx.y;
    *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    __syncthreads();
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    uint B_loaded_z = *(uint*)(zeros_ptr + ((k_0_0 * 32) / G) * (OC / 8));
    *(uint4*)B_loaded_scale = *(uint4*)(scaling_factors_ptr + ((k_0_0 * 32) / G) * OC);
    *(uint4*)B_loaded_zero = dequantize_s4_to_fp16x2_fused(B_loaded_z);
    extend_scale_bias_h8_h16(B_loaded_scale);
    extend_scale_bias_h8_h16(B_loaded_zero);
    compute_gemm(A_shared, B_ptr_local, C_warp, B_loaded_zero, B_loaded_scale);
    __syncthreads();
  }

  if (threadIdx.x < 4) {
    #pragma unroll
    for (int chunk_id = 0; chunk_id < 4; ++chunk_id) {
      __stcg((__half2*)(C_ptr + chunk_id * 16), __float22half2_rn(*(float2*)(C_warp + (chunk_id * 8))));
      __stcg((__half2*)(C_ptr + chunk_id * 16 + 8), __float22half2_rn(*(float2*)(C_warp + (chunk_id * 8) + 4)));
    }
  }
}


__global__ void gemm_forward_4bit_cuda_quick_m16n128k32(int G, int split_k_iters, half* __restrict__ A, int* __restrict__ B, half* __restrict__ scaling_factors, int* __restrict__ zeros, int M, int IC, int OC, half* __restrict__ C) 
{
  float C_warp[32];
  __shared__ half A_shared[16 * 32];

  for (int i = 0; i < 32; ++i) C_warp[i] = 0;

  int oc_block_num = ((OC + 128 - 1) / 128);
  static constexpr int row_stride_warp = 32 * 8 / 32;
  bool ld_A_flag = (blockIdx.x / oc_block_num * 16 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32) < M;

  half* A_ptr = A + (threadIdx.y * row_stride_warp + threadIdx.x / (32 / 8)) * IC + (threadIdx.x % (32 / 8)) * 8;
  int* B_ptr = B + (threadIdx.y * (OC / 8) * 2 + (threadIdx.x / (128 / 8)) * (OC / 8) + blockIdx.x * (128 / 8) + (threadIdx.x % (128 / 8))) * 8;
  half* A_shared_ptr = A_shared + threadIdx.y * 8 * 32 + threadIdx.x * 8;
  int channel = threadIdx.y * (OC / 8) * 2 + threadIdx.x / (128 / 8) * (OC / 8) + blockIdx.x * (128 / 8) + threadIdx.x % (128 / 8);
  int* zeros_ptr = zeros + (channel / 4);
  half* scaling_factors_ptr = scaling_factors + (channel / 4) * 8;
  half* C_ptr = C + blockIdx.y * M * OC + blockIdx.x * 128 + threadIdx.y * 64 + (threadIdx.x % 4) * 2;

  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * 32 + blockIdx.y >= IC) k_bound -= 1;

  half B_loaded_zero[16];
  half B_loaded_scale[16];
  #pragma unroll
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx.y;
    if (ld_A_flag) *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    __syncthreads();
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);
    uint B_loaded_z = *(uint*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    *(uint4*)B_loaded_scale = *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * OC);
    *(uint4*)B_loaded_zero = dequantize_s4_to_fp16x2_fused(B_loaded_z);
    extend_scale_bias_h8_h16(B_loaded_scale);
    extend_scale_bias_h8_h16(B_loaded_zero);
    compute_gemm(A_shared, B_ptr_local, C_warp, B_loaded_zero, B_loaded_scale);
    __syncthreads();
  }

  #pragma unroll
  for (int local_id = 0; local_id < 4; ++local_id) {
    #pragma unroll
    for (int chunk_id = 0; chunk_id < 4; ++chunk_id) {
      int row_offset = ((int)threadIdx.x) / 4 + local_id % 2 * 8;
      if (row_offset < M) __stcg((__half2*)(C_ptr + chunk_id * 16 + row_offset * OC + (local_id / 2) * 8), __float22half2_rn(*(float2*)(C_warp + (chunk_id * 8) + local_id * 2)));
    }
  }
}


__global__ void gemm_forward_4bit_cuda_quick_m32n128k32(int G, int split_k_iters, half* __restrict__ A, int* __restrict__ B, half* __restrict__ scaling_factors, int* __restrict__ zeros, int M, int IC, int OC, half* __restrict__ C) 
{
  float C_warp[32 * 2];
  __shared__ half A_shared[16 * 32 * 2];

  for (int i = 0; i < 32 * 2; ++i) C_warp[i] = 0;

  static constexpr int row_stride_warp = 32 * 8 / 32;
  const int oc_block_num = ((OC + 128 - 1) / 128);
  const bool ld_A_flag_1 = (threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32) < M;
  const bool ld_A_flag_2 = (threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32 + 16) < M;

  half* A_ptr = A + (blockIdx.x / oc_block_num * 32 + (threadIdx.y * row_stride_warp) + threadIdx.x / (32 / 8)) * IC + (threadIdx.x % (32 / 8)) * 8;
  int* B_ptr = B + (threadIdx.y * (OC / 8) * 2 + (threadIdx.x / (128 / 8)) * (OC / 8) + (blockIdx.x % oc_block_num) * (128 / 8) + (threadIdx.x % (128 / 8))) * 8;
  half* A_shared_ptr = A_shared + threadIdx.y * row_stride_warp * 32 + (threadIdx.x / (32 / 8)) * 32 + (threadIdx.x % (32 / 8)) * 8;
  int channel = threadIdx.y * (OC / 8) * 2 + (threadIdx.x / (128 / 8)) * (OC / 8) + (blockIdx.x % oc_block_num) * (128 / 8) + (threadIdx.x % (128 / 8));
  int* zeros_ptr = zeros + (channel / 4);
  half* scaling_factors_ptr = scaling_factors + (channel / 4) * 8;
  half* C_ptr = C + blockIdx.y * M * OC + (blockIdx.x % oc_block_num) * 128 + threadIdx.y * 64 + (threadIdx.x % 4) * 2;

  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * 32 + blockIdx.y >= IC) k_bound -= 1;

  half B_loaded_zero[16];
  half B_loaded_scale[16];
  #pragma unroll
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx.y;
    if(ld_A_flag_1) *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    if(ld_A_flag_2) *(uint4*)(A_shared_ptr + 16 * 32) = *(uint4*)(A_ptr + 16 * IC + (k_0_0 * 32));
    __syncthreads();
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);
    uint B_loaded_z = *(uint*)(zeros_ptr + ((k_0_0 * 32) / G) * (OC / 8));
    *(uint4*)B_loaded_scale = *(uint4*)(scaling_factors_ptr + ((k_0_0 * 32) / G) * OC);
    *(uint4*)B_loaded_zero = dequantize_s4_to_fp16x2_fused(B_loaded_z);
    extend_scale_bias_h8_h16(B_loaded_scale);
    extend_scale_bias_h8_h16(B_loaded_zero);
    compute_gemm_x2(A_shared, B_ptr_local, C_warp, B_loaded_zero, B_loaded_scale);
    __syncthreads();
  }

  const int row_offset = threadIdx.x / 4;
  #pragma unroll
  for (int chunk_id = 0; chunk_id < 4; ++chunk_id) {
    half* C_ptr_chunk = C_ptr + chunk_id * 16;
    float* C_warp_chunk = C_warp + chunk_id * 8;
    #pragma unroll
    for (int local_id = 0; local_id < 4; ++local_id) {
      const int row_offset_1 = row_offset + local_id % 2 * 8;
      const int row_offset_2 = row_offset + local_id % 2 * 8 + 16;
      if (row_offset_1 < M) __stcg((__half2*)(C_ptr_chunk + row_offset_1 * OC + (local_id / 2) * 8), __float22half2_rn(*(float2*)(C_warp_chunk      + local_id * 2)));
      if (row_offset_2 < M) __stcg((__half2*)(C_ptr_chunk + row_offset_2 * OC + (local_id / 2) * 8), __float22half2_rn(*(float2*)(C_warp_chunk + 32 + local_id * 2)));
    }
  }
}

__global__ void gemm_forward_4bit_cuda_quick_m64n128k32(int G, int split_k_iters, half* __restrict__ A, int* __restrict__ B, half* __restrict__ scaling_factors, int* __restrict__ zeros, int M, int IC, int OC, half* __restrict__ C) 
{
  float C_warp[32 * 4];
  __shared__ half A_shared[16 * 32 * 4];

  for (int i = 0; i < 32 * 4; ++i) C_warp[i] = 0;

  const int oc_block_num = ((OC + 128 - 1) / 128);
  static constexpr int row_stride_warp = 32 * 8 / 32;
  half* A_ptr = A + (blockIdx.x / oc_block_num * 64 + threadIdx.y * row_stride_warp + threadIdx.x / (32 / 8)) * IC + (threadIdx.x % (32 / 8)) * 8;
  int* B_ptr = B + (threadIdx.y * (OC / 8) * 2 + (threadIdx.x / (128 / 8)) * (OC / 8) + (blockIdx.x % oc_block_num) * (128 / 8) + (threadIdx.x % (128 / 8))) * 8;
  half* A_shared_ptr = A_shared + threadIdx.y * row_stride_warp * 32 + (threadIdx.x / (32 / 8)) * 32 + (threadIdx.x % (32 / 8)) * 8;
  int channel = threadIdx.y * (OC / 8) * 2 + threadIdx.x / (128 / 8) * (OC / 8) + (blockIdx.x % oc_block_num) * (128 / 8) + (threadIdx.x % (128 / 8));
  int* zeros_ptr = zeros + (channel / 4);
  half* scaling_factors_ptr = scaling_factors + (channel / 4) * 8;
  half* C_ptr = C + blockIdx.y * M * OC + (blockIdx.x % oc_block_num) * 128 + threadIdx.y * 64 + (threadIdx.x % 4) * 2;

  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * 32 + blockIdx.y >= IC) k_bound -= 1;

  half B_loaded_zero[16];
  half B_loaded_scale[16];
  #pragma unroll
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx.y;
    *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    *(uint4*)(A_shared_ptr + 16 * 32) = *(uint4*)(A_ptr + 16 * IC + (k_0_0 * 32));
    *(uint4*)(A_shared_ptr + 32 * 32) = *(uint4*)(A_ptr + 32 * IC + (k_0_0 * 32));
    *(uint4*)(A_shared_ptr + 48 * 32) = *(uint4*)(A_ptr + 48 * IC + (k_0_0 * 32));
    __syncthreads();
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);
    uint B_loaded_z = *(uint*)(zeros_ptr + ((k_0_0 * 32) / G) * (OC / 8));
    *(uint4*)B_loaded_scale = *(uint4*)(scaling_factors_ptr + ((k_0_0 * 32) / G) * OC);
    *(uint4*)B_loaded_zero = dequantize_s4_to_fp16x2_fused(B_loaded_z);
    extend_scale_bias_h8_h16(B_loaded_scale);
    extend_scale_bias_h8_h16(B_loaded_zero);
    compute_gemm_x2(A_shared, B_ptr_local, C_warp, B_loaded_zero, B_loaded_scale);
    compute_gemm_x2(A_shared + 32 * 32, B_ptr_local, C_warp + 64, B_loaded_zero, B_loaded_scale);
    __syncthreads();
  }

  const int row_offset = blockIdx.x / oc_block_num * 64 + threadIdx.x / 4;
  #pragma unroll
  for (int local_id = 0; local_id < 4; ++local_id) {
    const int row_offset_1 = row_offset + local_id % 2 * 8;
    const int row_offset_2 = row_offset + local_id % 2 * 8 + 16;
    const int row_offset_3 = row_offset + local_id % 2 * 8 + 32;
    const int row_offset_4 = row_offset + local_id % 2 * 8 + 48;
    half* C_ptr_local = C_ptr + (local_id / 2) * 8;
    float* C_warp_local = C_warp + local_id * 2;
    #pragma unroll
    for (int chunk_id = 0; chunk_id < 4; ++chunk_id) {
      __stcg((__half2*)(C_ptr_local + chunk_id * 16 + row_offset_1 * OC), __float22half2_rn(*(float2*)(C_warp_local + chunk_id * 8)));
      __stcg((__half2*)(C_ptr_local + chunk_id * 16 + row_offset_2 * OC), __float22half2_rn(*(float2*)(C_warp_local + chunk_id * 8 + 32)));
      __stcg((__half2*)(C_ptr_local + chunk_id * 16 + row_offset_3 * OC), __float22half2_rn(*(float2*)(C_warp_local + chunk_id * 8 + 64)));
      __stcg((__half2*)(C_ptr_local + chunk_id * 16 + row_offset_4 * OC), __float22half2_rn(*(float2*)(C_warp_local + chunk_id * 8 + 96)));
    }
  }
}

__global__ void gemm_forward_4bit_cuda_quick_m64n128k32_sub(int G, int split_k_iters, half* __restrict__ A, int* __restrict__ B, half* __restrict__ scaling_factors, int* __restrict__ zeros, int M, int IC, int OC, half* __restrict__ C) 
{
  float C_warp[32 * 4];
  __shared__ half A_shared[16 * 32 * 4];

  for (int i = 0; i < 32 * 4; ++i) C_warp[i] = 0;

  const int oc_block_num = ((OC + 128 - 1) / 128);
  static constexpr int row_stride_warp = 32 * 8 / 32;
  bool ld_A_flag_1 = (blockIdx.x / oc_block_num * 64 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32) < M;
  bool ld_A_flag_2 = (blockIdx.x / oc_block_num * 64 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32 + 16) < M;
  bool ld_A_flag_3 = (blockIdx.x / oc_block_num * 64 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32 + 32) < M;
  bool ld_A_flag_4 = (blockIdx.x / oc_block_num * 64 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32 + 48) < M;
  half* A_ptr = A + (blockIdx.x / oc_block_num * 64 + threadIdx.y * row_stride_warp + threadIdx.x / (32 / 8)) * IC + (threadIdx.x % (32 / 8)) * 8;
  int* B_ptr = B + (threadIdx.y * (OC / 8) * 2 + (threadIdx.x / (128 / 8)) * (OC / 8) + (blockIdx.x % oc_block_num) * (128 / 8) + (threadIdx.x % (128 / 8))) * 8;
  half* A_shared_ptr = A_shared + threadIdx.y * row_stride_warp * 32 + (threadIdx.x / (32 / 8)) * 32 + (threadIdx.x % (32 / 8)) * 8;
  int channel = threadIdx.y * (OC / 8) * 2 + threadIdx.x / (128 / 8) * (OC / 8) + (blockIdx.x % oc_block_num) * (128 / 8) + (threadIdx.x % (128 / 8));
  int* zeros_ptr = zeros + (channel / 4);
  half* scaling_factors_ptr = scaling_factors + (channel / 4) * 8;
  half* C_ptr = C + blockIdx.y * M * OC + (blockIdx.x % oc_block_num) * 128 + threadIdx.y * 64 + (threadIdx.x % 4) * 2;

  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * 32 + blockIdx.y >= IC) k_bound -= 1;

  half B_loaded_zero[16];
  half B_loaded_scale[16];
  #pragma unroll
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx.y;
    if (ld_A_flag_1) *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    if (ld_A_flag_2) *(uint4*)(A_shared_ptr + 16 * 32) = *(uint4*)(A_ptr + 16 * IC + (k_0_0 * 32));
    if (ld_A_flag_3) *(uint4*)(A_shared_ptr + 32 * 32) = *(uint4*)(A_ptr + 32 * IC + (k_0_0 * 32));
    if (ld_A_flag_4) *(uint4*)(A_shared_ptr + 48 * 32) = *(uint4*)(A_ptr + 48 * IC + (k_0_0 * 32));
    __syncthreads();
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);
    uint B_loaded_z = *(uint*)(zeros_ptr + ((k_0_0 * 32) / G) * (OC / 8));
    *(uint4*)B_loaded_scale = *(uint4*)(scaling_factors_ptr + ((k_0_0 * 32) / G) * OC);
    *(uint4*)B_loaded_zero = dequantize_s4_to_fp16x2_fused(B_loaded_z);
    extend_scale_bias_h8_h16(B_loaded_scale);
    extend_scale_bias_h8_h16(B_loaded_zero);
    compute_gemm_x2(A_shared, B_ptr_local, C_warp, B_loaded_zero, B_loaded_scale);
    compute_gemm_x2(A_shared + 32 * 32, B_ptr_local, C_warp + 64, B_loaded_zero, B_loaded_scale);
    __syncthreads();
  }

  const int row_offset = blockIdx.x / oc_block_num * 64 + threadIdx.x / 4;
  #pragma unroll
  for (int local_id = 0; local_id < 4; ++local_id) {
    const int row_offset_1 = row_offset + local_id % 2 * 8;
    const int row_offset_2 = row_offset + local_id % 2 * 8 + 16;
    const int row_offset_3 = row_offset + local_id % 2 * 8 + 32;
    const int row_offset_4 = row_offset + local_id % 2 * 8 + 48;
    half* C_ptr_local = C_ptr + (local_id / 2) * 8;
    float* C_warp_local = C_warp + local_id * 2;
    #pragma unroll
    for (int chunk_id = 0; chunk_id < 4; ++chunk_id) {
      if (row_offset_1 < M) __stcg((__half2*)(C_ptr_local + chunk_id * 16 + row_offset_1 * OC), __float22half2_rn(*(float2*)(C_warp_local + chunk_id * 8)));
      if (row_offset_2 < M) __stcg((__half2*)(C_ptr_local + chunk_id * 16 + row_offset_2 * OC), __float22half2_rn(*(float2*)(C_warp_local + chunk_id * 8 + 32)));
      if (row_offset_3 < M) __stcg((__half2*)(C_ptr_local + chunk_id * 16 + row_offset_3 * OC), __float22half2_rn(*(float2*)(C_warp_local + chunk_id * 8 + 64)));
      if (row_offset_4 < M) __stcg((__half2*)(C_ptr_local + chunk_id * 16 + row_offset_4 * OC), __float22half2_rn(*(float2*)(C_warp_local + chunk_id * 8 + 96)));
    }
  }
}


__global__ void reduce_psum(half* workspace, half* C, int M, int OC, int split_k_iters)
{
  const int M_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int OC_thread_idx = blockDim.y * blockIdx.y + threadIdx.y;

  if (M_thread_idx < M) {
    float psum_f[8];
    float4 psum_h = *(float4*)(workspace + M_thread_idx * OC + OC_thread_idx * 8);
    *(float2*)(psum_f)   = __half22float2(*(half2*)(&psum_h.x));
    *(float2*)(psum_f+2) = __half22float2(*(half2*)(&psum_h.y));
    *(float2*)(psum_f+4) = __half22float2(*(half2*)(&psum_h.z));
    *(float2*)(psum_f+6) = __half22float2(*(half2*)(&psum_h.w));
  
    float psum_f_d[8];
    for (int split_k_iter = 1; split_k_iter < split_k_iters; ++split_k_iter) {
      float4 psum_h_d = *(float4*)(workspace + split_k_iter * M * OC + M_thread_idx * OC + OC_thread_idx * 8);
      *(float2*)(psum_f_d)   = __half22float2(*(half2*)(&psum_h_d.x));
      *(float2*)(psum_f_d+2) = __half22float2(*(half2*)(&psum_h_d.y));
      *(float2*)(psum_f_d+4) = __half22float2(*(half2*)(&psum_h_d.z));
      *(float2*)(psum_f_d+6) = __half22float2(*(half2*)(&psum_h_d.w));
      for (int i = 0; i < 8; ++i) psum_f[i] += psum_f_d[i];
    }

    half results[8];
    *(half2*)(results)   = __float22half2_rn(*(float2*)(psum_f));
    *(half2*)(results+2) = __float22half2_rn(*(float2*)(psum_f+2));
    *(half2*)(results+4) = __float22half2_rn(*(float2*)(psum_f+4));
    *(half2*)(results+6) = __float22half2_rn(*(float2*)(psum_f+6));

    *(float4*)(C + M_thread_idx * OC + OC_thread_idx * 8) = *(float4*)(results);
  }
}


torch::Tensor gemm_forward_cuda_quick(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _workspace = torch::empty({split_k_iters, num_in_feats, _kernel.size(1) / 4 * 8}, options);
    at::Tensor _out_feats = torch::empty({num_in_feats, _kernel.size(1) / 4 * 8}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);

    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto workspace = reinterpret_cast<half*>(_workspace.data_ptr<at::Half>());
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());
    int group_size = num_in_channels / _scaling_factors.size(0);

    if (num_out_channels % 128 != 0)
        throw std::invalid_argument("OC is not multiple of cta_N = 128");
    if (num_out_channels % 8 != 0)
        throw std::invalid_argument("OC is not multiple of pack_num = 8");
    if (group_size % 32 != 0)
	      throw std::invalid_argument("Group size should be a multiple of 32");
    int oc_block_num = num_out_channels / 128 / 1;

    dim3 threads_per_block(32, 2);
    if (num_in_feats > 32) {
      if (num_in_feats % 64 == 0) {
        dim3 num_blocks_64(num_out_feats / 64 * oc_block_num, split_k_iters);
        gemm_forward_4bit_cuda_quick_m64n128k32<<<num_blocks_64, threads_per_block>>>(
            group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats, num_in_channels, num_out_channels, workspace);
        return _workspace.sum(0);
      }
      else {
        dim3 num_blocks_64((num_out_feats + 64 - 1) / 64 * oc_block_num, split_k_iters);
        gemm_forward_4bit_cuda_quick_m64n128k32_sub<<<num_blocks_64, threads_per_block>>>(
            group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats, num_in_channels, num_out_channels, workspace);
        return _workspace.sum(0);
      }
    }
    else if (num_in_feats > 16) {
      dim3 num_blocks_32(oc_block_num, split_k_iters);
      static constexpr int NUM_THREADS_REDUCE_X = 4;
      static constexpr int NUM_THREADS_REDUCE_Y = 16;
      dim3 threads_per_block_reduce(NUM_THREADS_REDUCE_X, NUM_THREADS_REDUCE_Y);
      dim3 num_blocks_reduce((num_out_feats + NUM_THREADS_REDUCE_X - 1) / NUM_THREADS_REDUCE_X, (num_out_channels / 8 + NUM_THREADS_REDUCE_Y - 1) / NUM_THREADS_REDUCE_Y);

      gemm_forward_4bit_cuda_quick_m32n128k32<<<num_blocks_32, threads_per_block>>>(
          group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats, num_in_channels, num_out_channels, workspace);
      reduce_psum<<<num_blocks_reduce, threads_per_block_reduce>>>(workspace, out_feats, num_in_feats, num_out_channels, split_k_iters);
      return _out_feats;
    }
    else if (num_in_feats == 1) {
      dim3 num_blocks_1(oc_block_num, split_k_iters);
      static constexpr int NUM_THREADS_REDUCE_X = 1;
      static constexpr int NUM_THREADS_REDUCE_Y = 16;
      dim3 threads_per_block_reduce(NUM_THREADS_REDUCE_X, NUM_THREADS_REDUCE_Y);
      dim3 num_blocks_reduce((num_out_feats + NUM_THREADS_REDUCE_X - 1) / NUM_THREADS_REDUCE_X, (num_out_channels / 8 + NUM_THREADS_REDUCE_Y - 1) / NUM_THREADS_REDUCE_Y);

      gemm_forward_4bit_cuda_quick_m1n128k32<<<num_blocks_1, threads_per_block>>>(
          group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats, num_in_channels, num_out_channels, workspace);
      reduce_psum<<<num_blocks_reduce, threads_per_block_reduce>>>(workspace, out_feats, num_in_feats, num_out_channels, split_k_iters);
      return _out_feats;
    }
    else {
      dim3 num_blocks(oc_block_num, split_k_iters);
      static constexpr int NUM_THREADS_REDUCE_X = 2;
      static constexpr int NUM_THREADS_REDUCE_Y = 16;
      dim3 threads_per_block_reduce(NUM_THREADS_REDUCE_X, NUM_THREADS_REDUCE_Y);
      dim3 num_blocks_reduce((num_out_feats + NUM_THREADS_REDUCE_X - 1) / NUM_THREADS_REDUCE_X, (num_out_channels / 8 + NUM_THREADS_REDUCE_Y - 1) / NUM_THREADS_REDUCE_Y);

      gemm_forward_4bit_cuda_quick_m16n128k32<<<num_blocks, threads_per_block>>>(
          group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats, num_in_channels, num_out_channels, workspace);
      reduce_psum<<<num_blocks_reduce, threads_per_block_reduce>>>(workspace, out_feats, num_in_feats, num_out_channels, split_k_iters);
      return _out_feats;
    }
}
