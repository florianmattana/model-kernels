#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("attention_int8(Tensor! out, Tensor input) -> ()");
#if defined(CPU_KERNEL)
  ops.impl("attention_int8", torch::kCPU, &attention_int8);
#elif defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("attention_int8", torch::kCUDA, &attention_int8);
#elif defined(METAL_KERNEL)
  ops.impl("attention_int8", torch::kMPS, attention_int8);
#elif defined(XPU_KERNEL)
  ops.impl("attention_int8", torch::kXPU, &attention_int8);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
