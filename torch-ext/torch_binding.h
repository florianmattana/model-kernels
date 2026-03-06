#pragma once

#include <torch/torch.h>

void attention_int8(torch::Tensor &out, torch::Tensor const &input);
