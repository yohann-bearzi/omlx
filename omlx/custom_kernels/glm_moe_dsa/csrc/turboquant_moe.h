#pragma once

#include "mlx/array.h"
#include "mlx/utils.h"

namespace omlx::glm_kernels {

mlx::core::array turboquant_gather_blocks(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& norms,
    const mlx::core::array& codebook,
    const mlx::core::array& block_meta,
    const mlx::core::array& block_count,
    int variant,
    int bits,
    mlx::core::StreamOrDevice s = {});

}  // namespace omlx::glm_kernels
