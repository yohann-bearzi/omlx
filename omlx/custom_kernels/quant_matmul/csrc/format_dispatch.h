#pragma once

#include <nanobind/nanobind.h>

namespace omlx::quant_matmul {

void register_format_dispatch(nanobind::module_& m);

}  // namespace omlx::quant_matmul
