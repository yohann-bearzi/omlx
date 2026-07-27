#include "turboquant_moe.h"

#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <string>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;

std::string tq_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&tq_binary_dir), &info)) {
      throw std::runtime_error("Unable to get omlx_glm_kernels binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

bool tq_row_contiguous(const array& arr) {
  return arr.flags().row_contiguous && arr.strides(-1) == 1;
}

struct TqBlocksVariant {
  int bm, bn, bk, wm, wn;
};

TqBlocksVariant tq_blocks_variant(int variant) {
  switch (variant) {
    case 0: return {8, 32, 32, 1, 2};
    case 1: return {16, 32, 32, 1, 2};
    case 2: return {32, 32, 32, 1, 2};
    case 3: return {16, 64, 32, 1, 2};
    case 4: return {32, 64, 32, 1, 2};
    default: {
      std::ostringstream msg;
      msg << "Unsupported TurboQuant blocks variant: " << variant << ".";
      throw std::invalid_argument(msg.str());
    }
  }
}

std::string tq_type_name(Dtype dtype) {
  if (dtype == float16) return "float16_t";
  if (dtype == bfloat16) return "bfloat16_t";
  std::ostringstream msg;
  msg << "Unsupported TurboQuant kernel dtype: " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

class TurboQuantGatherBlocksPrimitive : public Primitive {
 public:
  TurboQuantGatherBlocksPrimitive(Stream stream, int variant, int bits)
      : Primitive(stream), variant_(variant), bits_(bits) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "TurboQuantGatherBlocksPrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& norms = inputs[2];
    const auto& codebook = inputs[3];
    const auto& block_meta = inputs[4];
    const auto& block_count = inputs[5];

    out.set_data(allocator::malloc(out.nbytes()));

    const auto cfg = tq_blocks_variant(variant_);
    const int max_blocks = block_meta.shape(0);
    const int M = x.shape(0);
    const int K = x.shape(-1);
    const int N = weight.shape(1);

    std::string kname;
    concatenate(
        kname, "turboquant_gather_blocks_rhs_", tq_type_name(x.dtype()),
        "_bm_", cfg.bm, "_bn_", cfg.bn, "_bk_", cfg.bk,
        "_wm_", cfg.wm, "_wn_", cfg.wn, "_bits_", bits_);

    auto lib = d.get_library("omlx_glm_kernels", tq_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(weight, 1);
    compute_encoder.set_input_array(norms, 2);
    compute_encoder.set_input_array(codebook, 3);
    compute_encoder.set_input_array(block_meta, 4);
    compute_encoder.set_input_array(block_count, 5);
    compute_encoder.set_output_array(out, 6);
    compute_encoder.set_bytes(max_blocks, 7);
    compute_encoder.set_bytes(M, 8);
    compute_encoder.set_bytes(N, 9);
    compute_encoder.set_bytes(K, 10);

    MTL::Size grid_dims((N + cfg.bn - 1) / cfg.bn, max_blocks, 1);
    MTL::Size group_dims(cfg.wm * cfg.wn * 32, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(TurboQuantGatherBlocksPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const TurboQuantGatherBlocksPrimitive&>(other);
    return variant_ == rhs.variant_ && bits_ == rhs.bits_;
  }
  auto state() const { return std::make_tuple(variant_, bits_); }

 private:
  int variant_;
  int bits_;
};

}  // namespace

array turboquant_gather_blocks(
    const array& x,
    const array& weight,
    const array& norms,
    const array& codebook,
    const array& block_meta,
    const array& block_count,
    int variant,
    int bits,
    StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    throw std::invalid_argument("turboquant_gather_blocks: x must be fp16/bf16.");
  }
  if (weight.dtype() != uint32 || norms.dtype() != float16 ||
      codebook.dtype() != float32) {
    throw std::invalid_argument(
        "turboquant_gather_blocks: expected uint32 weight, fp16 norms, fp32 codebook.");
  }
  if (block_meta.dtype() != int32 || block_count.dtype() != int32) {
    throw std::invalid_argument(
        "turboquant_gather_blocks: block_meta/block_count must be int32.");
  }
  if (x.ndim() != 2 || weight.ndim() != 3 || norms.ndim() != 2 ||
      block_meta.ndim() != 2 || block_meta.shape(1) != 3 ||
      block_count.size() != 1) {
    throw std::invalid_argument("turboquant_gather_blocks: bad input ranks.");
  }
  if (!tq_row_contiguous(x) || !tq_row_contiguous(weight) ||
      !tq_row_contiguous(norms) || !tq_row_contiguous(block_meta) ||
      !tq_row_contiguous(block_count)) {
    throw std::invalid_argument(
        "turboquant_gather_blocks: inputs must be row contiguous.");
  }
  if (bits != 2 && bits != 4) {
    throw std::invalid_argument("turboquant_gather_blocks: bits must be 2 or 4.");
  }

  const int M = x.shape(0);
  const int N = weight.shape(1);
  Shape out_shape = {M, N};
  return array(
      std::move(out_shape), x.dtype(),
      std::make_shared<TurboQuantGatherBlocksPrimitive>(s, variant, bits),
      {x, weight, norms, codebook, block_meta, block_count});
}

}  // namespace omlx::glm_kernels
