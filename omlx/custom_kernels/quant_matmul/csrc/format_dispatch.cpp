// SPDX-License-Identifier: Apache-2.0
#include "format_dispatch.h"

#include <Python.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "mlx/array.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace omlx::quant_matmul {
namespace {

thread_local bool row_exact_armed = false;

class FormatLinearFast {
 public:
  FormatLinearFast(nb::object weight, nb::object bias,
                   std::vector<nb::object> operands, nb::dict proof_state)
      : weight_(std::move(weight)), bias_(std::move(bias)),
        operands_(std::move(operands)), proof_state_(std::move(proof_state)) {}

  void register_signature(nb::tuple key) {
    if (operands_.size() >= kArgCount) {
      return;
    }
    auto shape = nb::cast<std::vector<int>>(key[0]);
    auto dtype = nb::cast<mlx::core::Dtype>(key[1]);
    for (const auto& entry : entries_) {
      if (entry->shape == shape && entry->dtype == dtype) {
        return;
      }
    }
    if (entries_.size() == 64) {
      entries_.clear();
    }
    entries_.push_back(std::make_shared<Entry>(Entry{std::move(shape), dtype, std::move(key)}));
  }

  // Python keeps this bound METH_FASTCALL in the layer. Its self owns
  // the dispatcher, while the dispatcher owns no reference to the layer.
  nb::object bind() {
    static PyMethodDef method = {
        "replay", reinterpret_cast<PyCFunction>(replay), METH_FASTCALL, nullptr};
    auto self = nb::find(this);
    auto* bound = PyCFunction_NewEx(&method, self.ptr(), nullptr);
    if (!bound) {
      throw nb::python_error();
    }
    return nb::steal(bound);
  }

 private:
  static PyObject* replay(PyObject* self, PyObject* const* args,
                          Py_ssize_t count) {
    if (count != 2 || !PyDict_Check(args[0])) {
      PyErr_SetString(PyExc_TypeError, "expected a format layer and its input");
      return nullptr;
    }
    return nb::inst_ptr<FormatLinearFast>(self)->call(args[0], args[1]);
  }

  PyObject* call(PyObject* layer, PyObject* input) {
    if (row_exact_armed || Py_TYPE(input) != array_type_ ||
        !nb::inst_ready(input) ||
        PyDict_GetItem(layer, weight_key_.ptr()) != weight_.ptr()) {
      Py_RETURN_NONE;
    }
    auto* bias = PyDict_GetItem(layer, bias_key_.ptr());
    if (bias != (bias_.is_none() ? nullptr : bias_.ptr())) {
      Py_RETURN_NONE;
    }
    const auto* x = nb::inst_ptr<mlx::core::array>(input);
    const auto thread = PyThreadState_GetID(PyThreadState_Get());
    auto* kept = PyDict_GetItem(proof_state_.ptr(), kept_key_.ptr());
    if (!kept) {
      Py_RETURN_NONE;
    }
    if (last_ && last_kept_ == kept && last_thread_ == thread && last_->matches(*x)) {
      // Only the worker's live proof owns this record. Pin it across the
      // vectorcall, which can release the GIL and retire the proof.
      PyObject* owner = nullptr;
#if PY_VERSION_HEX >= 0x030D0000
      int live = PyWeakref_GetRef(last_owner_.ptr(), &owner);
      if (live < 0) {
        return nullptr;
      }
#else
      auto* target = PyWeakref_GetObject(last_owner_.ptr());
      if (target && target != Py_None) {
        owner = Py_NewRef(target);
      }
#endif
      if (owner) {
        nb::object pinned = nb::steal(owner);
        return launch(last_graph_, input, bias);
      }
    }
    for (const auto& entry : entries_) {
      if (entry->matches(*x)) {
        return resolve(entry, kept, thread, input, bias);
      }
    }
    Py_RETURN_NONE;
  }

  struct Entry {
    std::vector<int> shape;
    mlx::core::Dtype dtype;
    nb::tuple key;

    bool matches(const mlx::core::array& x) const {
      const auto& dims = x.shape();
      return dtype == x.dtype() &&
             std::equal(dims.begin(), dims.end(), shape.begin(), shape.end());
    }
  };

  PyObject* resolve(std::shared_ptr<Entry> entry, PyObject* kept, uint64_t thread,
                    PyObject* input, PyObject* bias) {
    auto* record = PyObject_CallMethodObjArgs(kept, replay_key_.ptr(), entry->key.ptr(), nullptr);
    if (!record || record == Py_None) {
      return record;
    }
    nb::object owner = nb::steal(record);
    auto* found = PyObject_GetAttr(record, graph_key_.ptr());
    if (!found) {
      return nullptr;
    }
    nb::object graph = nb::steal(found);
    auto* weak = PyWeakref_NewRef(record, nullptr);
    if (!weak) {
      return nullptr;
    }
    last_owner_ = nb::steal(weak);
    last_graph_ = found;
    last_kept_ = kept;
    last_thread_ = thread;
    last_ = std::move(entry);
    return launch(found, input, bias);
  }

  PyObject* launch(PyObject* graph, PyObject* input, PyObject* bias) const {
    std::array<PyObject*, kArgCount> args;
    args[0] = input;
    for (size_t i = 0; i < operands_.size(); ++i) {
      args[i + 1] = operands_[i].ptr();
    }
    auto* result = PyObject_Vectorcall(graph, args.data(), operands_.size() + 1, nullptr);
    if (!result || !bias) {
      return result;
    }
    auto* added = PyNumber_Add(result, bias_.ptr());
    Py_DECREF(result);
    return added;
  }

  static constexpr size_t kArgCount = 16;
  nb::object weight_, bias_;
  std::vector<nb::object> operands_;
  nb::dict proof_state_;
  PyTypeObject* array_type_ = reinterpret_cast<PyTypeObject*>(
      nb::type<mlx::core::array>().ptr());
  nb::str replay_key_{"replay"}, graph_key_{"graph"};
  nb::str weight_key_{"weight"}, bias_key_{"bias"}, kept_key_{"_kept"};
  std::vector<std::shared_ptr<Entry>> entries_;
  std::shared_ptr<Entry> last_;
  nb::object last_owner_;
  PyObject* last_graph_ = nullptr;  // borrowed while last_owner_'s target lives
  PyObject* last_kept_ = nullptr;   // identity only
  uint64_t last_thread_ = 0;
};

}  // namespace

void register_format_dispatch(nb::module_& m) {
  m.def("set_row_exact_armed", [](bool armed) { row_exact_armed = armed; });
  nb::class_<FormatLinearFast>(m, "FormatLinearFast")
      .def(nb::init<nb::object, nb::object, std::vector<nb::object>, nb::dict>(),
           "weight"_a, "bias"_a.none(), "operands"_a, "proof_state"_a)
      .def("register", &FormatLinearFast::register_signature)
      .def("bind", &FormatLinearFast::bind);
}

}  // namespace omlx::quant_matmul
