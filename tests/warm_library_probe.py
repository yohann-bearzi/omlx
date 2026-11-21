# SPDX-License-Identifier: Apache-2.0
"""Count actual Metal library builds for one face kernel in this subprocess."""

import ctypes
import ctypes.util
import gc
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from omlx.formats import kernels


def build_counter(name=b"custom_kernel_u35a_copy"):
    objc = ctypes.CDLL(ctypes.util.find_library("objc"))
    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    ptr = ctypes.c_void_p
    metal.MTLCreateSystemDefaultDevice.restype = ptr
    objc.object_getClass.argtypes, objc.object_getClass.restype = [ptr], ptr
    objc.sel_registerName.argtypes, objc.sel_registerName.restype = [
        ctypes.c_char_p
    ], ptr
    objc.class_getInstanceMethod.argtypes, objc.class_getInstanceMethod.restype = [
        ptr,
        ptr,
    ], ptr
    objc.method_getImplementation.argtypes, objc.method_getImplementation.restype = [
        ptr
    ], ptr
    objc.method_setImplementation.argtypes, objc.method_setImplementation.restype = [
        ptr,
        ptr,
    ], ptr
    device = metal.MTLCreateSystemDefaultDevice()
    cls = objc.object_getClass(device)
    selector = objc.sel_registerName(b"newLibraryWithSource:options:error:")
    method = objc.class_getInstanceMethod(cls, selector)
    assert method
    signature = ctypes.CFUNCTYPE(ptr, ptr, ptr, ptr, ptr, ptr)
    original = signature(objc.method_getImplementation(method))
    utf8 = objc.sel_registerName(b"UTF8String")
    as_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ptr, ptr)(("objc_msgSend", objc))
    counted = []

    @signature
    def count(receiver, cmd, source, options, error):
        text = as_string(source, utf8)
        result = original(receiver, cmd, source, options, error)
        if result and name in text:
            counted.append(hashlib.sha256(text).hexdigest())
        return result

    old = objc.method_setImplementation(method, ctypes.cast(count, ptr))

    # Hold the callback until its method is restored.
    def restore():
        objc.method_setImplementation(method, old)
        return count

    return counted, restore


def run():
    call = kernels.face_kernel(
        "u35a_copy",
        ["x"],
        {},
        ["y"],
        "",
        "uint i = thread_position_in_grid.x; y[i] = x[i] + 1;",
    )
    size = 4 * 1024 * 1024
    instance = call.instance([((size,), mx.float32)], [mx.float32])

    def stand_in():
        return kernels.StandIn(
            input_shapes=((8,),),
            output_shapes=((1,),),
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
        )

    counted, restore = build_counter()
    try:
        measure_peak = "--build-only" not in sys.argv
        if measure_peak:
            gc.collect()
            mx.reset_peak_memory()
            baseline = mx.get_active_memory()
        (future,) = kernels.warm({instance: stand_in})
        future.result(timeout=30)
        assert len(counted) == 1, counted
        if measure_peak:
            peak = mx.get_peak_memory() - baseline
            assert peak < 256 * 1024, peak
            print("warm peak above baseline", peak, "bytes", flush=True)

        x = mx.zeros((size,), mx.float32)
        (y,) = call(
            inputs=[x],
            output_shapes=[(size,)],
            output_dtypes=[mx.float32],
            grid=(size, 1, 1),
            threadgroup=(32, 1, 1),
        )
        mx.eval(y)
        assert call.instances() == (instance,)
        assert y.shape == (size,) and y[0].item() == y[-1].item() == 1
        assert len(counted) == 1, counted
        # A new specialization must be counted, so a broken hook cannot pass.
        (z,) = call(
            inputs=[x],
            output_shapes=[(1,)],
            output_dtypes=[mx.float16],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
        )
        mx.eval(z)
        assert len(counted) == 2, counted
        print(
            "library builds: warm 1, served launch 0, new specialization 1", flush=True
        )
    finally:
        restore()


if __name__ == "__main__":
    run()
