# SPDX-License-Identifier: Apache-2.0
"""The skeleton tuner's tables: the chip key, the loader, and the lookup."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.formats import ApplyRead, Dot, DotReads, Table, tuning
from omlx.formats.tuning import schema

CHIP = "applegpu_g17s_c1"


@pytest.fixture(autouse=True)
def fresh_tuner(monkeypatch, tmp_path):
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [tmp_path / "v1"])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    monkeypatch.setattr(tuning, "_SKELETON_DIGEST", {})
    tuning.register_skeleton("dense_tile", "skel-v1")
    tuning.reset_cache()
    yield
    tuning.reset_cache()


_FACE_SHELL = (
    "struct F {\n"
    "  __OMLX_READ_LOAD__ { return 0; }\n"
    "  __OMLX_READ_MADD__ { return 0; }\n"
    "};\n"
)


def _face(source=_FACE_SHELL, key="toy"):
    return Dot(key, source, "F", (), reads=DotReads())


def _entry(face=None, params=None, **key):
    face = face or _face()
    fields = {
        "route": "dense_tile",
        "face_id": face.key,
        "dtype": "bf16",
        "bits": 4,
        "group": 64,
        "n": 2560,
        "k": 5120,
        "rows_bucket": "1025-2048",
    }
    fields.update(key)
    if "n_bucket" in key:
        fields.pop("n", None)
    if "k_bucket" in key:
        fields.pop("k", None)
    return {
        "key": fields,
        "face_sha256": tuning.face_digest(face),
        "skeleton_sha256": tuning.skeleton_digest("dense_tile"),
        "measured_at": "2026-11-21T17:00:00Z",
        "params": params
        or {
            "backend": "nax",
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
        "exactness": "bitwise",
    }


def _write(path, chip, entries):
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{chip}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "chip": chip,
                "entries": entries,
            }
        )
    )


def _lookup(face, **over):
    fields = {
        "dtype": "bf16",
        "bits": 4,
        "group": 64,
        "n": 2560,
        "k": 5120,
        "rows": 2048,
        "chip": CHIP,
    }
    fields.update(over)
    return tuning.lookup("dense_tile", face, **fields)


def test_chip_id_names_the_architecture_and_core_count(monkeypatch):
    from omlx.utils import hardware

    monkeypatch.setattr(mx, "is_available", lambda device: True)
    monkeypatch.setattr(
        mx, "device_info", lambda device: {"architecture": "applegpu_g17s"}
    )
    monkeypatch.setattr(hardware, "get_gpu_core_count", lambda: 40)
    tuning.reset_cache()
    assert tuning.current_chip() == "applegpu_g17s_c40"


def test_lookup_falls_back_from_exact_to_bucketed_to_profile(tmp_path):
    face = _face()
    entries = [
        _entry(face),
        _entry(
            face,
            n_bucket="N1025-4096",
            k_bucket="K>4096",
            params={
                "backend": "nax",
                "bm": 128,
                "bn": 64,
                "bk": 128,
                "wm": 2,
                "wn": 2,
                "align_n": True,
            },
        ),
        _entry(
            face,
            face_id="*:lut_small",
            n_bucket="N1025-4096",
            k_bucket="K>4096",
            params={
                "backend": "nax",
                "bm": 32,
                "bn": 64,
                "bk": 64,
                "wm": 2,
                "wn": 2,
                "align_n": True,
            },
        ),
    ]
    _write(tmp_path / "v1", CHIP, entries)
    tuning.reset_cache()

    exact = _lookup(face)
    assert exact is not None and exact.bm == 64
    bucketed = _lookup(face, n=3840, k=6144)
    assert bucketed is not None and bucketed.bm == 128
    tabled = replace(_face(key="other"), tables=(Table("levels", mx.uint32, (1,)),))
    profile = _lookup(tabled, n=3840, k=6144)
    assert profile is not None and profile.bm == 32
    # A dtype and a row bucket the table does not know take parity (None).
    assert _lookup(face, dtype="fp16") is None
    assert _lookup(face, rows=17) is None


def test_an_unknown_chip_takes_the_parity_default(monkeypatch):
    monkeypatch.setattr(tuning, "current_chip", lambda: "applegpu_unknown_c1")
    assert _lookup(_face(), chip=None) is None


def test_an_illegal_entry_is_skipped_for_the_parity_default(tmp_path):
    face = _face()
    illegal = _entry(
        face,
        params={
            "backend": "nax",
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
            "pad_bytes": 65536,
        },
    )
    _write(tmp_path / "v1", CHIP, [illegal])
    tuning.reset_cache()
    assert _lookup(face) is None


def test_a_stale_skeleton_digest_is_ignored(tmp_path):
    face = _face()
    entry = _entry(face)
    entry["skeleton_sha256"] = "000000000000"
    _write(tmp_path / "v1", CHIP, [entry])
    tuning.reset_cache()
    assert _lookup(face) is None
    entry["skeleton_sha256"] = tuning.skeleton_digest("dense_tile")
    _write(tmp_path / "v1", CHIP, [entry])
    tuning.reset_cache()
    assert _lookup(face) is not None


def test_a_stale_face_digest_is_skipped(tmp_path):
    face = _face()
    entry = _entry(face)
    entry["face_sha256"] = "000000000000"
    _write(tmp_path / "v1", CHIP, [entry])
    tuning.reset_cache()
    assert _lookup(face) is None
    assert _lookup(_face(key="changed")) is None


def test_lookup_keeps_the_in_tree_entry_on_an_equal_score(tmp_path, monkeypatch):
    face = _face()
    in_tree = _entry(face, n_bucket="N1025-4096", k_bucket="K>4096")
    _write(tmp_path / "v1", CHIP, [in_tree])
    plugin = tmp_path / "plugin"
    plugin_entry = _entry(
        face,
        n_bucket="N1025-4096",
        k_bucket="K>4096",
        params={
            "backend": "nax",
            "bm": 128,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    _write(plugin, CHIP, [plugin_entry])
    calls = []
    real = schema.DenseTilePolicy.is_legal
    monkeypatch.setattr(
        schema.DenseTilePolicy,
        "is_legal",
        lambda self, *args, **kwargs: calls.append(self) or real(self, *args, **kwargs),
    )
    tuning.register_tuning_path(plugin / f"{CHIP}.json")
    tuning.reset_cache()
    got = _lookup(face)
    assert got is not None and got.bm == 64
    assert len(calls) == 1


def test_a_plugin_path_sits_below_the_in_tree_tables(tmp_path):
    face = _face()
    _write(
        tmp_path / "v1",
        CHIP,
        [
            _entry(
                face,
                params={
                    "backend": "nax",
                    "bm": 64,
                    "bn": 64,
                    "bk": 64,
                    "wm": 2,
                    "wn": 2,
                    "align_n": True,
                },
            )
        ],
    )
    plugin = tmp_path / "plugin"
    _write(
        plugin,
        CHIP,
        [
            _entry(
                face,
                params={
                    "backend": "nax",
                    "bm": 128,
                    "bn": 64,
                    "bk": 128,
                    "wm": 2,
                    "wn": 2,
                    "align_n": True,
                },
            )
        ],
    )
    tuning.register_tuning_path(plugin)
    tuning.reset_cache()
    got = _lookup(face)
    assert got is not None and got.bm == 64


def test_lookup_is_thread_safe_and_reads_a_table_once(tmp_path, monkeypatch):
    face = _face()
    _write(tmp_path / "v1", CHIP, [_entry(face)])
    reads = []
    real_read = type(tmp_path).read_text

    def counted(self, *args, **kwargs):
        reads.append(str(self))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "read_text", counted)
    tuning.reset_cache()

    def ask(_):
        return _lookup(face)

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(ask, range(16)))
    assert all(answer is not None and answer.bm == 64 for answer in answers)
    assert len(reads) <= 1


def test_every_in_tree_entry_is_current(tmp_path):
    # Re-register the live dense_tile skeleton (fresh_tuner swaps in a dummy
    # digest) and validate every shipped entry's skeleton_sha256, face_sha256,
    # and legality against all shipped affine tile faces.
    from omlx.formats import affine, tile_dense

    tuning.register_skeleton("dense_tile", tile_dense._SKELETON_TEXT)
    faces = {
        face.key: face
        for dt in (mx.bfloat16, mx.float16)
        for bits in (2, 3, 4, 5, 6, 8)
        for gs in (32, 64, 128)
        for face in (affine.tile_face(bits, gs, dt),)
    }
    chips: dict[str, list] = {}
    files = sorted(tuning._V1.glob("*.json"))
    assert {p.stem for p in files} >= {"applegpu_g15d_c80", "applegpu_g17s_c40"}
    for path in files:
        tuning._read_table(path, chips)
    assert len(chips["applegpu_g15d_c80"]) > 0
    assert len(chips["applegpu_g17s_c40"]) > 0
    for chip, entries in chips.items():
        for entry in entries:
            face = faces[entry.key["face_id"]]
            span = entry.rows_span
            sample_m = (
                span[1]
                if (span is not None and span[1] is not None)
                else max(4096, (span[0] if span and span[0] else 17))
            )
            assert entry.params.is_legal(
                face,
                entry.key["k"],
                entry.key["n"],
                sample_m,
                chip,
                dtype=entry.key["dtype"],
            ), f"illegal in-tree entry on {chip}: {entry}"
    assert tuning.stale_entries([tuning._V1], faces) == []


def test_stale_entries_lists_a_skeleton_and_a_face(tmp_path):
    face = _face()
    stale = _entry(face)
    stale["skeleton_sha256"] = "000000000000"
    other_face = _face(key="other")
    other = _entry(other_face)
    other["face_sha256"] = "111111111111"
    _write(tmp_path / "v1", CHIP, [stale, other])
    listed = tuning.stale_entries(
        [tmp_path / "v1"], {face.key: face, other_face.key: other_face}
    )
    assert any("skeleton_sha256" in item for item in listed)
    assert any("face_sha256" in item for item in listed)


def test_a_repin_needs_evidence_after_the_skeleton_change():
    with pytest.raises(ValueError, match="re-pin"):
        tuning.check_repin("2026-11-21T10:00:00Z", "2026-11-21T11:00:00Z")
    tuning.check_repin("2026-11-21T12:00:00Z", "2026-11-21T11:00:00Z")


def test_register_path_validates_the_schema(tmp_path):
    path = tmp_path / "bad"
    path.mkdir()
    (path / "chip.json").write_text("{not json")
    with pytest.raises(ValueError, match="tuning"):
        tuning.register_tuning_path(path)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "chip.json").write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError, match="schema_version"):
        tuning.register_tuning_path(empty)


def test_dense_tile_params_decode_and_reject_bad_geometry(tmp_path):
    params = schema.params_from_json(
        "dense_tile",
        {
            "backend": "nax",
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    assert isinstance(params, schema.DenseTilePolicy) and params.bm == 64
    with pytest.raises(ValueError, match="backend"):
        schema.params_from_json("dense_tile", {"backend": "quantum"})
    with pytest.raises(ValueError, match="bk"):
        schema.params_from_json(
            "dense_tile", {"backend": "steel", "bm": 64, "bn": 64, "bk": 48}
        )


def test_lookup_accepts_mlx_dtypes(tmp_path):
    face = _face()
    bf16 = _entry(face, dtype="bf16")
    fp16 = _entry(face, dtype="fp16")
    fp32 = _entry(
        face,
        dtype="fp32",
        params={
            "backend": "steel",
            "bm": 32,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    _write(tmp_path / "v1", CHIP, [bf16, fp16, fp32])
    tuning.reset_cache()
    assert _lookup(face, dtype=mx.bfloat16) is not None
    assert _lookup(face, dtype=mx.float16) is not None
    assert _lookup(face, dtype=mx.float32) is not None


def test_is_legal_enforces_nax_fragment_pairing():
    face = _face()
    # tile_matmad_nax has no zero-fill path for an odd TN above one, or an
    # odd TM behind a one-wide TN: those emit no MMA at all.
    for bm, bn in ((32, 32), (64, 96)):
        assert not schema.DenseTilePolicy("nax", bm, bn, 64, wm=2, wn=2).is_legal(
            face, 5120, 2496, 2048, "applegpu_g17s_c40"
        )
    for bm, bn in ((64, 32), (32, 64)):
        assert schema.DenseTilePolicy("nax", bm, bn, 64, wm=2, wn=2).is_legal(
            face, 5120, 2560, 2048, "applegpu_g17s_c40"
        )


def test_fp32_elements_reduce_the_tg_budget(tmp_path):
    face = _face()
    steel = _entry(
        face,
        dtype="fp32",
        params={
            "backend": "steel",
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    nax = _entry(
        face,
        dtype="fp32",
        params={
            "backend": "nax",
            "bm": 64,
            "bn": 64,
            "bk": 128,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    _write(tmp_path / "v1", CHIP, [steel, nax])
    tuning.reset_cache()
    assert _lookup(face, dtype=mx.float32) is None
    assert schema.DenseTilePolicy("steel", 32, 64, 64).is_legal(
        face, 5120, 2560, 2048, CHIP, dtype=mx.float32
    )


def test_stale_entries_scans_a_single_file(tmp_path):
    face = _face()
    plugin = tmp_path / "plugin"
    entry = _entry(face)
    entry["skeleton_sha256"] = "000000000000"
    _write(plugin, CHIP, [entry])
    stale = tuning.stale_entries([plugin / f"{CHIP}.json"], {face.key: face})
    assert any("skeleton_sha256" in item for item in stale)


def test_is_legal_rejects_non_positive_dims():
    face = _face()
    policy = schema.DenseTilePolicy("nax", 64, 64, 64)
    assert not policy.is_legal(face, 0, 2560, 2048, CHIP)
    assert not policy.is_legal(face, 5120, 0, 2048, CHIP)
    assert not policy.is_legal(face, 5120, 2560, 0, CHIP)


def test_the_tg_check_counts_tables():
    face = _face()
    tables = replace(face, tables=(Table("levels", mx.uint32, tuple(range(64))),))
    assert schema.DenseTilePolicy("steel", 64, 64, 64, pad_bytes=14336).is_legal(
        face, 5120, 2560, 2048, CHIP
    )
    assert not schema.DenseTilePolicy("steel", 64, 64, 64, pad_bytes=14336).is_legal(
        tables, 5120, 2560, 2048, CHIP
    )


def test_the_nax_gate_holds_its_generation_boundaries():
    face = _face()
    policy = schema.DenseTilePolicy("nax", 64, 64, 64)
    assert not policy.is_legal(face, 5120, 2560, 2048, "applegpu_g16d_c80")
    assert not policy.is_legal(face, 5120, 2560, 2048, "applegpu_g17p_c40")
    assert policy.is_legal(face, 5120, 2560, 2048, "applegpu_g17s_c40")
    assert policy.is_legal(face, 5120, 2560, 2048, "applegpu_g18p_c40")


def test_steel_warps_are_eight_rows():
    face = _face()
    # 48 rows is whole 8-row fragments at wm 2, not whole 16-row ones.
    assert schema.DenseTilePolicy("steel", 48, 64, 64, wm=2).is_legal(
        face, 5120, 2560, 2048, CHIP
    )


def test_is_legal_checks_tg_bytes_and_warp_tiles():
    face = _face()
    # 2*(128+128)*(128+8) B of staging exceeds the 32 KB a threadgroup has.
    assert not schema.DenseTilePolicy("steel", 128, 128, 128).is_legal(
        face, 5120, 2560, 2048, CHIP
    )
    # One weight buffer plus the pad does too.
    assert not schema.DenseTilePolicy("nax", 64, 64, 128, pad_bytes=20000).is_legal(
        face, 5120, 2560, 2048, CHIP
    )
    # NAX fragments are 16 rows: bm 16 with wm 2 has no whole warp tile.
    assert not schema.DenseTilePolicy("nax", 16, 64, 64, wm=2, wn=2).is_legal(
        face, 5120, 2560, 2048, CHIP
    )
    # Both staging buffers and no pad fit.
    assert schema.DenseTilePolicy("steel", 64, 64, 64).is_legal(
        face, 5120, 2560, 2048, CHIP
    )
    assert not schema.DenseTilePolicy("steel", 64, 64, 64, pad_bytes=15000).is_legal(
        face, 5120, 2560, 2048, CHIP
    )


def test_exact_entry_requires_face_sha256(tmp_path):
    face = _face()
    exact = _entry(face)
    exact["face_sha256"] = ""
    path = tmp_path / "plugin"
    _write(path, CHIP, [exact])
    with pytest.raises(ValueError, match="face_sha256"):
        tuning.register_tuning_path(path / f"{CHIP}.json")
    wildcard = _entry(face, face_id="*:plain")
    wildcard["face_sha256"] = ""
    _write(path, CHIP, [wildcard])
    tuning.register_tuning_path(path / f"{CHIP}.json")
    assert _lookup(face) is not None


def test_register_tuning_path_loads_single_file(tmp_path):
    face = _face()
    plugin = tmp_path / "plugin"
    _write(plugin, CHIP, [_entry(face)])
    tuning.register_tuning_path(plugin / f"{CHIP}.json")
    assert _lookup(face) is not None


def test_register_tuning_path_refuses_a_non_table(tmp_path):
    with pytest.raises(ValueError, match="not a table|holds no table"):
        tuning.register_tuning_path(tmp_path / "absent")
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema_version": 1, "chip": CHIP, "entries": []}))
    with pytest.raises(ValueError, match="chip"):
        tuning.register_tuning_path(wrong)


def test_rows_bucket_bounds():
    assert tuning.rows_bucket("dense_tile", 16) == "<17"
    assert tuning.rows_bucket("dense_tile", 17) == "17-24"
    assert tuning.rows_bucket("dense_tile", 2048) == "1025-2048"
    assert tuning.rows_bucket("dense_tile", 2049) == "2049+"
    assert tuning.rows_bucket("dense_qmv", 0) == "<1"
    assert tuning.rows_bucket("dense_qmv", 1) == "1"
    assert tuning.rows_bucket("dense_qmv", 24) == "17-24"
    assert tuning.rows_bucket("dense_qmv", 25) == "25+"
    assert tuning.rows_bucket("mma_tile", 0) == "<1"
    assert tuning.rows_bucket("mma_tile", 7) == "8"
    assert tuning.rows_bucket("mma_tile", 8) == "8"
    assert tuning.rows_bucket("mma_tile", 9) == "16"
    assert tuning.rows_bucket("mma_tile", 17) == "24"
    assert tuning.rows_bucket("mma_tile", 25) == "25+"


def test_current_chip_fails_closed(monkeypatch):
    from omlx.utils import hardware

    monkeypatch.setattr(mx, "is_available", lambda device: False)
    monkeypatch.setattr(hardware, "get_gpu_core_count", lambda: None)
    tuning.reset_cache()
    assert tuning.current_chip() == "unknown_c0"
    monkeypatch.setattr(mx, "is_available", lambda device: True)
    monkeypatch.setattr(
        mx, "device_info", lambda device: {"architecture": "applegpu_g15d"}
    )
    tuning.reset_cache()
    assert tuning.current_chip() == "applegpu_g15d_c0"
    monkeypatch.setattr(hardware, "get_gpu_core_count", lambda: 80)
    tuning.reset_cache()
    assert tuning.current_chip() == "applegpu_g15d_c80"


def test_nax_rejected_before_g17(tmp_path, monkeypatch):
    face = _face()
    entry = _entry(
        face,
        params={
            "backend": "nax",
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    _write(tmp_path / "v1", "applegpu_g15d_c80", [entry])
    monkeypatch.setattr(tuning, "current_chip", lambda: "applegpu_g15d_c80")
    tuning.reset_cache()
    assert _lookup(face, chip=None) is None


def test_face_digest_covers_type_and_vpt():
    first = replace(_face(), type="F1")
    second = replace(_face(), type="F2")
    assert tuning.face_digest(first) != tuning.face_digest(second)
    from omlx.formats import ApplyRead, Dot

    shell = _FACE_SHELL.replace("};", "  __OMLX_READ_APPLY__ { return 0; }\n};")
    narrow = Dot(
        "vpt2",
        shell,
        "F",
        (),
        reads=DotReads(apply=ApplyRead()),
        vpt=2,
    )
    wide = replace(narrow, vpt=4)
    assert tuning.face_digest(narrow) != tuning.face_digest(wide)


def test_lookup_falls_back_behind_a_stale_or_illegal_entry(tmp_path):
    face = _face()
    stale = _entry(face)
    stale["skeleton_sha256"] = "000000000000"
    bucketed = _entry(
        face,
        n_bucket="N1025-4096",
        k_bucket="K>4096",
        params={
            "backend": "nax",
            "bm": 128,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    _write(tmp_path / "v1", CHIP, [stale, bucketed])
    tuning.reset_cache()
    hit = _lookup(face)
    assert hit is not None and hit.bm == 128
    illegal = _entry(
        face,
        params={
            "backend": "nax",
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
            "pad_bytes": 65536,
        },
    )
    _write(tmp_path / "v1", CHIP, [illegal, bucketed])
    tuning.reset_cache()
    hit = _lookup(face)
    assert hit is not None and hit.bm == 128


def test_table_profile_splits_small_and_large_luts():
    assert tuning.table_profile(_face()) == "plain"
    small = replace(_face(), tables=(Table("levels", mx.uint32, tuple(range(64))),))
    assert tuning.table_profile(small) == "lut_small"
    large = replace(_face(), tables=(Table("levels", mx.uint32, tuple(range(65))),))
    assert tuning.table_profile(large) == "lut_tg"


def test_re_pin_refuses_an_equal_measurement_and_change():
    with pytest.raises(ValueError, match="re-pin"):
        tuning.check_repin("2026-11-21T17:00:00Z", "2026-11-21T17:00:00Z")


def test_n_bucket_holds_2048_in_the_middle_bucket(tmp_path):
    face = _face()
    params = {
        "backend": "nax",
        "bm": 64,
        "bn": 64,
        "bk": 64,
        "wm": 2,
        "wn": 2,
        "align_n": False,
    }
    _write(
        tmp_path / "v1",
        CHIP,
        [_entry(face, params=params, n_bucket="N1025-4096", k_bucket="K>4096")],
    )
    tuning.reset_cache()
    assert _lookup(face, n=2048, k=5120) is not None
    assert _lookup(face, n=2047, k=5120) is not None


def test_x_nreads_and_align_n_are_legal_checks():
    face = _face()
    assert not schema.DenseTilePolicy("nax", 64, 64, 64, x_nreads=999).is_legal(
        face, 5120, 2560, 2048, CHIP
    )
    assert not schema.DenseTilePolicy("nax", 64, 64, 64, align_n=True).is_legal(
        face, 5120, 2500, 2048, CHIP
    )
    assert schema.DenseTilePolicy("nax", 64, 64, 64, align_n=False).is_legal(
        face, 5120, 2500, 2048, CHIP
    )


def test_lookup_hashes_the_face_once(tmp_path, monkeypatch):
    face = _face()
    first = _entry(
        face,
        params={
            "backend": "nax",
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
            "pad_bytes": 65536,
        },
    )
    second = _entry(
        face,
        params={
            "backend": "nax",
            "bm": 128,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    _write(tmp_path / "v1", CHIP, [first, second])
    calls = []
    real = tuning.face_digest
    monkeypatch.setattr(
        tuning, "face_digest", lambda item: calls.append(item) or real(item)
    )
    tuning.reset_cache()
    assert _lookup(face) is not None
    assert len(calls) == 1


def test_the_nax_rule_mirrors_tile_matmad_nax():
    """is_legal's NAX parity must be the kernel's own dispatch predicate."""
    import re
    from pathlib import Path

    page = (
        Path(__file__).parents[1] / "omlx" / "custom_kernels" / "nax_tiles.py"
    ).read_text()
    start = page.index("METAL_FUNC void tile_matmad_nax(")
    body = page[start:]
    branches = re.findall(r"if constexpr \(([^)]+)\)", body)
    assert len(branches) == 2, branches
    conditions = [b.replace("&&", " and ").replace("||", " or ") for b in branches]
    assert "TN" in conditions[0] and "TM" in conditions[0]

    def legal(tm, tn):
        return any(eval(condition, {"TN": tn, "TM": tm}) for condition in conditions)

    face = _face()
    for tm in (1, 2, 3, 4):
        for tn in (1, 2, 3, 4):
            policy = schema.DenseTilePolicy(
                "nax", 16 * 2 * tm, 16 * 2 * tn, 64, wm=2, wn=2, align_n=False
            )
            assert policy.is_legal(face, 5120, 2560, 2048, "applegpu_g17s_c40") == (
                legal(tm, tn) and policy.wm * policy.wn * 32 >= 2 * policy.bn
            ), (tm, tn, conditions)


def test_the_element_size_comes_from_the_dtype():
    face = _face()
    policy = schema.DenseTilePolicy("steel", 64, 64, 64)
    assert policy.is_legal(face, 5120, 2560, 2048, CHIP, dtype=mx.float32) is False
    assert policy.is_legal(face, 5120, 2560, 2048, CHIP, dtype=mx.bfloat16) is True
    assert mx.float32.size == 4 and mx.bfloat16.size == 2


def test_steel_is_legal_enforces_block_loader_trows():
    face = _face()
    chip = "applegpu_g15d_c80"
    # TROWS = (wm*wn*32) / (bk / x_nreads) = 32; bm must be a whole multiple.
    assert not schema.DenseTilePolicy(
        "steel", 16, 64, 32, wm=2, wn=2, x_nreads=8
    ).is_legal(face, 5120, 2560, 2048, chip)
    assert not schema.DenseTilePolicy(
        "steel", 48, 64, 32, wm=2, wn=2, x_nreads=8
    ).is_legal(face, 5120, 2560, 2048, chip)
    assert not schema.DenseTilePolicy(
        "steel", 16, 64, 32, wm=2, wn=4, x_nreads=4
    ).is_legal(face, 5120, 2560, 2048, chip)
    assert schema.DenseTilePolicy("steel", 32, 64, 32, wm=2, wn=2, x_nreads=8).is_legal(
        face, 5120, 2560, 2048, chip
    )


def test_is_legal_checks_split_k_divides_k_tiles():
    face = _face()
    assert not schema.DenseTilePolicy("steel", 64, 64, 64, split_k=3).is_legal(
        face, 5120, 2560, 2048, CHIP
    )
    assert not schema.DenseTilePolicy("steel", 64, 64, 64, split_k=100).is_legal(
        face, 5120, 2560, 2048, CHIP
    )
    assert not schema.DenseTilePolicy("steel", 64, 64, 64, split_k=0).is_legal(
        face, 5120, 2560, 2048, CHIP
    )
    assert not schema.DenseTilePolicy("steel", 64, 64, 64, split_k=-1).is_legal(
        face, 5120, 2560, 2048, CHIP
    )
    assert not schema.DenseTilePolicy("steel", 64, 64, 64, split_k=1).is_legal(
        face, 5100, 2560, 2048, CHIP
    )
    assert schema.DenseTilePolicy("steel", 64, 64, 64, split_k=4).is_legal(
        face, 5120, 2560, 2048, CHIP
    )


def test_nax_gate_fails_closed_on_unknown_and_uppercase_chips():
    face = _face()
    nax = schema.DenseTilePolicy("nax", 64, 64, 64)
    assert not nax.is_legal(face, 5120, 2560, 2048, "unknown_c0")
    assert not nax.is_legal(face, 5120, 2560, 2048, "")
    assert not nax.is_legal(face, 5120, 2560, 2048, "applegpu_g17bogus")
    assert not nax.is_legal(face, 5120, 2560, 2048, "APPLEGPU_G15D_C80")
    assert not nax.is_legal(face, 5120, 2560, 2048, "APPLEGPU_G17P_C40")
    assert nax.is_legal(face, 5120, 2560, 2048, "APPLEGPU_G17S_C40")
    steel = schema.DenseTilePolicy("steel", 64, 64, 64)
    assert steel.is_legal(face, 5120, 2560, 2048, "applegpu_g15d_c80")
    assert steel.is_legal(face, 5120, 2560, 2048, "unknown_c0")


def test_lookup_uses_dtype_size_for_tg_budget(tmp_path):
    face = _face()
    assert not schema.DenseTilePolicy("steel", 64, 64, 64).is_legal(
        face, 5120, 2560, 2048, CHIP, dtype=mx.float64
    )
    # A raw dtype object whose .size is 4 under a bf16 key proves lookup
    # passes the object through to is_legal (the 2-byte fallback would fit).
    custom4 = SimpleNamespace(name="bf16", size=4)
    big_custom = _entry(
        face,
        dtype="bf16",
        params={
            "backend": "steel",
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    fit_custom = _entry(
        face,
        dtype="bf16",
        n_bucket="N1025-4096",
        k_bucket="K>4096",
        params={
            "backend": "steel",
            "bm": 32,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    _write(tmp_path / "v1", CHIP, [big_custom, fit_custom])
    tuning.reset_cache()
    got = _lookup(face, dtype=custom4)
    assert got is not None and got.bm == 32
    assert not schema.DenseTilePolicy("steel", 32, 64, 64).is_legal(
        face, 5120, 2560, 2048, CHIP, dtype="unknown_dtype"
    )


def test_public_api_in_all():
    for name in (
        "ChipTable",
        "check_repin",
        "current_chip",
        "face_digest",
        "lookup",
        "register_skeleton",
        "register_tuning_path",
        "reset_cache",
        "rows_bucket",
        "skeleton_digest",
        "stale_entries",
        "table_profile",
    ):
        assert name in tuning.__all__
        assert hasattr(tuning, name)


def test_lookup_skips_query_work_on_empty_table(tmp_path, monkeypatch):
    calls = []
    real = tuning.table_profile
    monkeypatch.setattr(
        tuning, "table_profile", lambda face: calls.append(face) or real(face)
    )
    assert _lookup(_face()) is None
    assert calls == []
    # Also skips query work when a non-empty table has no registered skeleton.
    face = _face()
    _write(tmp_path / "v1", CHIP, [_entry(face)])
    tuning.reset_cache()
    assert (
        tuning.lookup(
            "unregistered_route",
            face,
            dtype="bf16",
            bits=4,
            group=64,
            n=2560,
            k=5120,
            rows=2048,
            chip=CHIP,
        )
        is None
    )
    assert calls == []


def test_stale_entries_hashes_each_face_once(tmp_path, monkeypatch):
    face = _face()
    first = _entry(face)
    second = _entry(face, n=3840, k=6144)
    _write(tmp_path / "v1", CHIP, [first, second])
    calls = []
    real = tuning.face_digest
    monkeypatch.setattr(
        tuning, "face_digest", lambda item: calls.append(item) or real(item)
    )
    assert tuning.stale_entries([tmp_path / "v1"], {face.key: face}) == []
    assert len(calls) == 1


def test_read_table_rejects_bool_schema_version_and_empty_face_id(tmp_path):
    bool_ver = tmp_path / f"{CHIP}.json"
    bool_ver.write_text(
        json.dumps({"schema_version": True, "chip": CHIP, "entries": []})
    )
    with pytest.raises(ValueError, match="schema_version"):
        tuning.register_tuning_path(bool_ver)
    empty_id = _entry(_face(), face_id="")
    _write(tmp_path / "empty_id", CHIP, [empty_id])
    with pytest.raises(ValueError, match="face_id"):
        tuning.register_tuning_path(tmp_path / "empty_id" / f"{CHIP}.json")


def test_lookup_ranks_higher_score_regardless_of_entry_order(tmp_path):
    face = _face()
    wildcard_exact = _entry(
        face,
        face_id="*:plain",
        params={
            "backend": "nax",
            "bm": 32,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    bucketed = _entry(
        face,
        n_bucket="N1025-4096",
        k_bucket="K>4096",
        params={
            "backend": "nax",
            "bm": 128,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
        },
    )
    exact = _entry(face)
    # Exact (score 4) wins even when listed after lower-score matches.
    _write(tmp_path / "v1", CHIP, [wildcard_exact, bucketed, exact])
    tuning.reset_cache()
    got = _lookup(face)
    assert got is not None and got.bm == 64
    # Exact-face bucketed (score 3) beats wildcard exact-(n, k) (score 2).
    _write(tmp_path / "v1", CHIP, [wildcard_exact, bucketed])
    tuning.reset_cache()
    got = _lookup(face)
    assert got is not None and got.bm == 128


def test_fp32_stride_boundary_at_32kb():
    face = _face()
    # stride = 64 + 16//4 = 68; 4 * (32 + 64) * 68 = 26112; + 6656 == 32768.
    assert schema.DenseTilePolicy("steel", 32, 64, 64, pad_bytes=6656).is_legal(
        face, 5120, 2560, 2048, CHIP, dtype=mx.float32
    )
    assert not schema.DenseTilePolicy("steel", 32, 64, 64, pad_bytes=6657).is_legal(
        face, 5120, 2560, 2048, CHIP, dtype=mx.float32
    )


def test_stale_entries_defaults_to_table_dirs(tmp_path):
    face = _face()
    stale = _entry(face)
    stale["skeleton_sha256"] = "000000000000"
    _write(tmp_path / "v1", CHIP, [stale])
    listed = tuning.stale_entries(faces={face.key: face})
    assert any("skeleton_sha256" in item for item in listed)


def test_is_legal_threadgroup_warp_tile_and_negative_param_bounds():
    face = _face()
    # Threadgroup size: 8*4*32 == 1024 legal, 10*4*32 == 1280 illegal.
    assert schema.DenseTilePolicy(
        "nax", 128, 128, 64, wm=8, wn=4, align_n=False
    ).is_legal(face, 5120, 2560, 2048, CHIP)
    assert not schema.DenseTilePolicy(
        "nax", 160, 128, 64, wm=10, wn=4, align_n=False
    ).is_legal(face, 5120, 2560, 2048, CHIP)
    # BM/BN multiples of 16.
    assert not schema.DenseTilePolicy(
        "steel", 24, 64, 64, wm=1, wn=2, x_nreads=4
    ).is_legal(face, 5120, 2560, 2048, CHIP)
    assert not schema.DenseTilePolicy(
        "steel", 32, 24, 64, wm=2, wn=1, align_n=False
    ).is_legal(face, 5120, 2560, 2048, CHIP)
    # BN whole warp fragments (16 // 32 == 0 would pass tn % 2 == 0 without it).
    assert not schema.DenseTilePolicy(
        "nax", 64, 16, 64, wm=2, wn=2, align_n=False
    ).is_legal(face, 5120, 2560, 2048, CHIP)
    assert not schema.DenseTilePolicy(
        "steel", 64, 16, 64, wm=2, wn=4, align_n=False
    ).is_legal(face, 5120, 2560, 2048, CHIP)
    # Negative pad_bytes / gx.
    assert not schema.DenseTilePolicy("nax", 64, 64, 64, pad_bytes=-1).is_legal(
        face, 5120, 2560, 2048, CHIP
    )
    assert not schema.DenseTilePolicy("nax", 64, 64, 64, gx=-1).is_legal(
        face, 5120, 2560, 2048, CHIP
    )


def test_params_from_json_checks_unknown_keys_bools_and_param_staging():
    with pytest.raises(ValueError, match="unknown"):
        schema.params_from_json(
            "dense_tile",
            {"backend": "nax", "bm": 64, "bn": 64, "bk": 64, "bogus": 1},
        )
    with pytest.raises(ValueError, match="int"):
        schema.params_from_json(
            "dense_tile",
            {"backend": "nax", "bm": True, "bn": 64, "bk": 64},
        )
    with pytest.raises(ValueError, match="bool"):
        schema.params_from_json(
            "dense_tile",
            {"backend": "nax", "bm": 64, "bn": 64, "bk": 64, "align_n": "yes"},
        )
    with pytest.raises(ValueError, match="int"):
        schema.params_from_json(
            "dense_tile",
            {"backend": "nax", "bm": "64", "bn": 64, "bk": 64},
        )
    decoded = schema.params_from_json(
        "dense_tile",
        {
            "backend": "steel",
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "param_staging": "per_step",
        },
    )
    assert decoded.param_staging is schema.ParamStaging.PER_STEP
    with pytest.raises(ValueError, match="param_staging"):
        schema.params_from_json(
            "dense_tile",
            {
                "backend": "steel",
                "bm": 64,
                "bn": 64,
                "bk": 64,
                "param_staging": "bogus",
            },
        )


def test_shape_bucket_boundaries():
    assert tuning._n_bucket(1024) == "N<=1024"
    assert tuning._n_bucket(1025) == "N1025-4096"
    assert tuning._n_bucket(4096) == "N1025-4096"
    assert tuning._n_bucket(4097) == "N4097-32768"
    assert tuning._n_bucket(32768) == "N4097-32768"
    assert tuning._n_bucket(32769) == "N>32768"
    assert tuning._k_bucket(1024) == "K<=1024"
    assert tuning._k_bucket(1025) == "K1025-4096"
    assert tuning._k_bucket(4096) == "K1025-4096"
    assert tuning._k_bucket(4097) == "K>4096"


def test_wildcard_exact_shape_beats_wildcard_bucketed(tmp_path):
    face = _face()
    bucketed = _entry(
        face,
        face_id="*:plain",
        n_bucket="N1025-4096",
        k_bucket="K>4096",
        params={
            "backend": "nax",
            "bm": 32,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": False,
        },
    )
    exact = _entry(
        face,
        face_id="*:plain",
        params={
            "backend": "nax",
            "bm": 128,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": False,
        },
    )
    _write(tmp_path / "v1", CHIP, [bucketed, exact])
    tuning.reset_cache()
    got = _lookup(face)
    assert got is not None and got.bm == 128


def test_lookup_skips_is_legal_for_lower_score_after_a_best(tmp_path, monkeypatch):
    face = _face()
    bucketed = _entry(face, n_bucket="N1025-4096", k_bucket="K>4096")
    wildcard = _entry(
        face,
        face_id="*:plain",
        params={
            "backend": "nax",
            "bm": 128,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": False,
        },
    )
    _write(tmp_path / "v1", CHIP, [bucketed, wildcard])
    calls = []
    real = schema.DenseTilePolicy.is_legal
    monkeypatch.setattr(
        schema.DenseTilePolicy,
        "is_legal",
        lambda self, *args, **kwargs: calls.append(self) or real(self, *args, **kwargs),
    )
    tuning.reset_cache()
    got = _lookup(face)
    assert got is not None and got.bm == 64
    assert len(calls) == 1


def test_read_table_rejects_malformed_keys_and_float_version(tmp_path):
    face = _face()
    good = _entry(face)
    path = tmp_path / "v1"
    cases = []
    for version in (1.0, True, "1"):
        cases.append({"schema_version": version, "chip": CHIP, "entries": [good]})
    missing_rows = json.loads(json.dumps(good))
    missing_rows["key"].pop("rows_bucket")
    cases.append({"schema_version": 1, "chip": CHIP, "entries": [missing_rows]})
    for change in (
        {"row_bucket": "1025-2048"},
        {"extra": 1},
        {"dtype": "bfloat16"},
        {"dtype": "bf61"},
        {"dtype": "BF16"},
        {"face_id": "*:typo"},
        {"n": 2560},
        {"n_bucket": "N1025-4096"},
    ):
        entry = json.loads(json.dumps(good))
        entry["key"].update(change)
        if "row_bucket" in change:
            entry["key"].pop("rows_bucket")
        if "n" in change:
            entry["key"].pop("k")
        cases.append({"schema_version": 1, "chip": CHIP, "entries": [entry]})
    for case in cases:
        path.mkdir(exist_ok=True)
        (path / f"{CHIP}.json").write_text(json.dumps(case))
        with pytest.raises(ValueError):
            tuning.register_tuning_path(path)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"bits": 0}, "bits"),
        ({"bits": True}, "bits"),
        ({"group": True}, "group"),
        ({"n": 0}, "n"),
        ({"n": True}, "n"),
        ({"route": []}, "route"),
        ({"n_bucket": "N1025-4096"}, "k_bucket"),
    ],
)
def test_read_table_refuses_each_malformed_key_field(tmp_path, change, match):
    face = _face()
    entry = _entry(face)
    if "n_bucket" in change:
        entry["key"].pop("n")
        entry["key"].pop("k")
        entry["key"]["n_bucket"] = change["n_bucket"]
        if "k_bucket" in change:
            entry["key"]["k_bucket"] = change["k_bucket"]
    else:
        entry["key"].update(change)
    _write(tmp_path / "v1", CHIP, [entry])
    with pytest.raises(ValueError, match=match):
        tuning.register_tuning_path(tmp_path / "v1")


def test_read_table_rejects_unknown_bucket_labels(tmp_path):
    face = _face()
    for key_change in (
        # A custom lo-hi span is a legal label (PFX's row spans); a reversed
        # range is not, so an unknown-label typo still fails closed.
        {"rows_bucket": "33-32"},
        {"n_bucket": "N1024-4096", "k_bucket": "K>4096"},
        {"n_bucket": "N1025-4096", "k_bucket": "K>=4096"},
    ):
        entry = _entry(face)
        if "n_bucket" in key_change:
            entry["key"].pop("n")
            entry["key"].pop("k")
        entry["key"].update(key_change)
        _write(tmp_path / "v1", CHIP, [entry])
        with pytest.raises(ValueError, match="bucket"):
            tuning.register_tuning_path(tmp_path / "v1")


def test_lookup_empty_chip_fails_closed(tmp_path, monkeypatch):
    face = _face()
    _write(tmp_path / "v1", CHIP, [_entry(face)])
    monkeypatch.setattr(tuning, "current_chip", lambda: CHIP)
    tuning.reset_cache()
    assert _lookup(face, chip="") is None


def test_stale_entries_rejects_a_missing_path(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        tuning.stale_entries([tmp_path / "missing"])


def test_exactness_values_are_validated(tmp_path):
    face = _face()
    entry = _entry(face)
    entry["exactness"] = "rule17_fp32"
    _write(tmp_path / "v1", CHIP, [entry])
    tuning.reset_cache()
    assert _lookup(face) is not None
    entry["exactness"] = "bogus"
    (tmp_path / "v1" / f"{CHIP}.json").write_text(
        json.dumps({"schema_version": 1, "chip": CHIP, "entries": [entry]})
    )
    with pytest.raises(ValueError):
        tuning.register_tuning_path(tmp_path / "v1")


def test_face_digest_covers_shared_tensors_and_tables():
    shell = _FACE_SHELL.replace("};", "  __OMLX_READ_APPLY__ { return 0; }\n};")
    face = Dot("toy_shared", shell, "F", (), reads=DotReads(apply=ApplyRead()))
    shared = replace(face, shared="// shared\n")
    assert tuning.face_digest(face) != tuning.face_digest(shared)
    other = replace(face, tensors=("group_scales",))
    assert tuning.face_digest(face) != tuning.face_digest(other)
    other = replace(face, tables=(Table("levels", mx.uint32, (1, 2)),))
    assert tuning.face_digest(face) != tuning.face_digest(other)


def test_unknown_route_names_and_rows_bucket_are_honest():
    assert "dense_tile" in schema.route_names()
    assert tuning.rows_bucket("no_such_route", 7) == "7"


def test_current_chip_without_an_architecture_is_unknown(monkeypatch):
    from omlx.utils import hardware

    monkeypatch.setattr(mx, "is_available", lambda device: True)
    monkeypatch.setattr(mx, "device_info", lambda device: {})
    monkeypatch.setattr(hardware, "get_gpu_core_count", lambda: 80)
    tuning.reset_cache()
    assert tuning.current_chip() == "unknown_c80"


def test_check_repin_reads_naive_timestamps_as_utc():
    with pytest.raises(ValueError, match="re-pin"):
        tuning.check_repin("2026-11-21T17:00:00", "2026-11-21T17:00:00Z")


def test_direct_is_legal_refuses_a_bad_param_staging():
    face = _face()
    assert not schema.DenseTilePolicy(
        "nax", 64, 64, 64, param_staging="bogus"
    ).is_legal(face, 5120, 2560, 2048, CHIP)


def _stub_face():
    from types import SimpleNamespace

    return SimpleNamespace(tables=())


def test_mma_tile_is_legal_rejects_unsupported_bn_and_bk():
    from omlx.formats.tuning.schema import MmaTilePolicy

    face = _stub_face()
    assert not MmaTilePolicy(64, 32, 4).is_legal(face, 2560, 128, 8, "applegpu_g15d")
    assert not MmaTilePolicy(16, 48, 4).is_legal(face, 2880, 64, 8, "applegpu_g15d")
    for bn in (8, 16, 32):
        assert MmaTilePolicy(bn, 32, 4).is_legal(face, 2560, 64, 8, "applegpu_g15d")
    for bk in (32, 64, 128):
        assert MmaTilePolicy(16, bk, 4).is_legal(face, 2560, 64, 8, "applegpu_g15d")


def test_mma_tile_is_legal_checks_tg_limits_and_dtype():
    from omlx.formats.tuning.schema import MmaTilePolicy

    face = _stub_face()
    assert not MmaTilePolicy(16, 32, 64).is_legal(face, 2048, 64, 8, "applegpu_g15d")
    assert not MmaTilePolicy(16, 32, 16).is_legal(face, 2560, 4096, 24, "applegpu_g15d")
    assert not MmaTilePolicy(16, 32, 16).is_legal(
        face, 2560, 64, 8, "applegpu_g15d", dtype="fp32"
    )
    assert not MmaTilePolicy(16, 32, 4).is_legal(
        face, 2560, 64, 8, "applegpu_g15d", dtype="unknown"
    )
    assert MmaTilePolicy(16, 32, 16).is_legal(face, 2560, 64, 8, "applegpu_g15d")
    assert not MmaTilePolicy(8, 32, 36).is_legal(face, 1152, 64, 8, "applegpu_g15d")
    for m, legal in (
        (1, True),
        (7, True),
        (8, True),
        (9, False),
        (16, False),
        (17, False),
    ):
        assert (
            MmaTilePolicy(16, 64, 12).is_legal(face, 768, 64, m, "applegpu_g15d")
            is legal
        ), m
    assert "MmaTilePolicy" in tuning.__all__


def test_params_from_json_rejects_bad_types_for_mma_tile():
    import pytest

    from omlx.formats.tuning.schema import params_from_json

    for bad in ({"bn": "16"}, {"bn": True}, {"nsg": False}):
        with pytest.raises(ValueError):
            params_from_json("mma_tile", bad)


def test_mma_asserts_mirror_is_legal():
    import re

    from omlx.formats import verify_skeleton
    from omlx.formats.tuning.schema import MmaTilePolicy, mma_asserts

    class Face:
        type = "F"
        key = "f"
        source = ""
        tensors = ()
        tables = ()

    face = Face()
    source = verify_skeleton.emit_mma_source(
        verify_skeleton.tile_mma_ops(face, policy=MmaTilePolicy(32, 32, 4)),
        1,
        2560,
    )
    expressions = re.findall(r'static_assert\((.*?), "', source)
    assert expressions == [expression for expression, _ in mma_asserts()]
    for dtype, elem in (("bf16", 2), ("fp32", 4)):
        for bn in (0, 8, 16, 32, 64):
            for bk in (16, 32, 64, 128):
                for nsg in (0, 1, 4, 8, 32, 64):
                    for k, n, m in (
                        (2560, 64, 8),
                        (2048, 4096, 24),
                        (1024, 8, 8),
                        (256, 24, 16),
                    ):
                        bm = ((m + 7) // 8) * 8
                        symbols = {
                            "BN": bn,
                            "BK": bk,
                            "NSG": nsg,
                            "K": k,
                            "N": n,
                            "BM": bm,
                        }
                        truth = all(
                            [
                                eval(
                                    expression.replace("sizeof(T)", str(elem))
                                    .replace("||", " or ")
                                    .replace("&&", " and "),
                                    {"__builtins__": {}},
                                    symbols,
                                )
                                for expression in expressions
                            ]
                        )
                        expected = truth and bn > 0 and n % bn == 0
                        got = MmaTilePolicy(bn, bk, nsg).is_legal(
                            face, k, n, m, "applegpu_g15d", dtype=dtype
                        )
                        assert got == expected, (bn, bk, nsg, k, n, m, dtype)
    # Non-positive shape dimensions are refused.
    for k, n, m in ((0, 64, 8), (2560, 0, 8), (2560, 64, 0)):
        assert not MmaTilePolicy(16, 32, 8).is_legal(
            face, k, n, m, "applegpu_g15d", dtype="bf16"
        )
    # Exact 32,768-byte threadgroup boundary including face table_bytes.
    # 16 * (32 * 16 * 2 + 16 * 16 * 4) == 32768; a 4-byte Table pushes it over.
    assert MmaTilePolicy(16, 32, 16).is_legal(
        face, 2560, 64, 16, "applegpu_g15d", dtype="bf16"
    )
    tabled = SimpleNamespace(tables=(Table("levels", mx.uint32, (1,)),))
    assert not MmaTilePolicy(16, 32, 16).is_legal(
        tabled, 2560, 64, 16, "applegpu_g15d", dtype="bf16"
    )


def test_register_skeleton_refuses_a_route_without_row_buckets():
    import pytest

    from omlx.formats import tuning

    with pytest.raises(ValueError, match="no row buckets"):
        tuning.register_skeleton("unbucketed", "skeleton")
    tuning.register_skeleton("mma_tile", "skeleton")


def test_dense_tile_route_only_backends_and_nax_sched_decode_and_check_legality():
    face = _face()
    for backend in ("mlx", "wide", "rows"):
        pol = schema.params_from_json("dense_tile", {"backend": backend})
        assert pol == schema.DenseTilePolicy(backend)
        assert pol.is_legal(face, 2560, 2560, 48, "applegpu_g15d_c80")
        assert not schema.DenseTilePolicy(backend, bm=32).is_legal(
            face, 2560, 2560, 48, "applegpu_g15d_c80"
        )
        with pytest.raises(ValueError, match="bm|bn|bk"):
            schema.params_from_json("dense_tile", {"backend": backend, "bm": 32})
    db = schema.params_from_json(
        "dense_tile",
        {
            "backend": "nax",
            "bm": 32,
            "bn": 64,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
            "sched": "db",
        },
    )
    assert db.sched == "db"
    assert db.is_legal(face, 2560, 2560, 24, "applegpu_g17s_c40")
    # Double-buffered NAX requires aligned N, aligned K, even TN, and 2x Ws TG budget.
    assert not schema.DenseTilePolicy(
        "nax", 32, 64, 64, 2, 2, align_n=False, sched="db"
    ).is_legal(face, 2560, 2560, 24, "applegpu_g17s_c40")
    assert not schema.DenseTilePolicy(
        "nax", 64, 64, 128, 2, 2, align_n=True, sched="db"
    ).is_legal(face, 2560, 2560, 64, "applegpu_g17s_c40")
    assert not schema.DenseTilePolicy(
        "steel", 32, 64, 64, 2, 2, align_n=True, sched="db"
    ).is_legal(face, 2560, 2560, 64, "applegpu_g15d_c80")
    with pytest.raises(ValueError, match="sched"):
        schema.params_from_json(
            "dense_tile",
            {"backend": "nax", "bm": 64, "bn": 64, "bk": 64, "sched": "bogus"},
        )
    # Steel WeightLoader requires (BN*BK)/TGP >= 8 and dividing BK.
    assert not schema.DenseTilePolicy(
        "steel", 64, 16, 32, wm=4, wn=1, x_nreads=8
    ).is_legal(face, 2560, 2560, 64, "applegpu_g15d_c80")


def test_narrow_rows_span_beats_enclosing_bucket_and_exposes_row_bounds(tmp_path):
    face = _face()
    chip = "applegpu_g15d_c80"
    broad_lo = _entry(
        face,
        n=2560,
        k=2560,
        rows_bucket="33-48",
        params={
            "backend": "steel",
            "bm": 16,
            "bn": 16,
            "bk": 128,
            "wm": 2,
            "wn": 2,
            "align_n": True,
            "x_nreads": 8,
            "param_staging": "per_step",
            "vec_store": True,
        },
    )
    broad_hi = _entry(
        face,
        n=2560,
        k=2560,
        rows_bucket="49-64",
        params={
            "backend": "steel",
            "bm": 32,
            "bn": 16,
            "bk": 128,
            "wm": 2,
            "wn": 2,
            "align_n": True,
            "x_nreads": 8,
            "param_staging": "per_step",
        },
    )
    narrow = _entry(
        face,
        n=2560,
        k=2560,
        rows_bucket="44-52",
        params={"backend": "rows"},
    )
    # Write broad entries BEFORE narrow to confirm span specificity wins over order.
    _write(tmp_path / "v1", chip, [broad_lo, broad_hi, narrow])
    tuning.reset_cache()

    assert "row_bounds" in tuning.__all__
    bounds = tuning.row_bounds("dense_tile", chip=chip)
    assert 43 in bounds and 52 in bounds

    p43 = _lookup(face, n=2560, k=2560, rows=43, chip=chip)
    assert p43 is not None and (p43.backend, p43.bm, p43.bk) == ("steel", 16, 128)
    for r in range(44, 53):
        pr = _lookup(face, n=2560, k=2560, rows=r, chip=chip)
        assert pr is not None and pr.backend == "rows", r
    p53 = _lookup(face, n=2560, k=2560, rows=53, chip=chip)
    assert p53 is not None and (p53.backend, p53.bm, p53.bk) == ("steel", 32, 128)


def test_lookup_empty_chip_fails_closed_twice_contract(tmp_path, monkeypatch):
    face = _face()
    _write(tmp_path / "v1", CHIP, [_entry(face)])
    monkeypatch.setattr(tuning, "current_chip", lambda: CHIP)
    tuning.reset_cache()
    assert _lookup(face, chip=None) is not None
    assert _lookup(face, chip="") is None


def test_nax_is_legal_rejects_klt_exceeding_tgp():
    # kTGP = wm * wn * 32 < BN leaves rows of Ws uninitialized even at kTPR = 1.
    assert (
        schema.DenseTilePolicy("nax", 64, 128, 64, wm=1, wn=2).is_legal(
            _face(), 2560, 2560, 64, "applegpu_g17s_c40"
        )
        is False
    )
    assert (
        schema.DenseTilePolicy("nax", 32, 64, 64, wm=1, wn=1).is_legal(
            _face(), 2560, 2560, 32, "applegpu_g17s_c40"
        )
        is False
    )
    # kTGP == BN and BN < kTGP < 2 * BN both select kLT == BN (kTPR == 1) and
    # are refused until the kLT == BN mapping is fixed (DEFERRED).
    assert (
        schema.DenseTilePolicy("nax", 64, 128, 64, wm=2, wn=2).is_legal(
            _face(), 2560, 2560, 64, "applegpu_g17s_c40"
        )
        is False
    )
    assert (
        schema.DenseTilePolicy("nax", 32, 32, 64, wm=1, wn=1).is_legal(
            _face(), 2560, 2560, 32, "applegpu_g17s_c40"
        )
        is False
    )
    assert (
        schema.DenseTilePolicy("nax", 48, 64, 64, wm=3, wn=1).is_legal(
            _face(), 512, 64, 48, "applegpu_g17s_c40"
        )
        is False
    )
    assert (
        schema.DenseTilePolicy("nax", 64, 192, 32, wm=4, wn=2).is_legal(
            _face(), 512, 192, 64, "applegpu_g17s_c40"
        )
        is False
    )
    # Boundary: kTGP == 2 * BN selects kLT == 2 * BN (kTPR == 2) and stays legal.
    assert (
        schema.DenseTilePolicy("nax", 64, 64, 64, wm=2, wn=2).is_legal(
            _face(), 512, 64, 64, "applegpu_g17s_c40"
        )
        is True
    )


def test_single_row_span_is_accepted_and_reversed_is_refused(tmp_path):
    face = _face()
    entry = _entry(face, n=2560, k=2560, rows_bucket="17-17", params={"backend": "mlx"})
    _write(tmp_path / "v1", CHIP, [entry])
    tuning.reset_cache()
    try:
        table = tuning.lookup(
            "dense_tile",
            face,
            dtype=mx.bfloat16,
            bits=4,
            group=64,
            n=2560,
            k=2560,
            rows=17,
            chip=CHIP,
        )
        assert table is not None and table.backend == "mlx"
        for out_of_span in (16, 18):
            assert (
                tuning.lookup(
                    "dense_tile",
                    face,
                    dtype=mx.bfloat16,
                    bits=4,
                    group=64,
                    n=2560,
                    k=2560,
                    rows=out_of_span,
                    chip=CHIP,
                )
                is None
            )
        first = _entry(
            face, n=2560, k=2560, rows_bucket="1-1", params={"backend": "mlx"}
        )
        _write(tmp_path / "v1", CHIP, [first])
        tuning.reset_cache()
        assert (
            tuning.lookup(
                "dense_tile",
                face,
                dtype=mx.bfloat16,
                bits=4,
                group=64,
                n=2560,
                k=2560,
                rows=1,
                chip=CHIP,
            )
            is not None
        )
        bad = _entry(
            face, n=2560, k=2560, rows_bucket="18-17", params={"backend": "mlx"}
        )
        with pytest.raises(ValueError):
            _write(tmp_path / "v1", CHIP, [bad])
            tuning.reset_cache()
            tuning.lookup(
                "dense_tile",
                face,
                dtype=mx.bfloat16,
                bits=4,
                group=64,
                n=2560,
                k=2560,
                rows=17,
                chip=CHIP,
            )
    finally:
        tuning.reset_cache()


def test_bare_integer_rows_bucket_outside_route_is_refused(tmp_path):
    face = _face()
    bad_cases = (
        _entry(face, n=2560, k=2560, rows_bucket="17", params={"backend": "mlx"}),
        _entry(
            face,
            route="mma_tile",
            n=2560,
            k=2560,
            rows_bucket="32",
            params={"bn": 16, "bk": 32, "nsg": 8},
        ),
        _entry(face, n=2560, k=2560, rows_bucket="0-0", params={"backend": "mlx"}),
        _entry(face, n=2560, k=2560, rows_bucket="17-17\n", params={"backend": "mlx"}),
        _entry(face, n=2560, k=2560, rows_bucket="017-24", params={"backend": "mlx"}),
        _entry(
            face,
            n=2560,
            k=2560,
            rows_bucket="\u0661\u0667-\u0661\u0667",
            params={"backend": "mlx"},
        ),
        _entry(face, n=2560, k=2560, rows_bucket="17-024", params={"backend": "mlx"}),
        _entry(
            face, n=2560, k=2560, rows_bucket="1\u0667-24", params={"backend": "mlx"}
        ),
        _entry(
            face,
            n=2560,
            k=2560,
            rows_bucket="17-\u0662\u0664",
            params={"backend": "mlx"},
        ),
    )
    try:
        for bad in bad_cases:
            _write(tmp_path / "v1", CHIP, [bad])
            tuning.reset_cache()
            with pytest.raises(ValueError, match="rows_bucket"):
                tuning.lookup(
                    bad["key"]["route"],
                    face,
                    dtype=mx.bfloat16,
                    bits=4,
                    group=64,
                    n=2560,
                    k=2560,
                    rows=17,
                    chip=CHIP,
                )
    finally:
        tuning.reset_cache()


def test_row_bounds_preserves_custom_spans_below_17(tmp_path):
    face = _face()
    entry = _entry(face, n=2560, k=2560, rows_bucket="12-14", params={"backend": "mlx"})
    _write(tmp_path / "v1", CHIP, [entry])
    tuning.reset_cache()
    bounds = tuning.row_bounds("dense_tile", chip=CHIP)
    assert 11 in bounds and 14 in bounds


def test_face_digest_caches_per_face(monkeypatch):
    import hashlib

    calls = 0
    real_sha = hashlib.sha256

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_sha(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", counted)
    face = _face(key="cache_test_face_digest")
    first = tuning.face_digest(face)
    second = tuning.face_digest(face)
    assert first == second and calls == 1
