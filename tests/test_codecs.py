import numpy as np
import pytest

from minisearch.storage.codecs import (
    IntCodec,
    delta_decode,
    delta_decode_runs,
    delta_encode,
    delta_encode_runs,
    vbyte_decode,
    vbyte_decode_py,
    vbyte_encode,
    vbyte_encode_py,
)


def test_vbyte_known_encodings():
    assert vbyte_encode_py([0]) == b"\x00"
    assert vbyte_encode_py([5]) == b"\x05"
    assert vbyte_encode_py([127]) == b"\x7f"
    assert vbyte_encode_py([128]) == b"\x80\x01"
    assert vbyte_encode_py([300]) == b"\xac\x02"  # 300 = 0b10_0101100
    assert vbyte_encode_py([16384]) == b"\x80\x80\x01"


EDGE = [0, 1, 127, 128, 255, 16383, 16384, 2**21 - 1, 2**21, 2**28, 2**32 - 1, 2**35, 2**63 - 1]


@pytest.mark.parametrize("values", [[], [0], EDGE, EDGE * 3, list(range(1000))])
def test_vbyte_round_trip_and_implementations_agree(values):
    enc_py = vbyte_encode_py(values)
    enc_np = vbyte_encode(np.asarray(values, dtype=np.uint64))
    assert enc_py == enc_np
    assert vbyte_decode_py(enc_py) == values
    assert vbyte_decode(enc_py).tolist() == values


@pytest.mark.parametrize("seed", range(10))
def test_vbyte_random_round_trip(seed):
    rng = np.random.default_rng(seed)
    # Mix of magnitudes so values span 1..5 bytes.
    bits = rng.integers(0, 33, size=5000)
    values = rng.integers(0, 2**32, size=5000, dtype=np.uint64) >> (32 - bits).astype(np.uint64)
    enc = vbyte_encode(values)
    assert enc == vbyte_encode_py(values.tolist())
    assert np.array_equal(vbyte_decode(enc), values)


def test_vbyte_decode_count_limits_output():
    enc = vbyte_encode_py([1, 300, 5, 70000])
    assert vbyte_decode(enc, count=2).tolist() == [1, 300]
    assert vbyte_decode_py(enc, count=3) == [1, 300, 5]


def test_vbyte_rejects_negative():
    with pytest.raises(ValueError):
        vbyte_encode_py([-1])


def test_delta_round_trip():
    docs = np.array([3, 7, 8, 20, 1000, 1001])
    gaps = delta_encode(docs)
    assert gaps.tolist() == [3, 4, 1, 12, 980, 1]
    assert delta_decode(gaps).tolist() == docs.tolist()
    with pytest.raises(ValueError):
        delta_encode(np.array([5, 3]))


def test_delta_runs_restart_per_document():
    positions = np.array([3, 7, 9, 2, 5])
    tfs = np.array([3, 2])
    gaps = delta_encode_runs(positions, tfs)
    assert gaps.tolist() == [3, 4, 2, 2, 3]
    assert delta_decode_runs(gaps, tfs).tolist() == positions.tolist()


@pytest.mark.parametrize("seed", range(5))
def test_delta_runs_random(seed):
    rng = np.random.default_rng(seed)
    tfs = rng.integers(1, 10, size=300)
    positions = np.concatenate([np.sort(rng.choice(5000, size=t, replace=False)) for t in tfs])
    assert np.array_equal(delta_decode_runs(delta_encode_runs(positions, tfs), tfs), positions)


def test_gap_encoding_shrinks_dense_postings():
    docs = np.arange(0, 100_000, 3)
    raw = IntCodec("raw").encode(docs)
    compressed = IntCodec("vbyte").encode(delta_encode(docs))
    assert len(raw) == 4 * len(docs)
    assert len(compressed) == len(docs)  # every gap is 3 (or the first 0): one byte each
    assert IntCodec("raw").decode(raw).tolist() == docs.tolist()
