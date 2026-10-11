# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest

import collector.case_generator as cg

pytestmark = pytest.mark.unit


def test_sparse_max_query_tokens_requires_explicit_declaration():
    with pytest.raises(RuntimeError, match="sparse_max_query_tokens"):
        cg._dsv4_sparse_max_query_tokens({})


def test_sparse_max_query_tokens_null_disables_cap():
    assert cg._dsv4_sparse_max_query_tokens({"sparse_max_query_tokens": None}) is None


def test_sparse_max_query_tokens_declared_value():
    assert cg._dsv4_sparse_max_query_tokens({"sparse_max_query_tokens": 8192}) == 8192


def test_shipped_yaml_declares_serving_cap():
    assert cg._DSV4_SPARSE_MAX_QUERY_TOKENS == 8192


@pytest.mark.parametrize("kernel", ["paged_mqa_logits", "hca_attn"])
def test_sparse_rows_drop_over_budget_cases_and_count_them(kernel, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setattr(cg, "_selected_dsv4_models", lambda: ("model",))
    monkeypatch.setattr(cg, "_DSV4_SPARSE_BS_LIST", [1, 64])
    monkeypatch.setattr(cg, "_DSV4_SPARSE_ISL_LIST", [1, 8192])
    monkeypatch.setattr(cg, "_DSV4_SPARSE_PAST_KV_LIST", [0])
    monkeypatch.setattr(cg, "_DSV4_SPARSE_TP_LIST_ATTN", [1])
    monkeypatch.setattr(cg, "_DSV4_SPARSE_TP_LIST_INDEXER", [1])
    monkeypatch.setattr(cg, "_DSV4_SPARSE_MAX_QUERY_TOKENS", 8192)

    cases = cg._xpu_dsv4_sparse_rows(kernel)

    # (64, 8192) exceeds the 8192-token cap; (1, 8192) and (64, 1) stay.
    assert sorted((c[0], c[1], c[2]) for c in cases) == [(1, 1, 0), (1, 8192, 0), (64, 1, 0)]
    assert "dropped 1 case(s) (sparse_max_query_tokens=8192)" in capsys.readouterr().out


@pytest.mark.parametrize("kernel", ["paged_mqa_logits", "hca_attn"])
def test_sparse_rows_keep_everything_when_cap_is_null(kernel, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setattr(cg, "_selected_dsv4_models", lambda: ("model",))
    monkeypatch.setattr(cg, "_DSV4_SPARSE_BS_LIST", [1])
    monkeypatch.setattr(cg, "_DSV4_SPARSE_ISL_LIST", [8192])
    monkeypatch.setattr(cg, "_DSV4_SPARSE_PAST_KV_LIST", [0])
    monkeypatch.setattr(cg, "_DSV4_SPARSE_TP_LIST_ATTN", [1])
    monkeypatch.setattr(cg, "_DSV4_SPARSE_TP_LIST_INDEXER", [1])
    monkeypatch.setattr(cg, "_DSV4_SPARSE_MAX_QUERY_TOKENS", None)

    assert cg._xpu_dsv4_sparse_rows(kernel) == [[1, 8192, 0, 1, kernel, "model"]]
    assert "dropped" not in capsys.readouterr().out
