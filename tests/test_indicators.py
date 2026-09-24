# -*- coding: utf-8 -*-
"""技术指标计算测试：EMA 与 MACD。"""

import numpy as np
import pytest

from conftest import ramp


# ======================== _ema ========================
def test_ema_empty_input_returns_empty(strategy):
    assert strategy._ema([], 12).size == 0


def test_ema_single_value_equals_input(strategy):
    assert list(strategy._ema([3.0], 12)) == [3.0]


def test_ema_matches_hand_computed_values(strategy):
    # period=3 -> alpha = 2/(3+1) = 0.5
    # out[0]=1, out[1]=0.5*2+0.5*1=1.5, out[2]=0.5*3+0.5*1.5=2.25
    result = strategy._ema([1.0, 2.0, 3.0], 3)
    assert list(result) == pytest.approx([1.0, 1.5, 2.25])


def test_ema_preserves_length_and_returns_ndarray(strategy):
    result = strategy._ema([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    assert isinstance(result, np.ndarray)
    assert result.size == 5


def test_ema_of_constant_series_is_that_constant(strategy):
    assert list(strategy._ema([7.0] * 20, 5)) == pytest.approx([7.0] * 20)


def test_ema_first_value_is_seeded_with_first_close(strategy):
    """EMA 以首个收盘价作为初值，不做 SMA 预热。"""
    result = strategy._ema([10.0, 100.0], 2)   # alpha = 2/3
    assert result[0] == pytest.approx(10.0)
    assert result[1] == pytest.approx(2 / 3 * 100 + 1 / 3 * 10)


# ======================== calc_macd ========================
def test_calc_macd_returns_none_when_data_too_short(strategy):
    # slow=26，数据不足 26 根时无法计算
    assert strategy.calc_macd([10.0] * 25, 12, 26, 9) == (None, None, None)


def test_calc_macd_computes_on_exactly_slow_bars(strategy):
    """恰好 slow 根 K 线即可计算，不应返回 None。"""
    dif, dea, hist = strategy.calc_macd([10.0] * 26, 12, 26, 9)
    assert dif is not None and dea is not None and hist is not None


def test_calc_macd_of_constant_series_is_zero(strategy):
    dif, dea, hist = strategy.calc_macd([10.0] * 60, 12, 26, 9)
    assert (dif, dea, hist) == pytest.approx((0.0, 0.0, 0.0))


def test_calc_macd_hist_equals_dif_minus_dea(strategy):
    closes = ramp(10.0, 20.0, 60)
    dif, dea, hist = strategy.calc_macd(closes, 12, 26, 9)
    assert hist == pytest.approx(dif - dea)


def test_calc_macd_matches_manual_ema_chain(strategy):
    """用 _ema 手工串一遍 DIF/DEA，验证 calc_macd 的公式实现。"""
    closes = ramp(10.0, 15.0, 80) + ramp(15.0, 9.0, 40)

    dif, dea, hist = strategy.calc_macd(closes, 12, 26, 9)

    expected_dif = strategy._ema(closes, 12) - strategy._ema(closes, 26)
    expected_dea = strategy._ema(expected_dif, 9)
    assert dif == pytest.approx(expected_dif[-1])
    assert dea == pytest.approx(expected_dea[-1])
    assert hist == pytest.approx(expected_dif[-1] - expected_dea[-1])


def test_calc_macd_positive_on_rising_series(strategy):
    """持续上涨时快线在慢线之上，DIF 与柱状线均为正。"""
    dif, dea, hist = strategy.calc_macd(ramp(10.0, 30.0, 60), 12, 26, 9)
    assert dif > 0
    assert hist > 0


def test_calc_macd_negative_on_falling_series(strategy):
    dif, dea, hist = strategy.calc_macd(ramp(30.0, 10.0, 60), 12, 26, 9)
    assert dif < 0
    assert hist < 0


def test_calc_macd_uses_module_default_parameters(strategy):
    closes = ramp(10.0, 20.0, 60)
    assert strategy.calc_macd(closes) == pytest.approx(
        strategy.calc_macd(closes, strategy.MACD_FAST,
                           strategy.MACD_SLOW, strategy.MACD_SIGNAL))


def test_calc_macd_accepts_pandas_series(strategy):
    """策略里传入的是 numpy/列表，顺带确认 pandas Series 也能吃。"""
    import pandas as pd
    closes = pd.Series(ramp(10.0, 20.0, 60))
    dif, dea, hist = strategy.calc_macd(closes, 12, 26, 9)
    assert hist == pytest.approx(dif - dea)
