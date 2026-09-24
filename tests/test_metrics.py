# -*- coding: utf-8 -*-
"""绩效指标、胜率与 Markdown 表格渲染的测试。"""

import pandas as pd
import pytest

from conftest import (BUY, M_ANNUAL_RET, M_BARS, M_END, M_MAX_DD, M_SHARPE,
                      M_START, M_TOTAL_RET, M_TRADES, M_WIN_RATE, SELL,
                      UNKNOWN, FakeContextInfo, FakeDeal, make_nav, ramp)


def trade(code, direction, time, price, volume):
    """按 _win_rate / _calc_metrics 期望的顺序构造一笔成交。"""
    return [code, direction, time, price, volume]


def ctx_with_log(log):
    """构造只带 trade_log 的极简上下文。"""
    ctx = FakeContextInfo()
    ctx.trade_log = log
    return ctx


# ======================== _win_rate ========================
def test_win_rate_is_zero_without_trades(strategy):
    assert strategy._win_rate([]) == 0.0


def test_win_rate_counts_profitable_round_trip(strategy):
    trades = [trade('A.SH', BUY, '2024-01-01', 10.0, 100),
              trade('A.SH', SELL, '2024-01-02', 11.0, 100)]
    assert strategy._win_rate(trades) == pytest.approx(1.0)


def test_win_rate_counts_losing_round_trip(strategy):
    trades = [trade('A.SH', BUY, '2024-01-01', 10.0, 100),
              trade('A.SH', SELL, '2024-01-02', 9.0, 100)]
    assert strategy._win_rate(trades) == pytest.approx(0.0)


def test_win_rate_treats_breakeven_as_non_win(strategy):
    trades = [trade('A.SH', BUY, '2024-01-01', 10.0, 100),
              trade('A.SH', SELL, '2024-01-02', 10.0, 100)]
    assert strategy._win_rate(trades) == pytest.approx(0.0)


def test_win_rate_matches_fifo_across_partial_sells(strategy):
    """买入 100 股 @10，分两笔卖出：@11 盈利、@9 亏损 -> 胜率 50%。"""
    trades = [trade('A.SH', BUY, '2024-01-01', 10.0, 100),
              trade('A.SH', SELL, '2024-01-02', 11.0, 60),
              trade('A.SH', SELL, '2024-01-03', 9.0, 40)]
    assert strategy._win_rate(trades) == pytest.approx(0.5)


def test_win_rate_ignores_sells_beyond_available_position(strategy):
    """卖出量超过持仓量时，只按已匹配的部分计数。"""
    trades = [trade('A.SH', BUY, '2024-01-01', 10.0, 100),
              trade('A.SH', SELL, '2024-01-02', 11.0, 150)]
    assert strategy._win_rate(trades) == pytest.approx(1.0)


def test_win_rate_ignores_sell_without_prior_buy(strategy):
    trades = [trade('A.SH', SELL, '2024-01-02', 11.0, 100)]
    assert strategy._win_rate(trades) == 0.0


def test_win_rate_sorts_trades_by_time(strategy):
    """成交记录乱序时，应先按时间排序再 FIFO 匹配。"""
    trades = [trade('A.SH', SELL, '2024-01-02', 11.0, 100),
              trade('A.SH', BUY, '2024-01-01', 10.0, 100)]
    assert strategy._win_rate(trades) == pytest.approx(1.0)


def test_win_rate_keeps_codes_independent(strategy):
    """不同标的的持仓不能互相匹配。"""
    trades = [trade('A.SH', BUY, '2024-01-01', 10.0, 100),
              trade('B.SH', SELL, '2024-01-02', 20.0, 100)]
    assert strategy._win_rate(trades) == 0.0


def test_win_rate_ignores_unknown_direction(strategy):
    trades = [trade('A.SH', BUY, '2024-01-01', 10.0, 100),
              trade('A.SH', UNKNOWN, '2024-01-02', 11.0, 100),
              trade('A.SH', SELL, '2024-01-03', 11.0, 100)]
    assert strategy._win_rate(trades) == pytest.approx(1.0)


# ======================== _calc_metrics ========================
def test_metrics_defaults_when_nav_is_empty(strategy):
    m = strategy._calc_metrics([], [])
    assert m[M_BARS] == 0
    assert m[M_START] == 0.0
    assert m[M_END] == 0.0
    assert m[M_TOTAL_RET] == 0.0
    assert m[M_MAX_DD] == 0.0
    assert m[M_SHARPE] == 0.0
    assert m[M_WIN_RATE] == 0.0
    assert m[M_TRADES] == 0


def test_metrics_defaults_when_all_assets_are_non_positive(strategy):
    m = strategy._calc_metrics(make_nav([0.0, 0.0]), [])
    assert m[M_START] == 0.0
    assert m[M_TOTAL_RET] == 0.0


def test_metrics_returns_expected_keys(strategy):
    """键名是报告模板与下游用法的契约，改名需要同步改测试。"""
    assert set(strategy._calc_metrics([], [])) == {
        M_BARS, M_START, M_END, M_TOTAL_RET, M_ANNUAL_RET,
        M_MAX_DD, M_SHARPE, M_WIN_RATE, M_TRADES,
    }


def test_metrics_computes_return_and_drawdown(strategy):
    # 253 个交易日 -> 年化指数正好是 252/252 = 1，年化收益率与区间收益率相等
    values = ramp(100.0, 120.0, 101) + ramp(120.0, 90.0, 50) + ramp(90.0, 121.0, 102)
    m = strategy._calc_metrics(make_nav(values), [])

    assert m[M_BARS] == 253
    assert m[M_START] == pytest.approx(100.0)
    assert m[M_END] == pytest.approx(121.0)
    assert m[M_TOTAL_RET] == pytest.approx(21.0)
    assert m[M_ANNUAL_RET] == pytest.approx(21.0)
    # 峰值 120 回撤到 90 -> (120-90)/120 = 25%
    assert m[M_MAX_DD] == pytest.approx(25.0)


def test_metrics_max_drawdown_is_zero_for_monotonic_rise(strategy):
    m = strategy._calc_metrics(make_nav(ramp(100.0, 150.0, 60)), [])
    assert m[M_MAX_DD] == 0.0


def test_metrics_sharpe_is_zero_for_flat_nav(strategy):
    """净值恒定 -> 日收益率标准差为 0 -> 夏普取 0，避免除零。"""
    m = strategy._calc_metrics(make_nav([100.0] * 30), [])
    assert m[M_SHARPE] == 0.0


def test_metrics_sharpe_is_positive_for_rising_nav(strategy):
    values = [100.0 * (1.01 ** i) for i in range(60)]
    m = strategy._calc_metrics(make_nav(values), [])
    assert m[M_SHARPE] > 0


def test_metrics_filters_out_non_positive_assets(strategy):
    """净值序列里的 0（例如取数失败）不参与统计。"""
    m = strategy._calc_metrics(make_nav([100.0, 0.0, 110.0]), [])
    assert m[M_BARS] == 3
    assert m[M_START] == pytest.approx(100.0)
    assert m[M_END] == pytest.approx(110.0)


def test_metrics_reports_trade_count_and_win_rate(strategy):
    trades = [trade('A.SH', BUY, '2024-01-01', 10.0, 100),
              trade('A.SH', SELL, '2024-01-02', 11.0, 100),
              trade('B.SH', BUY, '2024-01-01', 10.0, 100),
              trade('B.SH', SELL, '2024-01-02', 9.0, 100)]
    m = strategy._calc_metrics(make_nav(ramp(100.0, 110.0, 30)), trades)

    assert m[M_TRADES] == 4
    assert m[M_WIN_RATE] == pytest.approx(50.0)


# ======================== Markdown 表格渲染 ========================
def test_md_table_renders_header_separator_and_body(strategy):
    rendered = strategy._md_table([['指标', '数值'], ['总交易数', 5], ['胜率', 50.0]])
    assert rendered.splitlines() == [
        '| 指标 | 数值 |',
        '| --- | --- |',
        '| 总交易数 | 5 |',
        '| 胜率 | 50.0 |',
    ]


def test_md_table_returns_empty_string_for_no_rows(strategy):
    assert strategy._md_table([]) == ''


def test_df_to_md_table_renders_dataframe(strategy):
    df = pd.DataFrame([{'代码': 'A.SH', '价格': 10.5}])
    assert strategy._df_to_md_table(df).splitlines() == [
        '| 代码 | 价格 |',
        '| --- | --- |',
        '| A.SH | 10.5 |',
    ]


# ======================== _collect_trades ========================
def test_collect_trades_reads_deal_records(strategy, qmt):
    qmt.deals = [FakeDeal('600519.SH', 0, 10.5, 1000, '2024-01-02 14:55:00'),
                 FakeDeal('600519.SH', 1, 11.0, 1000, '2024-01-03 14:55:00')]

    rows = strategy._collect_trades(ctx_with_log([]), 'acc')

    assert rows == [['600519.SH', BUY, '2024-01-02 14:55:00', 10.5, 1000],
                    ['600519.SH', SELL, '2024-01-03 14:55:00', 11.0, 1000]]


def test_collect_trades_skips_zero_volume_deals(strategy, qmt):
    qmt.deals = [FakeDeal('600519.SH', 0, 10.5, 0)]
    assert strategy._collect_trades(ctx_with_log([]), 'acc') == []


def test_collect_trades_falls_back_to_local_log(strategy, qmt):
    """券商 DEAL 查不到时（回测常见），用 handlebar 自己记的成交兜底。"""
    qmt.deals = []
    log = [{'股票代码': '600519.SH', '方向': BUY, '成交时间': '2024-01-02',
            '成交价格': 10.5, '成交量': 1000}]

    rows = strategy._collect_trades(ctx_with_log(log), 'acc')

    assert rows == [['600519.SH', BUY, '2024-01-02', 10.5, 1000]]


def test_collect_trades_falls_back_when_deal_api_raises(strategy, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError('模拟成交查询失败')

    monkeypatch.setattr(strategy, 'get_trade_detail_data', boom, raising=False)
    log = [{'股票代码': '600519.SH', '方向': SELL, '成交时间': '2024-01-02',
            '成交价格': 11.0, '成交量': 500}]

    rows = strategy._collect_trades(ctx_with_log(log), 'acc')

    assert rows == [['600519.SH', SELL, '2024-01-02', 11.0, 500]]
