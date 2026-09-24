# -*- coding: utf-8 -*-
"""回测报告导出（CSV / Markdown）与 stop 回调的测试。"""

import os

import pandas as pd
import pytest

from conftest import (BUY, K_CODE, K_DIR, K_PRICE, K_TIME, K_VOL, NAV_COLUMNS,
                      TRADE_COLUMNS, make_ctx)

BOM = b'\xef\xbb\xbf'


def report_ctx(tmp_path, monkeypatch, strategy, nav=None, log=None):
    """构造一个指向临时目录的报告上下文。"""
    monkeypatch.setattr(strategy, 'REPORT_DIR', str(tmp_path) + os.sep)
    ctx = make_ctx(accountid='acc')
    ctx.nav_list = nav if nav is not None else [('2024-01-01', 1_000_000.0),
                                                ('2024-01-02', 1_010_000.0)]
    ctx.trade_log = log if log is not None else [
        {K_CODE: '600519.SH', K_DIR: BUY, K_TIME: '2024-01-02',
         K_PRICE: 10.5, K_VOL: 1000},
    ]
    return ctx


# ======================== _build_markdown ========================
def test_build_markdown_includes_period_and_sector(strategy):
    nav_df = pd.DataFrame([['2024-01-01', 100.0], ['2024-01-05', 110.0]],
                          columns=['日期', '总资产'])
    metrics = strategy._calc_metrics([('2024-01-01', 100.0),
                                      ('2024-01-05', 110.0)], [])
    trades_df = pd.DataFrame(columns=TRADE_COLUMNS)

    md = strategy._build_markdown(None, metrics, trades_df, nav_df)

    assert md.startswith('# ')
    assert '2024-01-01' in md and '2024-01-05' in md
    assert strategy.SECTOR_NAME in md


def test_build_markdown_handles_empty_nav(strategy):
    metrics = strategy._calc_metrics([], [])
    trades_df = pd.DataFrame(columns=TRADE_COLUMNS)

    md = strategy._build_markdown(None, metrics, trades_df,
                                  pd.DataFrame(columns=['日期', '总资产']))

    assert md.startswith('# ')
    assert 'N/A' in md


def test_build_markdown_lists_trades(strategy):
    nav_df = pd.DataFrame([['2024-01-01', 100.0]], columns=['日期', '总资产'])
    metrics = strategy._calc_metrics([('2024-01-01', 100.0)], [])
    trades_df = pd.DataFrame([['600519.SH', BUY, '2024-01-02', 10.5, 1000]],
                             columns=TRADE_COLUMNS)

    md = strategy._build_markdown(None, metrics, trades_df, nav_df)

    assert '600519.SH' in md


def test_build_markdown_limits_trades_to_ten_rows(strategy):
    nav_df = pd.DataFrame([['2024-01-01', 100.0]], columns=['日期', '总资产'])
    metrics = strategy._calc_metrics([('2024-01-01', 100.0)], [])
    trades_df = pd.DataFrame(
        [[f'CODE{i:02d}.SZ', BUY, '2024-01-02', 10.0, 100] for i in range(15)],
        columns=TRADE_COLUMNS)

    md = strategy._build_markdown(None, metrics, trades_df, nav_df)

    assert 'CODE09.SZ' in md            # 第 10 笔仍进表格
    assert 'CODE10.SZ' not in md        # 第 11 笔起被截断


# ======================== _export_reports ========================
def test_export_reports_writes_all_three_files(strategy, qmt, tmp_path, monkeypatch):
    ctx = report_ctx(tmp_path, monkeypatch, strategy)

    strategy._export_reports(ctx)

    for name in ('backtest_trades.csv', 'backtest_nav.csv', 'backtest_report.md'):
        assert (tmp_path / name).exists(), name


def test_export_reports_creates_missing_directory(strategy, qmt, tmp_path, monkeypatch):
    target = tmp_path / 'nested' / 'reports'
    monkeypatch.setattr(strategy, 'REPORT_DIR', str(target) + os.sep)

    ctx = make_ctx(accountid='acc')
    ctx.nav_list = [('2024-01-01', 1_000_000.0)]
    ctx.trade_log = []

    strategy._export_reports(ctx)

    assert (target / 'backtest_report.md').exists()


def test_trades_csv_content_and_encoding(strategy, qmt, tmp_path, monkeypatch):
    ctx = report_ctx(tmp_path, monkeypatch, strategy)

    strategy._export_reports(ctx)

    path = tmp_path / 'backtest_trades.csv'
    assert path.read_bytes().startswith(BOM)      # utf-8-sig，Excel 打开不乱码
    df = pd.read_csv(path)
    assert list(df.columns) == TRADE_COLUMNS
    assert df.iloc[0][K_CODE] == '600519.SH'
    assert df.iloc[0][K_DIR] == BUY
    assert df.iloc[0][K_VOL] == 1000


def test_nav_csv_has_daily_return_column(strategy, qmt, tmp_path, monkeypatch):
    ctx = report_ctx(tmp_path, monkeypatch, strategy)

    strategy._export_reports(ctx)

    path = tmp_path / 'backtest_nav.csv'
    assert path.read_bytes().startswith(BOM)
    df = pd.read_csv(path)
    assert list(df.columns) == NAV_COLUMNS
    assert df['总资产'].tolist() == [1_000_000.0, 1_010_000.0]
    assert pd.isna(df['日收益率'].iloc[0])         # 首日无前收，收益率为空
    assert df['日收益率'].iloc[1] == pytest.approx(0.01)


def test_export_reports_without_trades_or_nav(strategy, qmt, tmp_path, monkeypatch):
    """回测没有任何成交/净值时也应正常产出文件，而不是抛异常。"""
    ctx = report_ctx(tmp_path, monkeypatch, strategy, nav=[], log=[])

    strategy._export_reports(ctx)

    assert pd.read_csv(tmp_path / 'backtest_trades.csv').empty
    assert pd.read_csv(tmp_path / 'backtest_nav.csv').empty
    assert (tmp_path / 'backtest_report.md').exists()


def test_report_markdown_is_utf8(strategy, qmt, tmp_path, monkeypatch):
    ctx = report_ctx(tmp_path, monkeypatch, strategy)

    strategy._export_reports(ctx)

    text = (tmp_path / 'backtest_report.md').read_text(encoding='utf-8')
    assert '600519.SH' in text


# ======================== stop 回调 ========================
def test_stop_exports_reports(strategy, qmt, tmp_path, monkeypatch):
    ctx = report_ctx(tmp_path, monkeypatch, strategy)

    strategy.stop(ctx)

    assert (tmp_path / 'backtest_report.md').exists()


def test_stop_swallows_export_failure(strategy, monkeypatch):
    """导出失败不能影响回测本身，stop() 必须吞掉异常。"""
    def boom(ContextInfo):
        raise RuntimeError('模拟导出失败')

    monkeypatch.setattr(strategy, '_export_reports', boom)

    strategy.stop(make_ctx())        # 不应抛出
