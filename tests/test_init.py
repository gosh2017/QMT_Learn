# -*- coding: utf-8 -*-
"""init 初始化逻辑与源码编码约定的测试。"""

from conftest import STRATEGY_PATH, FakeContextInfo, load_strategy_module


# ======================== 资金账号 ========================
def test_init_uses_configured_account_id(strategy, qmt, monkeypatch):
    monkeypatch.setattr(strategy, 'ACCOUNT_ID', 'MY_ACCOUNT')
    ctx = FakeContextInfo()
    strategy.init(ctx)
    assert ctx.accountid == 'MY_ACCOUNT'


def test_init_keeps_account_id_from_backtest_config(strategy, qmt):
    """回测参数里选的账号优先，不应被覆盖。"""
    ctx = FakeContextInfo(accountid='BACKTEST_ACCOUNT')
    strategy.init(ctx)
    assert ctx.accountid == 'BACKTEST_ACCOUNT'


def test_init_falls_back_to_placeholder_account(strategy, qmt):
    """没有账号时用占位账号兜底，避免后续 AttributeError。"""
    ctx = FakeContextInfo(accountid=None)
    strategy.init(ctx)
    assert ctx.accountid == 'QMT_BACKTEST'


# ======================== 股票池 ========================
def test_init_uses_sector_stock_list(strategy, qmt):
    qmt.sector = ['600519.SH', '000858.SZ']
    ctx = FakeContextInfo()

    strategy.init(ctx)

    assert ctx.stock_list == ['600519.SH', '000858.SZ']
    assert ctx.universe == ['600519.SH', '000858.SZ']


def test_init_filters_delisted_code(strategy, qmt):
    """600837.SH 已被国泰海通吸收合并退市，应从股票池中剔除。"""
    qmt.sector = ['600519.SH', '600837.SH']
    ctx = FakeContextInfo()

    strategy.init(ctx)

    assert ctx.stock_list == ['600519.SH']


def test_init_falls_back_to_context_stock_list(strategy, monkeypatch):
    """全局 get_stock_list_in_sector 不可用时，退回 ContextInfo 的同名方法。"""
    def boom(name):
        raise RuntimeError('模拟全局接口不可用')

    monkeypatch.setattr(strategy, 'get_stock_list_in_sector', boom, raising=False)
    ctx = FakeContextInfo()
    ctx.get_stock_list_in_sector = lambda name: ['000001.SZ']

    strategy.init(ctx)

    assert ctx.stock_list == ['000001.SZ']


def test_init_falls_back_to_builtin_stock_list(strategy, monkeypatch):
    """接口完全取不到数据时，用内置兜底股票池，保证回测仍能跑通。"""
    monkeypatch.setattr(strategy, 'get_stock_list_in_sector',
                        lambda name: [], raising=False)
    ctx = FakeContextInfo()
    ctx.get_stock_list_in_sector = lambda name: []

    strategy.init(ctx)

    assert len(ctx.stock_list) == 7
    assert '600519.SH' in ctx.stock_list
    assert '600837.SH' not in ctx.stock_list


def test_init_survives_set_universe_failure(strategy, qmt):
    qmt.sector = ['600519.SH']
    ctx = FakeContextInfo()

    def boom(stock_list):
        raise RuntimeError('模拟 set_universe 失败')

    ctx.set_universe = boom

    strategy.init(ctx)                      # 不应抛出

    assert ctx.stock_list == ['600519.SH']


# ======================== 运行时状态 ========================
def test_init_initialises_runtime_state(strategy, qmt):
    qmt.sector = ['600519.SH']
    ctx = FakeContextInfo()

    strategy.init(ctx)

    assert ctx.nav_list == []
    assert ctx.my_holdings == {}
    assert ctx.peak_profit == {}
    assert ctx.trade_log == []
    assert ctx.warned_no_asset is False


def test_init_can_run_twice_without_error(strategy, qmt):
    """QMT 重跑回测时会重新调用 init，状态必须被重置。"""
    qmt.sector = ['600519.SH']
    ctx = FakeContextInfo()
    strategy.init(ctx)
    ctx.nav_list.append(('2024-01-01', 1.0))
    ctx.my_holdings['600519.SH'] = 100

    strategy.init(ctx)

    assert ctx.nav_list == []
    assert ctx.my_holdings == {}


# ======================== 源码与回调约定 ========================
def test_strategy_source_is_gbk_encoded():
    """QMT 模型研究按 GBK 读取策略源码，首行必须声明编码，否则中文全乱码。"""
    raw = STRATEGY_PATH.read_bytes()
    first_line = raw.splitlines()[0]
    assert b'coding: gbk' in first_line or b'coding=gbk' in first_line
    raw.decode('gbk')                       # 整个文件必须能被 GBK 解码


def test_strategy_exposes_qmt_callbacks():
    """QMT 要求策略提供 init / handlebar / stop 三个回调。"""
    module = load_strategy_module()
    for name in ('init', 'handlebar', 'stop'):
        assert callable(getattr(module, name)), name


def test_strategy_imports_without_qmt_runtime():
    """QMT 内置函数只在函数体内调用，模块本身应能在 QMT 之外导入。"""
    module = load_strategy_module()
    assert module.MA_PERIOD > 0
    assert module.MAX_HOLDINGS > 0
    assert 0 < module.PROFIT_STOP_RATIO < 1
