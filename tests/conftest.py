# -*- coding: utf-8 -*-
"""
单元测试公共夹具。

被测对象是 QMT（迅投）〖模型研究〗里的策略脚本 strategy/ma_macd_dual_signal.py，
它有两个不便直接测试的地方，本文件负责抹平：

1. 源码为 **GBK 编码**（首行 `# -*- coding: gbk -*-`，QMT 要求）。
   用 importlib 按文件路径加载即可，Python 会依据该编码声明正确解码。

2. 依赖 QMT 内置的全局函数（passorder / get_trade_detail_data / timetostr /
   get_stock_list_in_sector）与 ContextInfo 上下文对象，这些在 QMT 之外都不存在。
   这里用假实现（Fake*）注入模块命名空间来模拟，测试用例即可脱离 QMT 客户端运行。
"""

import datetime as _dt
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = PROJECT_ROOT / 'strategy' / 'ma_macd_dual_signal.py'


# ======================== 模块加载 ========================
def load_strategy_module():
    """按文件路径把策略脚本加载成模块对象（每次调用返回全新副本）。"""
    spec = importlib.util.spec_from_file_location('ma_macd_dual_signal', STRATEGY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def strategy():
    """每个用例一份全新的策略模块，避免模块级全局状态（如 REPORT_DIR）在用例间串味。"""
    return load_strategy_module()


# ======================== 券商返回对象的假实现 ========================
class FakeAccount:
    """模拟 get_trade_detail_data(..., 'ACCOUNT') 返回的账户对象。"""

    def __init__(self, total_asset=1_000_000.0, cash=1_000_000.0):
        self.m_dTotalAsset = total_asset
        self.m_dBalance = cash


class FakePosition:
    """模拟 get_trade_detail_data(..., 'POSITION') 返回的持仓对象。"""

    def __init__(self, code, volume=1000, can_use=None, profit_rate=None, cost_price=None):
        self.m_strInstrumentID = code
        self.m_nVolume = volume
        # 可选字段：不传就当作券商没返回该属性，用于覆盖策略里的兜底分支
        if can_use is not None:
            self.m_nCanUseVolume = can_use
        if profit_rate is not None:
            self.m_dProfitRate = profit_rate
        if cost_price is not None:
            self.m_dCostPrice = cost_price


class FakeDeal:
    """模拟 get_trade_detail_data(..., 'DEAL') 返回的成交记录对象。"""

    def __init__(self, code, direction, price, volume, time='2024-01-02 14:55:00'):
        self.m_strInstrumentID = code
        self.m_nDirection = direction
        self.m_dPrice = price
        self.m_nVolume = volume
        self.m_strTradeTime = time


class FakeBroker:
    """
    模拟券商侧：记录 passorder 下单请求，并按需返回账户 / 持仓 / 成交数据。

    passorder 一旦被调用即视为成交（策略里用的 quickTrade=1），
    因此这里只记录参数，不模拟撮合与费用。
    """

    def __init__(self, total_asset=1_000_000.0, cash=1_000_000.0,
                 positions=(), deals=(), sector=None):
        self.orders = []                 # 每笔 passorder 的参数，便于断言
        self.total_asset = total_asset
        self.cash = cash
        self.positions = list(positions)
        self.deals = list(deals)
        self.sector = list(sector or [])  # get_stock_list_in_sector 的返回值
        self.raise_on_order = False       # 置 True 可模拟下单接口抛异常

    # ---- 假 QMT 内置函数 ----
    def passorder(self, opType, orderType, accountid, stock, prType, price, volume,
                  strategyName, quickTrade, userOrderId, ContextInfo):
        if self.raise_on_order:
            raise RuntimeError('模拟下单接口异常')
        self.orders.append({
            'opType': opType, 'orderType': orderType, 'accountid': accountid,
            'stock': stock, 'prType': prType, 'price': price, 'volume': volume,
            'strategyName': strategyName, 'quickTrade': quickTrade,
        })

    def get_trade_detail_data(self, account, kind, field):
        if field == 'ACCOUNT':
            return [FakeAccount(self.total_asset, self.cash)]
        if field == 'POSITION':
            return list(self.positions)
        if field == 'DEAL':
            return list(self.deals)
        return []

    def get_stock_list_in_sector(self, name):
        return list(self.sector)

    @property
    def buy_orders(self):
        return [o for o in self.orders if o['opType'] == 0]

    @property
    def sell_orders(self):
        return [o for o in self.orders if o['opType'] == 1]


def fake_timetostr(timetag, fmt='%Y-%m-%d'):
    """模拟 QMT 的 timetostr：把时间戳格式化成字符串。"""
    return _dt.datetime.fromtimestamp(timetag, _dt.timezone.utc).strftime(fmt)


@pytest.fixture()
def qmt(strategy, monkeypatch):
    """把 QMT 内置函数替换成假实现，返回可断言下单请求的 FakeBroker。"""
    broker = FakeBroker()
    monkeypatch.setattr(strategy, 'passorder', broker.passorder, raising=False)
    monkeypatch.setattr(strategy, 'get_trade_detail_data',
                        broker.get_trade_detail_data, raising=False)
    monkeypatch.setattr(strategy, 'get_stock_list_in_sector',
                        broker.get_stock_list_in_sector, raising=False)
    monkeypatch.setattr(strategy, 'timetostr', fake_timetostr, raising=False)
    return broker


# ======================== ContextInfo 的假实现 ========================
class FakeContextInfo:
    """
    模拟 QMT 传给 init / handlebar / stop 的 ContextInfo 上下文对象。

    history 形如 {'600519.SH': [收盘价, ...]}，是 get_history_data 的数据源。
    """

    def __init__(self, history=None, accountid='TEST_ACCOUNT', start_date='2024-01-01'):
        self._history = dict(history or {})
        self._start_date = _dt.date.fromisoformat(start_date)
        self.accountid = accountid
        self.barpos = 0
        self.universe = None
        self.history_calls = []          # 记录取数调用，便于断言参数

    def get_history_data(self, stock, field, period=-1, count=None):
        """默认实现 4 参数签名（较新 QMT 版本）。"""
        self.history_calls.append((stock, field, period, count))
        if stock not in self._history:
            return None
        data = list(self._history[stock])
        return data[-count:] if count else data

    def set_universe(self, stock_list):
        self.universe = list(stock_list)

    def get_bar_timetag(self, barpos):
        """用「起始日 + barpos 天」造时间戳，让 _date_str 能产出可读日期。"""
        day = self._start_date + _dt.timedelta(days=barpos)
        return _dt.datetime(day.year, day.month, day.day,
                            tzinfo=_dt.timezone.utc).timestamp()


class LegacyContextInfo(FakeContextInfo):
    """模拟旧版 QMT：get_history_data 只接受 3 个参数，用于覆盖签名兜底分支。"""

    def get_history_data(self, stock, field, count):
        return super().get_history_data(stock, field, -1, count)


# ======================== 与策略源码（GBK）对齐的常量 ========================
# 策略内部统一用这三个值表示买卖方向
BUY = '买入'
SELL = '卖出'
UNKNOWN = '未知'

# 策略写入 trade_log 的字段名
K_CODE = '股票代码'
K_DIR = '方向'
K_TIME = '成交时间'
K_PRICE = '成交价格'
K_VOL = '成交量'

# _calc_metrics 返回字典的键
M_BARS = '总交易日'
M_START = '期初资产'
M_END = '期末资产'
M_TOTAL_RET = '总收益率(%)'
M_ANNUAL_RET = '年化收益率(%)'
M_MAX_DD = '最大回撤(%)'
M_SHARPE = '夏普比率'
M_WIN_RATE = '胜率(%)'
M_TRADES = '总交易次数'

# 导出 CSV 的列名
TRADE_COLUMNS = [K_CODE, K_DIR, K_TIME, K_PRICE, K_VOL]
NAV_COLUMNS = ['日期', '总资产', '日收益率']


# ======================== 测试数据构造工具 ========================
def make_ctx(history=None, **kwargs):
    """构造一个已初始化运行时状态的 ContextInfo。"""
    ctx = FakeContextInfo(history=history, **kwargs)
    ctx.my_holdings = {}
    ctx.peak_profit = {}
    ctx.trade_log = []
    ctx.nav_list = []
    ctx.warned_no_asset = False
    return ctx


def make_nav(values, start_date='2024-01-01'):
    """把净值序列转成策略里 nav_list 的格式：[(日期, 总资产), ...]。"""
    base = _dt.date.fromisoformat(start_date)
    return [((base + _dt.timedelta(days=i)).isoformat(), float(v))
            for i, v in enumerate(values)]


def ramp(start, end, n):
    """生成 n 个从 start 线性过渡到 end 的数值（含首尾）。"""
    if n <= 1:
        return [float(start)]
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def cross_up_series(length=40, base=10.0, dip=9.5, spike=10.5):
    """
    构造「收盘价上穿 20 日均线」的收盘价序列：
    前 length-2 根横盘，倒数第 2 根下探（prev_close < prev_ma），最后一根拉升（close > ma）。
    """
    return [base] * (length - 2) + [dip, spike]


def cross_down_series(length=40, base=10.0, spike=10.5, dip=9.5):
    """构造「收盘价下穿 20 日均线」的收盘价序列（与上穿互为镜像）。"""
    return [base] * (length - 2) + [spike, dip]


def flat_series(length=40, base=10.0):
    """构造横盘序列：既不产生上穿也不产生下穿信号。"""
    return [base] * length
