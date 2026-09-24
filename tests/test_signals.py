# -*- coding: utf-8 -*-
"""handlebar 的买卖信号与仓位控制测试。"""

import pytest

from conftest import (FakePosition, cross_down_series, cross_up_series,
                      flat_series, make_ctx)

STRONG = '600519.SH'
WEAK = '000858.SZ'

# 两个都能触发「上穿 + 柱状线为正」的形态，但强势标的的柱状线明显更大
STRONG_SERIES = cross_up_series(40, spike=10.8)
WEAK_SERIES = cross_up_series(40, spike=10.5)


def setup_ctx(history, stock_list, accountid='acc'):
    """构造一个只差 handlebar 调用的上下文。"""
    ctx = make_ctx(history=history, accountid=accountid)
    ctx.stock_list = list(stock_list)
    return ctx


# ======================== 买入信号 ========================
def test_buys_on_ma_cross_up_with_positive_macd_hist(strategy, qmt):
    ctx = setup_ctx({STRONG: cross_up_series(40)}, [STRONG])

    strategy.handlebar(ctx)

    assert len(qmt.buy_orders) == 1
    order = qmt.buy_orders[0]
    assert order['stock'] == STRONG
    assert order['price'] == pytest.approx(10.5)
    # 总资产 100 万 / 最多持仓 5 只 = 20 万，按收盘价 10.5 元取整到 100 股
    assert order['volume'] == 19000
    assert ctx.my_holdings[STRONG] == 19000


@pytest.mark.parametrize('hist', [-0.01, 0.0])
def test_does_not_buy_without_positive_macd_hist(strategy, qmt, monkeypatch, hist):
    """仅有均线上穿还不够：柱状线必须严格大于 0。"""
    monkeypatch.setattr(strategy, 'calc_macd', lambda *a, **k: (0.0, 0.0, hist))
    ctx = setup_ctx({STRONG: cross_up_series(40)}, [STRONG])

    strategy.handlebar(ctx)

    assert qmt.buy_orders == []


def test_does_not_buy_when_price_stays_below_ma(strategy, qmt):
    ctx = setup_ctx({STRONG: flat_series(40)}, [STRONG])
    strategy.handlebar(ctx)
    assert qmt.buy_orders == []


def test_skips_stock_with_insufficient_history(strategy, qmt):
    """历史 K 线不足以计算 MACD（26+9）时直接跳过，不应报错。"""
    ctx = setup_ctx({STRONG: cross_up_series(30)}, [STRONG])
    strategy.handlebar(ctx)
    assert qmt.buy_orders == []


def test_skips_stock_without_history_data(strategy, qmt):
    ctx = setup_ctx({}, [STRONG])
    strategy.handlebar(ctx)
    assert qmt.buy_orders == []


def test_does_not_buy_stock_already_held(strategy, qmt):
    ctx = setup_ctx({STRONG: cross_up_series(40)}, [STRONG])
    ctx.my_holdings[STRONG] = 1000

    strategy.handlebar(ctx)

    assert qmt.buy_orders == []


# ======================== 买入排序与仓位约束 ========================
def test_prefers_stronger_macd_momentum_when_slots_are_limited(strategy, qmt):
    """持仓名额不足时，按 MACD 柱状线从强到弱挑标的。"""
    # 先确认前提：强势标的的柱状线确实更大，且两者都为正
    assert strategy.calc_macd(STRONG_SERIES)[2] > strategy.calc_macd(WEAK_SERIES)[2] > 0

    ctx = setup_ctx({WEAK: WEAK_SERIES, STRONG: STRONG_SERIES},
                    [WEAK, STRONG])
    ctx.my_holdings.update({f'FILL{i}.SH': 100 for i in range(4)})   # 只剩 1 个名额

    strategy.handlebar(ctx)

    assert [o['stock'] for o in qmt.buy_orders] == [STRONG]


def test_buy_order_follows_momentum_ranking(strategy, qmt):
    ctx = setup_ctx({WEAK: WEAK_SERIES, STRONG: STRONG_SERIES},
                    [WEAK, STRONG])

    strategy.handlebar(ctx)

    assert [o['stock'] for o in qmt.buy_orders] == [STRONG, WEAK]


def test_stops_buying_when_holdings_reach_limit(strategy, qmt):
    ctx = setup_ctx({STRONG: cross_up_series(40)}, [STRONG])
    ctx.my_holdings.update({f'FILL{i}.SH': 100 for i in range(strategy.MAX_HOLDINGS)})

    strategy.handlebar(ctx)

    assert qmt.buy_orders == []


def test_skips_buy_when_cash_is_below_one_lot(strategy, qmt):
    """现金不足一手（100 股）时不下单。"""
    qmt.cash = 1000.0
    ctx = setup_ctx({STRONG: cross_up_series(40)}, [STRONG])

    strategy.handlebar(ctx)

    assert qmt.buy_orders == []


def test_buy_size_never_exceeds_available_cash(strategy, qmt):
    """可用现金少于等权目标市值时，按现金的 98% 下单。"""
    qmt.cash = 50_000.0
    ctx = setup_ctx({STRONG: cross_up_series(40)}, [STRONG])

    strategy.handlebar(ctx)

    order = qmt.buy_orders[0]
    assert order['volume'] == 4600            # int(50000*0.98/10.5/100)*100
    assert order['volume'] * order['price'] <= 50_000.0


def test_buy_volume_is_rounded_to_lots_of_100(strategy, qmt):
    ctx = setup_ctx({STRONG: cross_up_series(40)}, [STRONG])
    strategy.handlebar(ctx)
    assert qmt.buy_orders[0]['volume'] % 100 == 0


# ======================== 卖出信号 ========================
def test_sells_on_ma_cross_down(strategy, qmt):
    ctx = setup_ctx({STRONG: cross_down_series(40)}, [])
    ctx.my_holdings[STRONG] = 1000
    qmt.positions = [FakePosition(STRONG, volume=1000, can_use=1000)]

    strategy.handlebar(ctx)

    assert len(qmt.sell_orders) == 1
    assert qmt.sell_orders[0]['volume'] == 1000
    assert qmt.sell_orders[0]['price'] == pytest.approx(9.5)
    assert STRONG not in ctx.my_holdings


def test_sells_when_profit_drawdown_exceeds_threshold(strategy, qmt):
    """持仓盈利从 10% 回撤到 4%，回撤 6% > 5% 阈值，触发止盈。"""
    ctx = setup_ctx({STRONG: flat_series(40)}, [])
    ctx.my_holdings[STRONG] = 1000

    qmt.positions = [FakePosition(STRONG, volume=1000, can_use=1000, profit_rate=0.10)]
    strategy.handlebar(ctx)
    assert qmt.sell_orders == []              # 首次仅记录峰值，回撤为 0
    assert ctx.peak_profit[STRONG] == pytest.approx(0.10)

    qmt.positions = [FakePosition(STRONG, volume=1000, can_use=1000, profit_rate=0.04)]
    strategy.handlebar(ctx)

    assert len(qmt.sell_orders) == 1
    assert qmt.sell_orders[0]['volume'] == 1000


def test_holds_when_drawdown_is_below_threshold(strategy, qmt):
    ctx = setup_ctx({STRONG: flat_series(40)}, [])
    ctx.my_holdings[STRONG] = 1000

    qmt.positions = [FakePosition(STRONG, volume=1000, can_use=1000, profit_rate=0.10)]
    strategy.handlebar(ctx)
    qmt.positions = [FakePosition(STRONG, volume=1000, can_use=1000, profit_rate=0.07)]
    strategy.handlebar(ctx)

    assert qmt.sell_orders == []


def test_holds_losing_position_without_ma_cross_down(strategy, qmt):
    """从未盈利过的持仓不触发「盈利回撤止盈」，只能等均线下穿。"""
    ctx = setup_ctx({STRONG: flat_series(40)}, [])
    ctx.my_holdings[STRONG] = 1000

    qmt.positions = [FakePosition(STRONG, volume=1000, can_use=1000, profit_rate=-0.20)]
    strategy.handlebar(ctx)

    assert qmt.sell_orders == []
    assert ctx.peak_profit[STRONG] == pytest.approx(0.0)


def test_holds_position_when_profit_turns_negative_after_peak(strategy, qmt):
    """回撤虽已超阈值，但当前仍为亏损时不按「盈利回撤」卖出。"""
    ctx = setup_ctx({STRONG: flat_series(40)}, [])
    ctx.my_holdings[STRONG] = 1000

    qmt.positions = [FakePosition(STRONG, volume=1000, can_use=1000, profit_rate=0.10)]
    strategy.handlebar(ctx)
    qmt.positions = [FakePosition(STRONG, volume=1000, can_use=1000, profit_rate=-0.02)]
    strategy.handlebar(ctx)

    assert qmt.sell_orders == []


def test_respects_t_plus_one_restriction(strategy, qmt):
    """当日买入的持仓 m_nCanUseVolume 为 0，即使触发信号也不能卖出。"""
    ctx = setup_ctx({STRONG: cross_down_series(40)}, [])
    ctx.my_holdings[STRONG] = 1000
    qmt.positions = [FakePosition(STRONG, volume=1000, can_use=0)]

    strategy.handlebar(ctx)

    assert qmt.sell_orders == []
    assert ctx.my_holdings[STRONG] == 1000


def test_does_not_sell_when_no_history(strategy, qmt):
    ctx = setup_ctx({}, [])
    ctx.my_holdings[STRONG] = 1000
    qmt.positions = [FakePosition(STRONG, volume=1000, can_use=1000)]

    strategy.handlebar(ctx)

    assert qmt.sell_orders == []


def test_sell_happens_before_buy_in_same_bar(strategy, qmt):
    """同一根 K 线上先卖出腾出名额，再用释放的资金买入。"""
    ctx = setup_ctx({STRONG: cross_up_series(40),
                               WEAK: cross_down_series(40)},
                    [STRONG])
    ctx.my_holdings[WEAK] = 1000
    qmt.positions = [FakePosition(WEAK, volume=1000, can_use=1000)]

    strategy.handlebar(ctx)

    assert [o['opType'] for o in qmt.orders] == [strategy.ORDER_OP_SELL,
                                                 strategy.ORDER_OP_BUY]


# ======================== 净值记录与告警 ========================
def test_records_daily_nav(strategy, qmt):
    ctx = setup_ctx({}, [])
    qmt.total_asset = 1_234_567.0

    strategy.handlebar(ctx)

    assert ctx.nav_list == [('2024-01-01', 1_234_567.0)]


def test_nav_uses_bar_date(strategy, qmt):
    ctx = setup_ctx({}, [])
    ctx.barpos = 3
    strategy.handlebar(ctx)
    assert ctx.nav_list[0][0] == '2024-01-04'


def test_warns_once_when_account_asset_is_zero(strategy, qmt, capsys):
    ctx = setup_ctx({}, [])
    qmt.total_asset = 0.0

    strategy.handlebar(ctx)
    strategy.handlebar(ctx)

    assert ctx.warned_no_asset is True
    # 告警带 [handlebar] 前缀，用 ASCII 片段计数以免依赖中文
    assert capsys.readouterr().out.count('[handlebar]') == 1


def test_keeps_previous_asset_when_end_of_bar_query_fails(strategy, qmt, monkeypatch):
    """收盘后再次取资产失败时，用本 bar 开头的资产值兜底。"""
    ctx = setup_ctx({}, [])
    qmt.total_asset = 100_000.0
    calls = {'n': 0}

    def flaky(account, kind, field):
        if field == 'ACCOUNT':
            calls['n'] += 1
            if calls['n'] > 1:            # bar 末尾那次查询返回空
                return []
        return qmt.get_trade_detail_data(account, kind, field)

    monkeypatch.setattr(strategy, 'get_trade_detail_data', flaky)

    strategy.handlebar(ctx)

    assert ctx.nav_list[0][1] == pytest.approx(100_000.0)
