# -*- coding: utf-8 -*-
"""取数、账户/持仓解析、下单封装等辅助函数的测试。"""

from types import SimpleNamespace

import pytest

from conftest import (BUY, K_CODE, K_DIR, K_PRICE, K_VOL, SELL, UNKNOWN,
                      FakeContextInfo, FakePosition, LegacyContextInfo, make_ctx)


# ======================== _date_str ========================
def test_date_str_formats_bar_timetag(strategy, qmt):
    ctx = FakeContextInfo(start_date='2024-03-05')
    ctx.barpos = 2
    assert strategy._date_str(ctx) == '2024-03-07'


def test_date_str_falls_back_to_barpos_when_timetostr_fails(strategy, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError('模拟 timetostr 不可用')

    monkeypatch.setattr(strategy, 'timetostr', boom, raising=False)
    ctx = FakeContextInfo()
    ctx.barpos = 7
    assert strategy._date_str(ctx) == '7'


# ======================== _get_close_history ========================
def test_get_close_history_reads_four_arg_signature(strategy, qmt):
    ctx = FakeContextInfo(history={'600519.SH': [1.0, 2.0, 3.0]})
    assert strategy._get_close_history(ctx, '600519.SH', 2) == [2.0, 3.0]


def test_get_close_history_falls_back_to_legacy_signature(strategy, qmt):
    """旧版 QMT 的 get_history_data 只接受 3 个参数，应能自动降级。"""
    ctx = LegacyContextInfo(history={'600519.SH': [1.0, 2.0, 3.0]})
    assert strategy._get_close_history(ctx, '600519.SH', 3) == [1.0, 2.0, 3.0]


def test_get_close_history_returns_none_for_unknown_stock(strategy, qmt):
    ctx = FakeContextInfo(history={})
    assert strategy._get_close_history(ctx, '000001.SZ', 10) is None


def test_get_close_history_drops_nan_values(strategy, qmt):
    ctx = FakeContextInfo(history={'600519.SH': [10.0, float('nan'), 12.0]})
    assert strategy._get_close_history(ctx, '600519.SH', 3) == [10.0, 12.0]


def test_get_close_history_returns_none_on_non_numeric_data(strategy, qmt):
    ctx = FakeContextInfo(history={'600519.SH': [10.0, None]})
    assert strategy._get_close_history(ctx, '600519.SH', 2) is None


def test_get_close_history_returns_none_when_api_raises(strategy, qmt):
    class BrokenContext(FakeContextInfo):
        def get_history_data(self, *args, **kwargs):
            raise RuntimeError('模拟行情接口异常')

    assert strategy._get_close_history(BrokenContext(), '600519.SH', 10) is None


# ======================== 账户 / 持仓解析 ========================
def test_get_total_asset_and_cash(strategy, qmt):
    qmt.total_asset, qmt.cash = 888_888.0, 66_666.0
    assert strategy._get_total_asset('acc') == pytest.approx(888_888.0)
    assert strategy._get_cash('acc') == pytest.approx(66_666.0)


def test_get_total_asset_returns_zero_when_account_missing(strategy, monkeypatch):
    monkeypatch.setattr(strategy, 'get_trade_detail_data',
                        lambda *a, **k: [], raising=False)
    assert strategy._get_total_asset('acc') == 0.0
    assert strategy._get_cash('acc') == 0.0


def test_get_positions_builds_code_keyed_dict(strategy, qmt):
    qmt.positions = [FakePosition('600519.SH'), FakePosition('000858.SZ')]
    positions = strategy._get_positions('acc')
    assert set(positions) == {'600519.SH', '000858.SZ'}
    assert positions['600519.SH'].m_strInstrumentID == '600519.SH'


def test_get_positions_skips_entries_without_code(strategy, qmt):
    qmt.positions = [FakePosition(''), FakePosition('600519.SH')]
    assert set(strategy._get_positions('acc')) == {'600519.SH'}


def test_get_positions_returns_empty_dict_on_error(strategy, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError('模拟持仓接口异常')

    monkeypatch.setattr(strategy, 'get_trade_detail_data', boom, raising=False)
    assert strategy._get_positions('acc') == {}


# ======================== _profit_rate ========================
def test_profit_rate_is_zero_without_position(strategy):
    assert strategy._profit_rate(None, 10.0) == 0.0


def test_profit_rate_prefers_broker_reported_rate(strategy):
    pos = FakePosition('600519.SH', profit_rate=0.123)
    assert strategy._profit_rate(pos, 10.0) == pytest.approx(0.123)


def test_profit_rate_falls_back_to_cost_price(strategy):
    pos = FakePosition('600519.SH', cost_price=10.0)
    assert strategy._profit_rate(pos, 12.0) == pytest.approx(0.2)


def test_profit_rate_falls_back_when_broker_rate_is_not_numeric(strategy):
    """m_dProfitRate 取到非数值时应改用成本价估算，而不是抛异常。"""
    pos = FakePosition('600519.SH', cost_price=10.0)
    pos.m_dProfitRate = 'N/A'
    assert strategy._profit_rate(pos, 12.0) == pytest.approx(0.2)


def test_profit_rate_is_zero_when_cost_price_is_zero(strategy):
    pos = FakePosition('600519.SH', cost_price=0.0)
    assert strategy._profit_rate(pos, 12.0) == 0.0


# ======================== _usable_volume ========================
def test_usable_volume_uses_broker_available_volume(strategy):
    """A 股 T+1：当日买入不可卖，必须听券商的 m_nCanUseVolume。"""
    pos = FakePosition('600519.SH', volume=1000, can_use=600)
    assert strategy._usable_volume(pos, 1000) == 600


def test_usable_volume_is_zero_when_broker_says_locked(strategy):
    pos = FakePosition('600519.SH', volume=1000, can_use=0)
    assert strategy._usable_volume(pos, 1000) == 0


def test_usable_volume_falls_back_to_local_holdings_without_position(strategy):
    assert strategy._usable_volume(None, 500) == 500


def test_usable_volume_is_zero_when_position_lacks_field(strategy):
    pos = SimpleNamespace(m_strInstrumentID='600519.SH')
    assert strategy._usable_volume(pos, 500) == 0


def test_usable_volume_is_zero_when_nothing_held(strategy):
    assert strategy._usable_volume(None, None) == 0


# ======================== _deal_direction ========================
@pytest.mark.parametrize('raw', [0, 48, 'B'])
def test_deal_direction_maps_buy_codes(strategy, raw):
    assert strategy._deal_direction(SimpleNamespace(m_nDirection=raw)) == BUY


@pytest.mark.parametrize('raw', [1, 49, 'S'])
def test_deal_direction_maps_sell_codes(strategy, raw):
    assert strategy._deal_direction(SimpleNamespace(m_nDirection=raw)) == SELL


@pytest.mark.parametrize('text,expected', [
    ('买入', BUY), ('证券买入', BUY),
    ('卖出', SELL), ('证券卖出', SELL),
])
def test_deal_direction_falls_back_to_text_field(strategy, text, expected):
    deal = SimpleNamespace(m_nDirection=99, m_strDirection=text)
    assert strategy._deal_direction(deal) == expected


def test_deal_direction_is_unknown_for_unrecognised_values(strategy):
    deal = SimpleNamespace(m_nDirection=99, m_strDirection='其他')
    assert strategy._deal_direction(deal) == UNKNOWN


# ======================== _buy / _sell ========================
def test_buy_places_limit_order_and_records_holding(strategy, qmt):
    ctx = make_ctx(accountid='acc')
    strategy._buy(ctx, '600519.SH', 10.5, 1000)

    assert len(qmt.orders) == 1
    order = qmt.orders[0]
    assert order['opType'] == strategy.ORDER_OP_BUY
    assert order['orderType'] == strategy.ORDER_TYPE_LIMIT
    assert order['prType'] == strategy.PR_TYPE_LIMIT
    assert order['stock'] == '600519.SH'
    assert order['price'] == pytest.approx(10.5)
    assert order['volume'] == 1000
    assert order['strategyName'] == strategy.STRATEGY_NAME

    assert ctx.my_holdings['600519.SH'] == 1000
    assert len(ctx.trade_log) == 1
    assert ctx.trade_log[0][K_CODE] == '600519.SH'
    assert ctx.trade_log[0][K_DIR] == BUY
    assert ctx.trade_log[0][K_PRICE] == pytest.approx(10.5)
    assert ctx.trade_log[0][K_VOL] == 1000


def test_buy_accumulates_existing_holding(strategy, qmt):
    ctx = make_ctx(accountid='acc')
    ctx.my_holdings['600519.SH'] = 500
    strategy._buy(ctx, '600519.SH', 10.0, 300)
    assert ctx.my_holdings['600519.SH'] == 800


@pytest.mark.parametrize('volume', [0, -100])
def test_buy_ignores_non_positive_volume(strategy, qmt, volume):
    ctx = make_ctx(accountid='acc')
    strategy._buy(ctx, '600519.SH', 10.0, volume)
    assert qmt.orders == []
    assert ctx.my_holdings == {}
    assert ctx.trade_log == []


def test_buy_swallows_order_exception(strategy, qmt):
    """下单接口异常不应打断 handlebar，也不应污染本地持仓。"""
    ctx = make_ctx(accountid='acc')
    qmt.raise_on_order = True
    strategy._buy(ctx, '600519.SH', 10.0, 1000)
    assert ctx.my_holdings == {}
    assert ctx.trade_log == []


def test_sell_places_order_and_reduces_holding(strategy, qmt):
    ctx = make_ctx(accountid='acc')
    ctx.my_holdings['600519.SH'] = 1000
    strategy._sell(ctx, '600519.SH', 11.0, 400)

    assert qmt.orders[0]['opType'] == strategy.ORDER_OP_SELL
    assert qmt.orders[0]['volume'] == 400
    assert ctx.my_holdings['600519.SH'] == 600
    assert ctx.trade_log[0][K_DIR] == SELL


def test_sell_clears_state_on_full_exit(strategy, qmt):
    ctx = make_ctx(accountid='acc')
    ctx.my_holdings['600519.SH'] = 1000
    ctx.peak_profit['600519.SH'] = 0.2

    strategy._sell(ctx, '600519.SH', 11.0, 1000)

    assert '600519.SH' not in ctx.my_holdings
    assert '600519.SH' not in ctx.peak_profit


def test_sell_keeps_peak_profit_on_partial_exit(strategy, qmt):
    ctx = make_ctx(accountid='acc')
    ctx.my_holdings['600519.SH'] = 1000
    ctx.peak_profit['600519.SH'] = 0.2

    strategy._sell(ctx, '600519.SH', 11.0, 400)

    assert ctx.peak_profit['600519.SH'] == pytest.approx(0.2)


def test_sell_clamps_at_zero_when_overselling(strategy, qmt):
    ctx = make_ctx(accountid='acc')
    ctx.my_holdings['600519.SH'] = 100
    strategy._sell(ctx, '600519.SH', 11.0, 500)
    assert '600519.SH' not in ctx.my_holdings


def test_sell_ignores_non_positive_volume(strategy, qmt):
    ctx = make_ctx(accountid='acc')
    ctx.my_holdings['600519.SH'] = 1000
    strategy._sell(ctx, '600519.SH', 11.0, 0)
    assert qmt.orders == []
    assert ctx.my_holdings['600519.SH'] == 1000


def test_sell_swallows_order_exception(strategy, qmt):
    ctx = make_ctx(accountid='acc')
    ctx.my_holdings['600519.SH'] = 1000
    qmt.raise_on_order = True
    strategy._sell(ctx, '600519.SH', 11.0, 1000)
    assert ctx.my_holdings['600519.SH'] == 1000
