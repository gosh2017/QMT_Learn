# -*- coding: gbk -*-
"""
================================================================================
策略名称：MA20 + MACD 双信号动量策略（日线）
运行环境：迅投QMT（大QMT）〖模型研究〗内置 Python 环境
--------------------------------------------------------------------------------
策略逻辑：
    标的        ：沪深300成分股（板块 '沪深300'）
    买入信号    ：收盘价上穿20日均线，且 MACD 柱状线（hist）> 0
    卖出信号    ：收盘价下穿20日均线，或 持仓盈利回撤超过 5%
    仓位        ：每只股票等权分配（总资产 / 最多持仓数），最多同时持有 5 只
    调仓频率    ：每日 K 线触发一次（handlebar）
    回测报告    ：stop() 中自动导出成交明细、净值序列、Markdown 报告到 G:/QMT_Reports/

回调结构（QMT 标准）：
    def init(ContextInfo)      —— 初始化：股票池、参数、运行时状态
    def handlebar(ContextInfo) —— 每根 K 线触发一次：计算信号、下单、记录净值
    def stop(ContextInfo)      —— 回测结束：收集成交、计算绩效、导出报告（异常全部捕获）

说明：本文件为 GBK 编码（QMT 模型研究环境要求，中文注释原样保留）。
      本项目已在 .vscode/settings.json 中为 Python 文件配置 GBK，
      VSCode 打开/保存本文件会自动使用 GBK，无需手动转换。
================================================================================
"""

# ======================== 导入依赖 ========================
import os
import math
import numpy as np
import pandas as pd
from collections import deque, defaultdict

# ======================== 策略参数（可自行调整） ========================
ACCOUNT_ID        = ''           # TODO：填写你的资金账号；留空则使用 QMT 回测参数中配置的账号
SECTOR_NAME       = '沪深300'    # 股票池板块名称（沪深300 / 上证50 / 中证500 等）
MA_PERIOD         = 20           # 均线周期
MACD_FAST         = 12           # MACD 快线周期
MACD_SLOW         = 26           # MACD 慢线周期
MACD_SIGNAL       = 9            # MACD 信号线周期
MAX_HOLDINGS      = 5            # 最多同时持有的股票数量
PROFIT_STOP_RATIO = 0.05         # 盈利回撤止盈阈值（5%）
BAR_BUFFER        = 120          # 拉取的历史 K 线根数（需覆盖 MACD 暖机：26+9 即可，留余量）
REPORT_DIR        = 'G:/QMT_Reports/'   # 报告导出目录（不存在则自动创建）

# ---- passorder 委托类型常量（A股限价）----
# 注意：不同 QMT 版本 orderType 取值可能不同（常见 0 或 1），如回测无成交请先核对此项。
ORDER_OP_BUY      = 0            # opType: 买入
ORDER_OP_SELL     = 1            # opType: 卖出
ORDER_TYPE_LIMIT  = 1            # orderType: 限价（若不成交可改为 0 再试）
PR_TYPE_LIMIT     = 0            # prType: 限价（按 price 参数定价）
QUICK_TRADE       = 1            # quickTrade: 1=启用快速成交
STRATEGY_NAME     = 'MA_MACD'    # 策略名（标记订单）


# ======================== 技术指标计算（纯 numpy，避免外部依赖） ========================
def _ema(values, period):
    """计算指数移动平均 EMA，返回与输入等长的数组。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def calc_macd(closes, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """
    根据收盘价序列计算 MACD。
    返回最新一根 K 线的 (dif, dea, hist)：
        dif  = EMA(close, fast) - EMA(close, slow)
        dea  = EMA(dif, signal)
        hist = dif - dea   （即 MACD 柱状线）
    数据不足时返回 (None, None, None)。
    """
    closes = np.asarray(closes, dtype=float)
    if closes.size < slow:
        return None, None, None
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    dif = ema_fast - ema_slow
    dea = _ema(dif, signal)
    hist = dif - dea
    return float(dif[-1]), float(dea[-1]), float(hist[-1])


# ======================== 行情 / 账户 / 持仓取数封装 ========================
def _date_str(ContextInfo):
    """取当前 K 线日期字符串；timetostr 是 QMT 内置函数。"""
    try:
        return timetostr(ContextInfo.get_bar_timetag(ContextInfo.barpos), '%Y-%m-%d')
    except Exception:
        return str(ContextInfo.barpos)


def _get_close_history(ContextInfo, stock, count):
    """
    取最近 count 根日线收盘价，返回 float 列表；失败或数据不足返回 None。
    兼容不同 QMT 版本 get_history_data 的参数顺序：
        形式1: get_history_data(stock, field, period=-1, count)
        形式2: get_history_data(stock, field, count)
    """
    data = None
    try:
        # period=-1 表示沿用策略当前周期（日线回测即取日线）
        data = ContextInfo.get_history_data(stock, 'close', -1, count)
    except TypeError:
        try:
            data = ContextInfo.get_history_data(stock, 'close', count)
        except Exception:
            return None
    except Exception:
        return None
    if data is None:
        return None
    try:
        arr = [float(x) for x in data]          # 若 data 为标量会抛 TypeError
    except TypeError:
        return None
    # 剔除 None / NaN（沪深300流动性好，通常无缺口）
    arr = [x for x in arr if not (x is None or (isinstance(x, float) and math.isnan(x)))]
    return arr


def _get_account_obj(account):
    """取账户对象（get_trade_detail_data 的 'ACCOUNT' 项）。"""
    try:
        acc = get_trade_detail_data(account, 'STOCK', 'ACCOUNT')
        if acc:
            return acc[0]
    except Exception:
        pass
    return None


def _get_total_asset(account):
    """账户总资产。"""
    obj = _get_account_obj(account)
    if obj is not None:
        try:
            return float(getattr(obj, 'm_dTotalAsset', 0.0) or 0.0)
        except Exception:
            return 0.0
    return 0.0


def _get_cash(account):
    """账户可用资金。"""
    obj = _get_account_obj(account)
    if obj is not None:
        try:
            return float(getattr(obj, 'm_dBalance', 0.0) or 0.0)
        except Exception:
            return 0.0
    return 0.0


def _get_positions(account):
    """取所有持仓，返回 {股票代码: 持仓对象}。"""
    res = {}
    try:
        pos_list = get_trade_detail_data(account, 'STOCK', 'POSITION')
        for p in (pos_list or []):
            code = getattr(p, 'm_strInstrumentID', '')
            if code:
                res[code] = p
    except Exception:
        pass
    return res


def _profit_rate(pos, close):
    """单只持仓的浮盈比例。优先用券商返回的 m_dProfitRate，否则用成本价估算。"""
    if pos is None:
        return 0.0
    rate = getattr(pos, 'm_dProfitRate', None)
    try:
        if rate is not None:
            return float(rate)
    except Exception:
        pass
    cost = getattr(pos, 'm_dCostPrice', None)
    if cost:
        try:
            return float(close) / float(cost) - 1.0
        except Exception:
            return 0.0
    return 0.0


def _usable_volume(pos, my_hold_vol):
    """可卖数量：优先券商 m_nCanUseVolume（已考虑 A 股 T+1），其次用自记持仓。"""
    if pos is not None:
        # 券商返回的可卖数量已考虑 A 股 T+1，直接采用（可能为 0）；只有取不到持仓时才退回自记持仓
        try:
            return int(getattr(pos, 'm_nCanUseVolume', 0) or 0)
        except Exception:
            return 0
    return int(my_hold_vol or 0)


# ======================== 下单封装（passorder） ========================
def _buy(ContextInfo, stock, price, volume):
    """限价买入：opType=0, orderType=限价, prType=限价, price=当前收盘价。"""
    if volume <= 0:
        return
    try:
        passorder(ORDER_OP_BUY, ORDER_TYPE_LIMIT, ContextInfo.accountid, stock,
                  PR_TYPE_LIMIT, float(price), int(volume),
                  STRATEGY_NAME, QUICK_TRADE, 0, ContextInfo)
        # 同步更新自记持仓与成交日志（兜底，便于报告生成）
        ContextInfo.my_holdings[stock] = ContextInfo.my_holdings.get(stock, 0) + int(volume)
        ContextInfo.trade_log.append({'股票代码': stock, '方向': '买入',
                                      '成交时间': _date_str(ContextInfo),
                                      '成交价格': float(price), '成交量': int(volume)})
    except Exception as e:
        print('[买入异常]', stock, e)


def _sell(ContextInfo, stock, price, volume):
    """限价卖出：opType=1。"""
    if volume <= 0:
        return
    try:
        passorder(ORDER_OP_SELL, ORDER_TYPE_LIMIT, ContextInfo.accountid, stock,
                  PR_TYPE_LIMIT, float(price), int(volume),
                  STRATEGY_NAME, QUICK_TRADE, 0, ContextInfo)
        prev = ContextInfo.my_holdings.get(stock, 0)
        ContextInfo.my_holdings[stock] = max(0, prev - int(volume))
        if ContextInfo.my_holdings[stock] == 0:
            ContextInfo.my_holdings.pop(stock, None)
            ContextInfo.peak_profit.pop(stock, None)
        ContextInfo.trade_log.append({'股票代码': stock, '方向': '卖出',
                                      '成交时间': _date_str(ContextInfo),
                                      '成交价格': float(price), '成交量': int(volume)})
    except Exception as e:
        print('[卖出异常]', stock, e)


# ======================== 初始化 ========================
def init(ContextInfo):
    # 1. 账号：优先用代码顶部 ACCOUNT_ID；否则取 QMT 回测参数中注入的账号；
    #    两者都没有时给兜底占位（部分 QMT 版本回测不自动注入 accountid，直接读会抛 AttributeError）
    if ACCOUNT_ID:
        ContextInfo.accountid = ACCOUNT_ID
    elif not getattr(ContextInfo, 'accountid', None):
        ContextInfo.accountid = 'QMT_BACKTEST'
        print('[init] 警告：未检测到资金账号，已使用占位账号 QMT_BACKTEST。'
              '如回测无成交，请在回测参数中选择账号，或在代码顶部 ACCOUNT_ID 填入。')
    account = ContextInfo.accountid

    # 2. 股票池：优先全局函数，其次 ContextInfo 方法，最后兜底代表券
    stock_list = []
    try:
        stock_list = get_stock_list_in_sector(SECTOR_NAME) or []
    except Exception:
        pass
    if not stock_list:
        try:
            stock_list = ContextInfo.get_stock_list_in_sector(SECTOR_NAME) or []
        except Exception:
            stock_list = []
    if not stock_list:
        # 兜底：若板块接口未返回数据，用几只沪深300代表券保证策略可跑通
        stock_list = ['600519.SH', '000858.SZ', '600036.SH',
                      '000333.SZ', '601318.SH', '600276.SH', '000651.SZ']
    # 过滤已退市/无效代码：600837.SH 海通证券已被国泰君安吸收合并、2025 年退市，
    # QMT 板块快照可能残留该代码，set_universe 会告警，故剔除。
    _invalid_codes = {'600837.SH'}
    stock_list = [c for c in stock_list if c not in _invalid_codes]
    ContextInfo.stock_list = stock_list
    try:
        ContextInfo.set_universe(stock_list)      # 订阅行情，确保 get_history_data 有数据
    except Exception:
        pass

    # 3. 运行时状态
    ContextInfo.nav_list      = []                 # 每日净值：[(日期, 总资产), ...]
    ContextInfo.my_holdings   = {}                 # 自记持仓 {代码: 数量}（下单即时更新）
    ContextInfo.peak_profit   = {}                 # 每只持仓的历史最高浮盈比例，用于回撤止盈
    ContextInfo.trade_log     = []                 # 自记成交（DEAL 取数失败时的兜底）
    ContextInfo.warned_no_asset = False          # 是否已告警：账户资产取不到

    print('[init] 资金账号:', account, '| 股票池规模:', len(stock_list))
    print('[init] 参数: MA=%d, MACD=%d/%d/%d, 最多持仓=%d, 盈利回撤止盈=%.1f%%'
          % (MA_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, MAX_HOLDINGS, PROFIT_STOP_RATIO * 100))


# ======================== 每根 K 线触发 ========================
def handlebar(ContextInfo):
    account = ContextInfo.accountid
    date_str = _date_str(ContextInfo)

    # 取账户总资产（用于等权仓位计算与净值记录）
    total_asset = _get_total_asset(account)

    # 账户资产取不到时告警一次，避免策略静默零成交
    if total_asset <= 0 and not getattr(ContextInfo, 'warned_no_asset', False):
        ContextInfo.warned_no_asset = True
        print('[handlebar] 警告：账户总资产为 0，get_trade_detail_data 可能取不到账户数据，'
              '策略将无法买入。请检查回测资金账号，或在代码顶部 ACCOUNT_ID 填入。')

    # ---------- 1. 卖出判断（先卖后买，释放资金） ----------
    broker_pos = _get_positions(account)
    for stock in list(ContextInfo.my_holdings.keys()):
        closes = _get_close_history(ContextInfo, stock, BAR_BUFFER)
        if not closes or len(closes) < MA_PERIOD + 2:
            continue
        close     = closes[-1]
        prev_close = closes[-2]
        ma        = float(np.mean(closes[-MA_PERIOD:]))            # 含当日
        prev_ma   = float(np.mean(closes[-MA_PERIOD - 1:-1]))      # 不含当日

        # 收盘下穿20日均线：前日收盘 > 前日MA，且当日收盘 < 当日MA
        cross_down = (prev_close > prev_ma) and (close < ma)

        # 盈利回撤止盈：从历史最高浮盈回撤超过阈值且仍盈利
        pos         = broker_pos.get(stock)
        profit_rate = _profit_rate(pos, close)
        peak        = max(ContextInfo.peak_profit.get(stock, 0.0), profit_rate)
        ContextInfo.peak_profit[stock] = peak
        drawdown    = peak - profit_rate     # 从峰值回撤的幅度

        sell_reason = None
        if cross_down:
            sell_reason = '收盘下穿20日均线'
        elif peak > 0 and drawdown >= PROFIT_STOP_RATIO and profit_rate > 0:
            sell_reason = '盈利回撤%.1f%%止盈' % (drawdown * 100)

        if sell_reason:
            vol = _usable_volume(pos, ContextInfo.my_holdings.get(stock, 0))
            if vol > 0:
                _sell(ContextInfo, stock, close, vol)

    # ---------- 2. 买入判断 ----------
    n_holding       = len(ContextInfo.my_holdings)
    available_slots = MAX_HOLDINGS - n_holding
    candidates      = []
    if available_slots > 0:
        for stock in ContextInfo.stock_list:
            if stock in ContextInfo.my_holdings:
                continue
            closes = _get_close_history(ContextInfo, stock, BAR_BUFFER)
            if not closes or len(closes) < max(MA_PERIOD + 2, MACD_SLOW + MACD_SIGNAL + 5):
                continue
            close      = closes[-1]
            prev_close = closes[-2]
            ma         = float(np.mean(closes[-MA_PERIOD:]))
            prev_ma    = float(np.mean(closes[-MA_PERIOD - 1:-1]))

            # 收盘上穿20日均线：前日收盘 < 前日MA，且当日收盘 > 当日MA
            cross_up = (prev_close < prev_ma) and (close > ma)
            _, _, hist = calc_macd(closes)
            if cross_up and hist is not None and hist > 0:
                candidates.append((stock, close, hist))   # hist 越大动量越强
        # 按 MACD 柱状线降序，优先选动量更强的标的
        candidates.sort(key=lambda x: x[2], reverse=True)

        cash = _get_cash(account)
        for stock, close, _ in candidates[:available_slots]:
            # 等权：单只目标市值 = 总资产 / 最多持仓数；同时不超过可用资金的 98%
            target_value = total_asset / MAX_HOLDINGS
            use_value    = min(target_value, cash * 0.98)
            shares       = int(use_value / close / 100) * 100   # A 股按 100 股整手
            if shares >= 100:
                _buy(ContextInfo, stock, close, shares)
                cash -= use_value    # 预扣，避免单 bar 超额下单

    # ---------- 3. 记录每日净值（取本 bar 结束后的总资产） ----------
    end_asset = _get_total_asset(account)
    ContextInfo.nav_list.append((date_str, end_asset if end_asset > 0 else total_asset))


# ======================== 回测结束：导出报告 ========================
def stop(ContextInfo):
    print('[stop] 开始收集回测结果并导出报告 ...')
    try:
        _export_reports(ContextInfo)
        print('[stop] 报告已导出至:', REPORT_DIR)
    except Exception as e:
        # 关键：stop() 中的异常必须被捕获，避免导出失败导致策略报错
        print('[stop] 导出报告失败（已被捕获，不影响策略）:', e)
        import traceback
        traceback.print_exc()


# ======================== 报告生成内部函数 ========================
def _collect_trades(ContextInfo, account):
    """收集每笔成交明细；优先券商 DEAL 记录，失败则用自记日志兜底。"""
    rows = []
    try:
        deals = get_trade_detail_data(account, 'STOCK', 'DEAL')
        for d in (deals or []):
            code   = getattr(d, 'm_strInstrumentID', '')
            direct = _deal_direction(d)
            price  = float(getattr(d, 'm_dPrice', 0) or 0)
            volume = int(getattr(d, 'm_nVolume', 0) or 0)
            t      = getattr(d, 'm_strTradeTime', '') or getattr(d, 'm_strOpDate', '') or ''
            if volume > 0:
                rows.append([code, direct, str(t), price, volume])
    except Exception as e:
        print('[trades] 获取 DEAL 失败，改用自记日志:', e)

    if not rows:
        # 兜底：使用 handlebar 中自记录的成交
        for t in ContextInfo.trade_log:
            rows.append([t.get('股票代码', ''), t.get('方向', ''),
                         t.get('成交时间', ''), t.get('成交价格', 0), t.get('成交量', 0)])
    return rows


def _deal_direction(d):
    """把成交记录的方向字段统一成 '买入'/'卖出'（兼容 0/1 与 48/49 两种取值）。"""
    v = getattr(d, 'm_nDirection', None)
    if v in (0, 48, 'B'):
        return '买入'
    if v in (1, 49, 'S'):
        return '卖出'
    s = getattr(d, 'm_strDirection', '')
    if '买' in str(s):
        return '买入'
    if '卖' in str(s):
        return '卖出'
    return '未知'


def _calc_metrics(nav_list, trades):
    """基于净值序列与成交明细计算绩效指标。"""
    m = {
        '总交易日': len(nav_list),
        '期初资产': 0.0, '期末资产': 0.0,
        '总收益率(%)': 0.0, '年化收益率(%)': 0.0,
        '最大回撤(%)': 0.0, '夏普比率': 0.0,
        '胜率(%)': 0.0, '总交易次数': len(trades),
    }
    if not nav_list:
        return m

    assets = [float(x[1]) for x in nav_list if float(x[1]) > 0]
    if not assets:
        return m
    start, end = assets[0], assets[-1]
    m['期初资产'] = round(start, 2)
    m['期末资产'] = round(end, 2)

    # 总收益 / 年化收益（按 252 交易日年化，几何）
    if start > 0:
        m['总收益率(%)'] = round((end / start - 1.0) * 100, 2)
        n = len(assets) - 1
        if n > 0 and end > 0 and start > 0:
            m['年化收益率(%)'] = round(((end / start) ** (252.0 / n) - 1.0) * 100, 2)

    # 最大回撤
    peak, max_dd = assets[0], 0.0
    for v in assets:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    m['最大回撤(%)'] = round(max_dd * 100, 2)

    # 夏普比率（日收益均值 / 标准差 * sqrt(252)，无风险利率取 0）
    arr = np.asarray(assets, dtype=float)
    if arr.size > 1:
        rets = np.diff(arr) / arr[:-1]
        std = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
        m['夏普比率'] = round(float(rets.mean()) / std * math.sqrt(252), 3) if std > 0 else 0.0

    # 胜率（按每只股票 FIFO 匹配买卖，计算盈利平仓笔数占比）
    m['胜率(%)'] = round(_win_rate(trades) * 100, 2)
    return m


def _win_rate(trades):
    """按股票代码分组、时间排序，FIFO 匹配买入→卖出，统计盈利平仓占比。"""
    by_code = defaultdict(list)
    for code, direction, t, price, volume in trades:
        by_code[code].append((str(t), direction, float(price), int(volume)))
    wins, total = 0, 0
    for code, lst in by_code.items():
        lst.sort(key=lambda x: x[0])
        q = deque()                       # 队列元素: [买入价, 剩余数量]
        for _, direction, price, volume in lst:
            if direction == '买入':
                q.append([price, volume])
            elif direction == '卖出':
                remaining = volume
                while remaining > 0 and q:
                    buy_price, buy_vol = q[0]
                    matched = min(buy_vol, remaining)
                    if price > buy_price:     # 卖出价高于买入价 → 盈利
                        wins += 1
                    total += 1
                    remaining -= matched
                    buy_vol -= matched
                    if buy_vol <= 0:
                        q.popleft()
                    else:
                        q[0][1] = buy_vol
    return wins / total if total > 0 else 0.0


def _md_table(rows):
    """把二维列表渲染成 Markdown 表格（不依赖 tabulate）。"""
    if not rows:
        return ''
    header = '| ' + ' | '.join(str(c) for c in rows[0]) + ' |'
    sep    = '| ' + ' | '.join('---' for _ in rows[0]) + ' |'
    body   = ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows[1:]]
    return '\n'.join([header, sep] + body)


def _df_to_md_table(df):
    """DataFrame -> Markdown 表格（手动版，to_markdown 不可用时兜底）。"""
    rows = [list(df.columns)] + df.astype(str).values.tolist()
    return _md_table(rows)


def _build_markdown(ContextInfo, metrics, trades_df, nav_df):
    """组装 backtest_report.md 的完整内容。"""
    L = []
    L.append('# MA20 + MACD 双信号策略 · 回测报告')
    L.append('')
    start_date = nav_df['日期'].iloc[0] if (not nav_df.empty and '日期' in nav_df.columns) else 'N/A'
    end_date   = nav_df['日期'].iloc[-1] if (not nav_df.empty and '日期' in nav_df.columns) else 'N/A'
    L.append('**策略名称：** MA20 + MACD 双均线动量策略（日线）  ')
    L.append('**回测时间段：** %s ~ %s  ' % (start_date, end_date))
    L.append('**标的池：** %s 成分股，等权配置，最多持有 %d 只  ' % (SECTOR_NAME, MAX_HOLDINGS))
    L.append('**调仓频率：** 每日收盘前检查信号  ')
    L.append('')

    # ---- 核心绩效指标表 ----
    L.append('## 一、核心绩效指标')
    metric_rows = [
        ['指标', '数值'],
        ['总交易日', metrics['总交易日']],
        ['期初资产', round(metrics['期初资产'], 2)],
        ['期末资产', round(metrics['期末资产'], 2)],
        ['总收益率(%)', metrics['总收益率(%)']],
        ['年化收益率(%)', metrics['年化收益率(%)']],
        ['最大回撤(%)', metrics['最大回撤(%)']],
        ['夏普比率', metrics['夏普比率']],
        ['胜率(%)', metrics['胜率(%)']],
        ['总交易次数', metrics['总交易次数']],
    ]
    L.append(_md_table(metric_rows))
    L.append('')

    # ---- 收益曲线描述 ----
    L.append('## 二、收益曲线描述')
    if not nav_df.empty:
        a0   = float(nav_df['总资产'].iloc[0])
        a1   = float(nav_df['总资产'].iloc[-1])
        amax = float(nav_df['总资产'].max())
        amin = float(nav_df['总资产'].min())
        trend = '上行' if a1 >= a0 else '回落'
        L.append('账户净值从期初 **%.2f** 起步，期间最高触及 **%.2f**、最低探至 **%.2f**，'
                 '期末收于 **%.2f**，整体呈 **%s** 走势。' % (a0, amax, amin, a1, trend))
    else:
        L.append('（无净值数据）')
    L.append('')

    # ---- 前 10 笔成交明细 ----
    L.append('## 三、前 10 笔成交明细')
    if trades_df.empty:
        L.append('（无成交记录）')
    else:
        top = trades_df.head(10)
        try:
            # 使用 pandas 的 to_markdown（需 tabulate）；失败则用手动兜底
            L.append(top.to_markdown(index=False))
        except Exception:
            L.append(_df_to_md_table(top))
    L.append('')
    L.append('> 本报告由 QMT 策略 `stop()` 自动生成，导出路径：`%s`' % REPORT_DIR)
    return '\n'.join(L)


def _export_reports(ContextInfo):
    """导出三种格式：成交明细 CSV、净值 CSV、绩效 Markdown 报告。"""
    account = ContextInfo.accountid

    # 0. 创建导出目录
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    # 1. 成交明细 CSV
    trades = _collect_trades(ContextInfo, account)
    trades_df = pd.DataFrame(trades, columns=['股票代码', '方向', '成交时间', '成交价格', '成交量'])
    trades_df.to_csv(os.path.join(REPORT_DIR, 'backtest_trades.csv'),
                     index=False, encoding='utf-8-sig')   # utf-8-sig 便于 Excel 正确显示中文

    # 2. 每日净值 CSV
    nav_df = pd.DataFrame(ContextInfo.nav_list, columns=['日期', '总资产'])
    if not nav_df.empty:
        nav_df['日收益率'] = nav_df['总资产'].pct_change()
    nav_df.to_csv(os.path.join(REPORT_DIR, 'backtest_nav.csv'),
                  index=False, encoding='utf-8-sig')

    # 3. 绩效指标
    metrics = _calc_metrics(ContextInfo.nav_list, trades)

    # 4. Markdown 报告
    md_text = _build_markdown(ContextInfo, metrics, trades_df, nav_df)
    with open(os.path.join(REPORT_DIR, 'backtest_report.md'), 'w', encoding='utf-8') as f:
        f.write(md_text)

    # 控制台简要打印
    print('[stop] 绩效汇总：', metrics)


# =================================================================================
# 【使用说明】（代码结束后请阅读）
# --------------------------------------------------------------------------------
# 一、回测前准备（重要）：
#   1. 在 QMT 客户端打开〖数据管理〗→〖行情数据下载〗，下载：
#        - 品种：沪深300 成分股（可勾选“沪深300”板块一次性下载）；
#        - 周期：日线；
#        - 时间区间：覆盖你要回测的起止日期（建议多往前下载 2 个月，供均线/MACD 暖机）。
#      若未下载日线历史数据，handlebar 中 get_history_data 返回为空，策略不会有任何交易。
#   2. 确认〖数据管理〗中板块成分已更新（沪深300 调仓日之后才有最新成分）。
#
# 二、在〖模型研究〗中配置回测参数：
#   1. 新建策略 → 把本文件全部内容粘贴进去 → 保存。
#   2. 点“回测”/“回测参数”，按下表设置：
#        周期           日线（1d）
#        开始时间        如 2020-01-01
#        结束时间        如 2024-12-31
#        初始资金        如 1,000,000
#        佣金(买卖)      0.0003 （万三）
#        印花税          0.001  （仅卖出单边）
#        过户费          0.00002（沪市单边，深市免）
#        滑点            0.001  （千一，可按需调 0）
#        账号            选择你的回测账号；或在代码顶部 ACCOUNT_ID 填入
#   3. 点“开始回测”。回测结束后 stop() 自动导出三类文件到 G:/QMT_Reports/：
#        backtest_trades.csv  成交明细（UTF-8，Excel 可直接打开）
#        backtest_nav.csv     每日净值序列（含日收益率）
#        backtest_report.md   Markdown 绩效汇总报告
#
# 三、常见排错：
#   - 若回测无成交：先检查日线数据是否已下载；再核对 passorder 的 ORDER_TYPE_LIMIT
#     （部分版本为 0，可把顶部常量改为 0 再试）。
#   - 若 CSV 中文乱码：用 Excel “数据→从文本”导入并选 UTF-8；或改用 utf-8-sig（已是）。
#   - 若 to_markdown 报错：说明未装 tabulate，代码已自动回退为手写 Markdown 表格，无需处理。
# =================================================================================
