import pandas as pd
import numpy as np

def safe_divide(numerator, denominator, default=0.0):
    result = np.full_like(numerator, default, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result

def time_feature(df, name="time"):
    """
    time归一化与周期性编码
    周期性编码可能需要细化调整
    """
    t = pd.to_datetime(df[name], format='%H:%M:%S')
    hours = t.dt.hour
    minutes = t.dt.minute
    seconds = t.dt.second
    df['time_frac'] = (hours * 3600 + minutes * 60 + seconds) / 86400.0

    df['sin_time'] = np.sin(2 * np.pi * df['time_frac'])
    df['cos_time'] = np.cos(2 * np.pi * df['time_frac'])
    return df, ["sin_time", "cos_time", "time_frac"]

def stock_feature(df):
    """除滑窗特征外的所有特征"""
    bid_cols = ['n_bid1', 'n_bid2', 'n_bid3', 'n_bid4', 'n_bid5']
    ask_cols = ['n_ask1', 'n_ask2', 'n_ask3', 'n_ask4', 'n_ask5']
    bsize_cols = ['n_bsize1', 'n_bsize2', 'n_bsize3', 'n_bsize4', 'n_bsize5']
    asize_cols = ['n_asize1', 'n_asize2', 'n_asize3', 'n_asize4', 'n_asize5']

    # 矢量化操作，避免逐列操作
    # 把涨跌幅范围[-1,1]映射到[0,2]
    df[bid_cols] += 1
    df[ask_cols] += 1

    # 使用 NumPy 进行矢量化计算
    # 这里的 bid_values, ask_values, bsize_values, asize_values 是二维数组
    # bid_values[:, 0] 是 bid1 的值， bid_values[:, 1] 是 bid2 的值，以此类推
    bid_values = df[bid_cols].values
    ask_values = df[ask_cols].values
    bsize_values = df[bsize_cols].values
    asize_values = df[asize_cols].values
    high_values = df["n_high"].values
    low_values = df["n_low"].values
    close_values = df["n_close"].values
    open_values = df["n_open"].values
    mid_values = df["n_midprice"].values
    # 此处是一档中间价，简称中间价
    
    
    # 下面正式进入特征工程
    

    # 五档价差 Bid-Ask Spread
    """
    spread越小，市场流动性越高，预示上涨趋势；
    spread越大，交易成本越大，预示下跌趋势
    """
    spread = ask_values - bid_values
    for i in range(5):
        df[f'spread{i + 1}'] = spread[:, i]
    # 这里做了五档，后续根据特征重要性进行筛选
    # df['spread1'], df['spread2'], df['spread3'] = spread[:, 0], spread[:, 1], spread[:, 2]


    # 五档中间价 mid_price
    """
    mid price是市场公允价格估计，更加接近理论价格
    """
    mid_price = (ask_values + bid_values) / 2
    for i in range(5):
        df[f'mid_price{i + 1}'] = mid_price[:, i]
    # 这里做了五档，后续根据特征重要性进行筛选
    # df['mid_price1'], df['mid_price2'], df['mid_price3'] = mid_price[:, 0], mid_price[:, 1], mid_price[:, 2]


    # 最近成交价与中间价价差 deviation
    """
    deviation是当前价格与中间价的差值，衡量的是当前成交价偏离买卖双方中间价格的程度和方向
    >0时偏向买方，<0时偏向卖方，反映交易倾斜，可用于推断当前市场中主动交易方向
    """
    df["deviation"] = close_values - mid_values
    # 这里的中间价是mid_price1
    # 或许也可以safe_divide(close_values - mid_values, mid_values, 0.0)
    # 但是用它的话特征提取时不能去掉n_midprice(因为不是常数）


    # 相对开盘价的额外涨跌幅 momentum_from_open
    """
    当前成交价涨跌幅与开盘价涨跌幅的差值
    即从开盘到当前，又涨（或跌）了多少
    """
    df["mfo"] = close_values - open_values
    # 如果safe_divide(close_values - open_values, open_values, 0.0)
    # 那么特征提取时可以去掉n_open(因为是常数，否则会冗余)



    # 五档订单累计失衡差值 Order Imbalance Difference
    """
    前置说明：所有 n_bsize* / n_asize* 已在数据预处理时做了 “1 + 挂单量涨跌幅” 处理，
    即这些列代表挂单量相对于基准的比例（>1 表示增量，<1 表示减量）。

    该指标利用挂单量“比例”之和的差值来衡量前 n 档深度内的买卖力量增量对比。

    定义：
        OID_n = ∑[i=1..n] n_bsize_i - ∑[i=1..n] n_asize_i

    说明：
    - n 从 1 到 5，对应前 n 档的累计挂单量比例总和差值；
    - OID_n > 0 表示买方挂单量比例增幅更大，上行信号更强；
    - OID_n < 0 表示卖方挂单量比例增幅更大，下行信号更强；
    - 值域无固定范围，可直接反映“增量”强度。
    """
    for n in range(1, 6):
        bid_change = df[[f"n_bsize{i}" for i in range(1, n + 1)]].sum(axis=1)
        ask_change = df[[f"n_asize{i}" for i in range(1, n + 1)]].sum(axis=1)
        df[f'oid_{n}'] = bid_change - ask_change


    # 五档加权价格压力累计差值 Price Pressure Difference
    """
    前置说明：所有 n_ask* / n_bid*、n_asize* / n_bsize* 均为“1+涨跌幅”形式。

    该指标基于挂单量比例与对应价格的乘积差，衡量前 n 档深度内的价量压力净差。

    定义：
        PPD_n = ∑[i=1..n](n_ask_i × n_asize_i) – ∑[i=1..n](n_bid_i × n_bsize_i)

    说明：
    - n 从 1 到 5，分别对应前 n 档的加权差值；
    - PPD_n > 0 表示卖方价量压力更大，下行驱动力强；
    - PPD_n < 0 表示买方价量压力更大，上行驱动力强；
    - 结果为净压力差，直观反映瞬时市场冲击信号。
    """
    for n in range(1, 6):
        ask_pressure = sum(df[f"n_ask{i}"] * df[f"n_asize{i}"] for i in range(1, n + 1))
        bid_pressure = sum(df[f"n_bid{i}"] * df[f"n_bsize{i}"] for i in range(1, n + 1))
        df[f'ppd_{n}'] = ask_pressure - bid_pressure


    
    # 五档订单累计失衡比 Order Imbalance Ratio
    """
    前置说明：所有 n_bsize* / n_asize* 已做 “1 + 涨跌幅” 处理，代表挂单量比例。

    该指标在上述比例基础上归一化，衡量前 n 档深度内的结构性失衡度。

    定义：
        OIR_n = (∑[i=1..n] n_bsize_i – ∑[i=1..n] n_asize_i)
                / (∑[i=1..n] n_bsize_i + ∑[i=1..n] n_asize_i)

    说明：
    - 先求前 n 档挂单量比例之和的差，再除以总和，结果 ∈ [-1, 1]；
    - OIR_n > 0 表示买方比例占优，OIR_n < 0 表示卖方比例占优；
    - 适用于消除总挂单量波动影响，便于跨时段、跨品种比较。
    """
    for n in range(5):
        bid_sum = df[[f"n_bsize{i+1}" for i in range(n)]].sum(axis=1)
        ask_sum = df[[f"n_asize{i+1}" for i in range(n)]].sum(axis=1)
        df[f'oir_{n+1}'] = safe_divide(bid_sum - ask_sum, bid_sum + ask_sum, 0.0)
    # 这里没有做单一档位的 OIR 特征，后续可以预测效果进行调整



    # 五档加权价格压力累计失衡比 Price Pressure Imbalance Ratio
    """
    前置说明：所有 n_* 列均为“1 + 涨跌幅”形式，代表比例数据。

    该指标在“1+比例”基础上加入价格权重，衡量前 n 档深度内的结构性价量压力不对称。

    定义：
        PPI_n = (∑[i=1..n](n_ask_i * n_asize_i) – ∑[i=1..n](n_bid_i * n_bsize_i))
                / (∑[i=1..n] n_asize_i + ∑[i=1..n] n_bsize_i)

    说明：
    - 将挂单量比例乘以档位价格计算加权压力；
    - n 从 1 到 5，分别对应前 n 档的结构失衡度；
    - PPI_n ∈ [-1, 1]，数值越正表示卖压越强，越负表示买压越强；
    - 能同时反映买卖比例和价格变化，适合高频结构性特征构建。
    """
    for n in range(5):
        ask_pressure = sum(
            df[f"n_ask{i+1}"] * df[f"n_asize{i+1}"] for i in range(n)
        )
        bid_pressure = sum(
            df[f"n_bid{i+1}"] * df[f"n_bsize{i+1}"] for i in range(n)
        )
        total_volume = df[[f"n_bsize{i+1}" for i in range(n)]].sum(axis=1) + \
                    df[[f"n_asize{i+1}" for i in range(n)]].sum(axis=1)
        df[f'ppi_{n+1}'] = safe_divide(ask_pressure - bid_pressure, total_volume, 0.0)
    # 这里没有做单一档位的 PPI 特征，后续可以预测效果进行调整


    # 价格位置指标 Price Position Indicators
    """
    反映当前价格在一定时间窗口内的位置
    常见指标为相对位置指标 Relative Position Indicator 和威廉指标 Williams %R 
    此处选择相对位置指标 Relative Position
    """
    df["RP"] = safe_divide(close_values - low_values, high_values - low_values, 0.5)
    # 由于没有收盘价，如果用最近成交价来代替威廉指标中的收盘价，会导致其与相对位置指标的加和为1
    # 所以这里选择相对位置指标


    df["depth_bid"] = bsize_values.sum(axis=1)
    df["depth_ask"] = asize_values.sum(axis=1)
    df["amplitude"] = safe_divide(high_values - low_values, open_values)
    # 这几个还没想好怎么处理，先用着


    # 对 amount_delta 应用 np.log1p
    df['amount'] = np.log1p(df['amount_delta'].values)
    # 返回所需的特征列
    # 这里还需要一个大单特征
    

    feature_col_names = [
        'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3',
        'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1',
        'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4',
        'n_ask5', 'n_asize5', 
        "amount",
        "RP",
        "deviation",
        "mfo",
        "depth_bid",
        "depth_ask",
        "amplitude",
    ]
    for i in range(6):
        feature_col_names.extend([f'spread{i}', f'mid_price{i}',
                                  f'oid_{n}', f'ppd_{n}', f'oir_{n}', f'ppi_{n}'])
    return df, feature_col_names


def main_feature(df, only_columns=False):
    """返回df一定要保留sym列，之后要使用对应sym的模型进行预测"""
    total = []
    if only_columns:
        df = df.iloc[:1].copy()
    for func in [time_feature, stock_feature]:
        df, t = func(df)
        total.extend(t)
    if only_columns:
        return total
    return df[total + ["sym"]]



