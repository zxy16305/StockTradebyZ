import random
import time
from datetime import datetime

import pandas as pd

import fetch_kline
import fetch_kline_by_date
import select_stock
import requests
import json
import gm.api as gmApi
import akshare as ak

from config import Config


def main():
    """
    1. 从数据目录中读取所有股票的k线数据
    2. 从配置文件中读取筛选条件
    3. 筛选符合条件的股票
    4. 上传符合条件的股票到数据库
    :return:
    """
    # out = "./data"
# success = fetch_kline.run(
    #     datasource="aktools",
    #     frequency=4,
    #     min_mktcap=5e9,
    #     max_mktcap=float("inf"),
    #     start="20200101",
    #     end="today",
    #     out="./data",
    #     exclude_gem=False,
    #     workers=1,
    #     ts_token="your_actual_token"  # 可选，根据数据源决定
    # )
    out = "./data2"

    """实时快照，返回列：code, mktcap（单位：元）"""
    for attempt in range(1, 4):
        try:
            df = ak.stock_zh_a_spot_em()
            break
        except Exception as e:
            print("AKShare 获取市值快照失败(%d/3): %s", attempt, e)
            time.sleep(backoff := random.uniform(1, 3) * attempt)
    else:
        raise RuntimeError("AKShare 连续三次拉取市值快照失败！")

        # 自动取当天日期
    trade_date = pd.Timestamp(datetime.now().date())

    mapping = {
        "代码": "code",
        "今开": "open",
        "最高": "high",
        "最低": "low",
        "最新价": "close",
        "成交量": "volume",
    }

    # 只保留需要的列，并重命名
    df_extra = df[list(mapping.keys())].rename(columns=mapping)

    # 补充日期列
    df_extra["date"] = trade_date

    # 调整列顺序
    df_extra = df_extra[["code","date","open","high","low","close","volume"]]

    results, trade_date = select_stock.run(
        data_dir=out,
        config="./configs.json",
        # 可选参数：指定日期或股票列表
        # date="2025-08-18",
        # tickers="600000,600036"
        extra_data=df_extra
    )

    # 上传结果到服务器
    upload_results_intraday(results, trade_date)


def upload_results_intraday(results, trade_date):
    """
    上传选股结果到服务器
    :param results: 选股结果
    :param trade_date: 交易日期
    :return:
    """
    # 服务器地址
    host = f"http://{Config.MS_HOST}"
    # host = "http://localhost:8080"

    # 遍历所有策略的结果
    for strategy, stock_list in results.items():
        if stock_list is None:
            continue
            
        # 准备上传数据
        data = {
            "date": trade_date.strftime("%Y-%m-%d"),
            "strategy": strategy,
            "stockSelectionResult": json.dumps(stock_list),
            "isIntraday": True
        }
        
        # 发送POST请求
        try:
            response = requests.post(
                f"{host}/api/select",
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'insomnia/2023.5.8'
                },
                json=data
            )
            
            if response.status_code == 200:
                print(f"策略'{strategy}'的选股结果上传成功")
            else:
                print(f"策略'{strategy}'的选股结果上传失败，状态码: {response.status_code}")
        except Exception as e:
            print(f"策略'{strategy}'的选股结果上传时发生异常: {e}")


if __name__ == '__main__':
    main()