from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import akshare as ak
import pandas as pd
import requests
import tushare as ts
import gm.api as gmApi
from mootdx.quotes import Quotes
from tqdm import tqdm

from config import Config

warnings.filterwarnings("ignore")

# --------------------------- 全局日志配置 --------------------------- #
LOG_FILE = Path("./logs/fetch.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fetch_mktcap")

# 屏蔽第三方库多余 INFO 日志
for noisy in ("httpx", "urllib3", "_client", "akshare"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# --------------------------- 市值快照 --------------------------- #


# ---------- 主入口 ---------- #
CACHE_DAYS = 10  # 缓存天数，写入一次磁盘

# ------------------ 主函数 ------------------ #
def run(
        start="20190101",
        end="today",
        out="./data",
        ts_token=None
):
    # ---------- Token 处理 ---------- #
    import tushare as ts
    ts_token = ts_token or " "
    ts.set_token(ts_token)
    global pro
    pro = ts.pro_api()


    # ---------- 日期解析 ---------- #
    if end.lower() == "today":
        now = dt.datetime.now()
        if now.hour < 15:
            end_date = (now - dt.timedelta(days=1)).strftime("%Y%m%d")
        else:
            end_date = now.strftime("%Y%m%d")
    else:
        end_date = end

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 本地日期记录 ---------- #
    record_path = out_dir / "last_date.json"
    if record_path.exists():
        last_record = json.loads(record_path.read_text())
        start_date = last_record.get("last_date", start)
    else:
        start_date = start

    # ---------- 日期列表 ---------- #
    # df = pro.trade_cal(exchange='SSE', is_open='1',
    #                    start_date=start_date,
    #                    end_date=end_date,
    #                    fields='cal_date')
    # trade_dates = df['cal_date'].tolist()

    date_range = pd.date_range(
        pd.to_datetime(start_date, format="%Y%m%d"),
        pd.to_datetime(end_date, format="%Y%m%d")
    )
    trade_dates = [d.strftime("%Y%m%d") for d in date_range]

    if not trade_dates:
        logger.info("没有需要更新的日期")
        return True

    logger.info("开始抓取 %d 个交易日: %s → %s", len(trade_dates), trade_dates[0], trade_dates[-1])

    # 在主循环外维护一个 {code: last_date} 字典，记录每支股票已有数据的最后日期
    last_dates = {}

    # 先扫描已有文件，建立 last_date 索引
    for f in out_dir.glob("*.csv"):
        try:
            code = f.stem
            df = pd.read_csv(f, usecols=["date"])
            if not df.empty:
                last_dates[code] = df["date"].iloc[-1]  # 已经排好序
        except Exception:
            continue
    cache = {}  # {stock_code: list of rows}
    cache_count = 0

    # ---------- 顺序抓取，每天全市场数据 ---------- #
    for trade_date in trade_dates:
        df = fetch_one_day(trade_date)
        if df is None or df.empty:
            continue

        logger.info("%s 抓取完成，共 %d 支股票", trade_date, df["ts_code"].nunique())

        # 处理数据并放入缓存
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
        df = df[["ts_code", "date", "open", "high", "low", "close", "volume"]]

        for code, sub_df in df.groupby("ts_code"):
            stock_code = code[:6]
            last_date = last_dates.get(stock_code, "00000000")
            sub_df = sub_df[sub_df["date"] > last_date]

            if sub_df.empty:
                continue

            if stock_code not in cache:
                cache[stock_code] = []
            cache[stock_code].append(sub_df)
        cache_count += 1

        # 每 CACHE_DAYS 或最后一天写一次磁盘
        if cache_count >= CACHE_DAYS or trade_date == trade_dates[-1]:
            for stock_code, dfs in cache.items():
                csv_path = out_dir / f"{stock_code}.csv"
                combined = pd.concat(dfs, ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"]).sort_values("date")

                # 如果已有文件，读取旧数据并合并
                if csv_path.exists():
                    old_df = pd.read_csv(csv_path, parse_dates=["date"])
                    combined = pd.concat([old_df, combined], ignore_index=True)
                # 统一日期为字符串，去重排序
                combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
                combined = combined[["date", "open", "high", "low", "close", "volume"]]
                combined.to_csv(csv_path, index=False)
                last_dates[stock_code] = combined["date"].max()

            # 每天抓完更新最新日期
            record_path.write_text(json.dumps({"last_date": trade_date}))
            cache.clear()
            cache_count = 0
            logger.info("%s 批量写入完成", trade_date)

        # time.sleep(random.uniform(0.1, 0.5))

    logger.info("全部完成，数据已保存至 %s", out_dir.resolve())
    return True


# ------------------ 按日期抓取 ------------------ #
def fetch_one_day(trade_date: str):
    for attempt in range(1, 4):
        try:
            df = (
                pro.daily(trade_date=trade_date)
                .rename(columns={'vol': 'volume'})
                .assign(date=lambda x: pd.to_datetime(x["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d"))
            )

            if df is None or df.empty:
                logger.debug("%s 无数据", trade_date)
                return None

            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

            return df
        except Exception:
            logger.exception("%s 第 %d 次抓取失败", trade_date, attempt)
            time.sleep(random.uniform(1, 3) * attempt)
    logger.error("%s 三次抓取均失败，已跳过", trade_date)
    return None

# ---------- 命令行入口（保持兼容） ---------- #
def main():
    parser = argparse.ArgumentParser(description="按市值筛选 A 股并抓取历史 K 线")
    parser.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD 或 'today'")
    parser.add_argument("--end", default="today", help="结束日期 YYYYMMDD 或 'today'")
    parser.add_argument("--out", default="./data", help="输出目录")
    parser.add_argument("--ts-token", help="tushare的token，优先级高于内部默认")  # 新增命令行参数
    args = parser.parse_args()

    # 调用核心函数，保持命令行兼容性
    run(
        start=args.start,
        end=args.end,
        out=args.out,
        ts_token=args.ts_token  # 传入命令行的token
    )

if __name__ == "__main__":
    main()
