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
from mootdx.quotes import Quotes
from tqdm import tqdm

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

def _get_mktcap_ak() -> pd.DataFrame:
    """实时快照，返回列：code, mktcap（单位：元）"""
    for attempt in range(1, 4):
        try:
            df = ak.stock_zh_a_spot_em()
            break
        except Exception as e:
            logger.warning("AKShare 获取市值快照失败(%d/3): %s", attempt, e)
            time.sleep(backoff := random.uniform(1, 3) * attempt)
    else:
        raise RuntimeError("AKShare 连续三次拉取市值快照失败！")

    df = df[["代码", "总市值"]].rename(columns={"代码": "code", "总市值": "mktcap"})
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    return df

# --------------------------- 股票池筛选 --------------------------- #

def get_constituents(
    min_cap: float,
    max_cap: float,
    small_player: bool,
    mktcap_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    df = mktcap_df if mktcap_df is not None else _get_mktcap_ak()

    cond = (df["mktcap"] >= min_cap) & (df["mktcap"] <= max_cap)
    if small_player:
        cond &= ~df["code"].str.startswith(("300", "301", "688", "8", "4"))

    codes = df.loc[cond, "code"].str.zfill(6).tolist()

    # 附加股票池 appendix.json
    try:
        with open("appendix.json", "r", encoding="utf-8") as f:
            appendix_codes = json.load(f)["data"]
    except FileNotFoundError:
        appendix_codes = []
    codes = list(dict.fromkeys(appendix_codes + codes))  # 去重保持顺序

    logger.info("筛选得到 %d 只股票", len(codes))
    return codes

# --------------------------- 历史 K 线抓取 --------------------------- #
COLUMN_MAP_HIST_AK = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}

_FREQ_MAP = {
    0: "5m",
    1: "15m",
    2: "30m",
    3: "1h",
    4: "day",
    5: "week",
    6: "mon",
    7: "1m",
    8: "1m",
    9: "day",
    10: "3mon",
    11: "year",
}

# ---------- Tushare 工具函数 ---------- #

def _to_ts_code(code: str) -> str:
    return f"{code.zfill(6)}.SH" if code.startswith(("60", "68", "9")) else f"{code.zfill(6)}.SZ"


def _get_kline_tushare(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    adj_flag = None if adjust == "" else adjust
    for attempt in range(1, 4):
        try:
            df = ts.pro_bar(
                ts_code=ts_code,
                adj=adj_flag,
                start_date=start,
                end_date=end,
                freq="D",
            )
            break
        except Exception as e:
            logger.warning("Tushare 拉取 %s 失败(%d/3): %s", code, attempt, e)
            time.sleep(random.uniform(1, 2) * attempt)
    else:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"trade_date": "date", "vol": "volume"})[
        ["date", "open", "close", "high", "low", "volume"]
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    df[[c for c in df.columns if c != "date"]] = df[[c for c in df.columns if c != "date"]].apply(
        pd.to_numeric, errors="coerce"
    )    
    return df.sort_values("date").reset_index(drop=True)

# ---------- AKShare 工具函数 ---------- #

def _get_kline_akshare(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    for attempt in range(1, 4):
        try:
            # 默认延迟 100ms~500ms
            time.sleep(random.uniform(0.1, 0.5))
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
            break
        except Exception as e:
            logger.warning("AKShare 拉取 %s 失败(%d/3): %s", code, attempt, e)
            time.sleep(random.uniform(1, 2) * attempt)
    else:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = (
        df[list(COLUMN_MAP_HIST_AK)]
        .rename(columns=COLUMN_MAP_HIST_AK)
        .assign(date=lambda x: pd.to_datetime(x["date"]))
    )
    df[[c for c in df.columns if c != "date"]] = df[[c for c in df.columns if c != "date"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df[["date", "open", "close", "high", "low", "volume"]]
    return df.sort_values("date").reset_index(drop=True)

# ---------- AKTools 工具函数 ---------- #

def _get_kline_aktools(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    url = "http://host.docker.internal:8000/api/public/stock_zh_a_hist"
    params = {
        "symbol": code,
        "period": "daily",
        "start_date": start,
        "end_date": end,
        "adjust": adjust,
    }
    for attempt in range(1, 4):
        try:
            # 默认延迟 100ms~500ms
            time.sleep(random.uniform(1, 3))

            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame(data)
            break
        except Exception as e:
            logger.warning("Aktools 拉取 %s 失败(%d/3): %s", code, attempt, e)
            # 带退避+抖动
            time.sleep(random.uniform(1, 2) * attempt + random.uniform(0.1, 0.5))
    else:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = (
        df[list(COLUMN_MAP_HIST_AK)]
        .rename(columns=COLUMN_MAP_HIST_AK)
        .assign(date=lambda x: pd.to_datetime(x["date"]))
    )
    df[[c for c in df.columns if c != "date"]] = df[[c for c in df.columns if c != "date"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df[["date", "open", "close", "high", "low", "volume"]]
    return df.sort_values("date").reset_index(drop=True)

# ---------- Mootdx 工具函数 ---------- #

def _get_kline_mootdx(code: str, start: str, end: str, adjust: str, freq_code: int) -> pd.DataFrame:    
    symbol = code.zfill(6)
    freq = _FREQ_MAP.get(freq_code, "day")
    client = Quotes.factory(market="std")
    try:
        # 默认延迟 100ms~500ms
        time.sleep(random.uniform(0.5, 2))
        df = client.bars(symbol=symbol, frequency=freq, adjust=adjust or None)
    except Exception as e:
        logger.warning("Mootdx 拉取 %s 失败: %s", code, e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.rename(
        columns={"datetime": "date", "open": "open", "high": "high", "low": "low", "close": "close", "vol": "volume"}
    )
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    start_ts = pd.to_datetime(start, format="%Y%m%d")
    end_ts = pd.to_datetime(end, format="%Y%m%d")
    df = df[(df["date"].dt.date >= start_ts.date()) & (df["date"].dt.date <= end_ts.date())].copy()    
    df = df.sort_values("date").reset_index(drop=True)    
    return df[["date", "open", "close", "high", "low", "volume"]]

# ---------- 通用接口 ---------- #

def get_kline(
    code: str,
    start: str,
    end: str,
    adjust: str,
    datasource: str,
    freq_code: int = 4,
) -> pd.DataFrame:
    if datasource == "tushare":
        return _get_kline_tushare(code, start, end, adjust)
    elif datasource == "akshare":
        return _get_kline_akshare(code, start, end, adjust)
    elif datasource == "aktools":
        return _get_kline_aktools(code, start, end, adjust)
    elif datasource == "mootdx":
        return _get_kline_mootdx(code, start, end, adjust, freq_code)
    else:
        raise ValueError("datasource 仅支持 'tushare', 'akshare' 或 'mootdx'")

# ---------- 数据校验 ---------- #

def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df

def drop_dup_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]
# ---------- 单只股票抓取 ---------- #
def fetch_one(
    code: str,
    start: str,
    end: str,
    out_dir: Path,
    incremental: bool,
    datasource: str,
    freq_code: int,
):    
    csv_path = out_dir / f"{code}.csv"

    # 增量更新：若本地已有数据则从最后一天开始
    if incremental and csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, parse_dates=["date"])
            last_date = existing["date"].max()
            if last_date.date() >= pd.to_datetime(end, format="%Y%m%d").date():
                logger.debug("%s 已是最新，无需更新", code)
                return
            start = last_date.strftime("%Y%m%d")
        except Exception:
            logger.exception("读取 %s 失败，将重新下载", csv_path)

    for attempt in range(1, 4):
        try:            
            new_df = get_kline(code, start, end, "qfq", datasource, freq_code)
            if new_df.empty:
                logger.debug("%s 无新数据", code)
                break
            new_df = validate(new_df)
            if csv_path.exists() and incremental:
                old_df = pd.read_csv(
                    csv_path,
                    parse_dates=["date"],
                    index_col=False
                )
                old_df = drop_dup_columns(old_df)
                new_df = drop_dup_columns(new_df)
                new_df = (
                    pd.concat([old_df, new_df], ignore_index=True)
                    .drop_duplicates(subset="date")
                    .sort_values("date")
                )
            new_df.to_csv(csv_path, index=False)
            break
        except Exception:
            logger.exception("%s 第 %d 次抓取失败", code, attempt)
            time.sleep(random.uniform(1, 3) * attempt)  # 指数退避
    else:
        logger.error("%s 三次抓取均失败，已跳过！", code)


# ---------- 主入口 ---------- #

def run(
        datasource="tushare",
        frequency=4,
        exclude_gem=True,
        min_mktcap=5e9,
        max_mktcap=float("+inf"),
        start="20190101",
        end="today",
        out="./data",
        workers=3,
        ts_token=None  # 允许外部传入token，更灵活
):
    """
    直接调用抓取K线的核心函数

    参数与命令行参数一致，增加了ts_token参数方便外部传入
    """
    # ---------- Token 处理 ---------- #
    if datasource == "tushare":
        import tushare as ts
        # 优先使用传入的token，否则使用默认
        ts_token = ts_token or " "  # 这里可以保留默认token
        ts.set_token(ts_token)
        global pro
        pro = ts.pro_api()

    # ---------- 日期解析 ---------- #
    start_date = dt.date.today().strftime("%Y%m%d") if start.lower() == "today" else start
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

    # ---------- 市值快照 & 股票池 ---------- #
    mktcap_df = _get_mktcap_ak()

    codes_from_filter = get_constituents(
        min_mktcap,
        max_mktcap,
        exclude_gem,
        mktcap_df=mktcap_df,
    )
    # 加上本地已有的股票，确保旧数据也能更新
    local_codes = [p.stem for p in out_dir.glob("*.csv")]
    codes = sorted(set(codes_from_filter) | set(local_codes))

    if not codes:
        logger.error("筛选结果为空，请调整参数！")
        return False  # 用返回值表示执行状态

    logger.info(
        "开始抓取 %d 支股票 | 数据源:%s | 频率:%s | 日期:%s → %s",
        len(codes),
        datasource,
        _FREQ_MAP[frequency],
        start_date,
        end_date,
    )

    # ---------- 多线程抓取 ---------- #
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                fetch_one,
                code,
                start_date,
                end_date,
                out_dir,
                True,
                datasource,
                frequency,
            )
            for code in codes
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            pass

    logger.info("全部任务完成，数据已保存至 %s", out_dir.resolve())
    return True  # 执行成功


# ---------- 命令行入口（保持兼容） ---------- #
def main():
    parser = argparse.ArgumentParser(description="按市值筛选 A 股并抓取历史 K 线")
    parser.add_argument("--datasource", choices=["tushare", "akshare", "mootdx", "aktools"], default="tushare", help="历史 K 线数据源")
    parser.add_argument("--frequency", type=int, choices=list(_FREQ_MAP.keys()), default=4, help="K线频率编码，参见说明")
    parser.add_argument("--exclude-gem", default=True, help="True则排除创业板/科创板/北交所")
    parser.add_argument("--min-mktcap", type=float, default=5e9, help="最小总市值（含），单位：元")
    parser.add_argument("--max-mktcap", type=float, default=float("+inf"), help="最大总市值（含），单位：元，默认无限制")
    parser.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD 或 'today'")
    parser.add_argument("--end", default="today", help="结束日期 YYYYMMDD 或 'today'")
    parser.add_argument("--out", default="./data", help="输出目录")
    parser.add_argument("--workers", type=int, default=3, help="并发线程数")
    parser.add_argument("--ts-token", help="tushare的token，优先级高于内部默认")  # 新增命令行参数
    args = parser.parse_args()

    # 调用核心函数，保持命令行兼容性
    run(
        datasource=args.datasource,
        frequency=args.frequency,
        exclude_gem=args.exclude_gem,
        min_mktcap=args.min_mktcap,
        max_mktcap=args.max_mktcap,
        start=args.start,
        end=args.end,
        out=args.out,
        workers=args.workers,
        ts_token=args.ts_token  # 传入命令行的token
    )

if __name__ == "__main__":
    main()
