from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

import fetch_kline

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 将日志写入文件
        logging.FileHandler("select_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("select")


# ---------- 工具 ----------

def load_data(data_dir: Path, codes: Iterable[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        fp = data_dir / f"{code}.csv"
        if not fp.exists():
            logger.warning("%s 不存在，跳过", fp.name)
            continue
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
        frames[code] = df
    return frames


def load_config(cfg_path: Path) -> List[Dict[str, Any]]:
    if not cfg_path.exists():
        logger.error("配置文件 %s 不存在", cfg_path)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        cfg_raw = json.load(f)

    # 兼容三种结构：单对象、对象数组、或带 selectors 键
    if isinstance(cfg_raw, list):
        cfgs = cfg_raw
    elif isinstance(cfg_raw, dict) and "selectors" in cfg_raw:
        cfgs = cfg_raw["selectors"]
    else:
        cfgs = [cfg_raw]

    if not cfgs:
        logger.error("configs.json 未定义任何 Selector")
        sys.exit(1)

    return cfgs


def instantiate_selector(cfg: Dict[str, Any]):
    """动态加载 Selector 类并实例化"""
    cls_name: str = cfg.get("class")
    if not cls_name:
        raise ValueError("缺少 class 字段")

    try:
        module = importlib.import_module("Selector")
        cls = getattr(module, cls_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载 Selector.{cls_name}: {e}") from e

    params = cfg.get("params", {})
    return cfg.get("alias", cls_name), cls(**params)


# ---------- 主函数 ----------
# ---------- 核心逻辑函数（返回数据） ---------- #
def run(
        data_dir="./data",
        config="./configs.json",
        date=None,
        tickers="all",
        min_mktcap=5e9
):
    """
    执行选股逻辑并返回结果

    参数:
        data_dir: 行情数据目录
        config: 配置文件路径
        date: 交易日（YYYY-MM-DD），None则使用最新日期
        tickers: 股票代码列表或"all"

    返回:
        dict: 选股结果，结构为 {alias: picks_list, ...}
        datetime: 实际使用的交易日
    """
    # --- 加载行情 ---
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在")

    df = fetch_kline._get_mktcap_ak()

    codes = (
        [f.stem for f in data_dir.glob("*.csv")]
        if tickers.lower() == "all"
        else [c.strip() for c in tickers.split(",") if c.strip()]
    )

    if min_mktcap is not None:
        # 只保留市值 >= min_mktcap 的股票
        mktcap_dict = df.set_index("code")["mktcap"].to_dict()
        codes = [c for c in codes if mktcap_dict.get(c, 0) >= min_mktcap]

    if not codes:
        raise ValueError("股票池为空！")

    data = load_data(data_dir, codes)
    if not data:
        raise ValueError("未能加载任何行情数据")

    # --- 处理交易日 ---
    trade_date = (
        pd.to_datetime(date)
        if date
        else max(df["date"].max() for df in data.values())
    )

    # --- 加载 Selector 配置 ---
    selector_cfgs = load_config(Path(config))

    # --- 执行选股并收集结果 ---
    results = {}
    for cfg in selector_cfgs:
        if cfg.get("activate", True) is False:
            continue

        try:
            alias, selector = instantiate_selector(cfg)
            picks = selector.select(trade_date, data)
            results[alias] = picks  # 存储选股结果
        except Exception as e:
            logger.error("跳过配置 %s：%s", cfg, e)
            results[alias] = None  # 出错时标记为None

    return results, trade_date


# ---------- 命令行入口（负责打印输出） ---------- #
def main():
    p = argparse.ArgumentParser(description="Run selectors defined in configs.json")
    p.add_argument("--data-dir", default="./data", help="CSV 行情目录")
    p.add_argument("--config", default="./configs.json", help="Selector 配置文件")
    p.add_argument("--date", help="交易日 YYYY-MM-DD；缺省=数据最新日期")
    p.add_argument("--tickers", default="all", help="'all' 或逗号分隔股票代码列表")
    p.add_argument("--min-mktcap", type=float, default=5e9, help="最小总市值（含），单位：元")
    args = p.parse_args()

    try:
        # 调用核心函数获取数据
        results, trade_date = run(
            data_dir=args.data_dir,
            config=args.config,
            date=args.date,
            tickers=args.tickers,
            min_mktcap=args.min_mktcap
        )

        # 处理日期显示
        if not args.date:
            logger.info("未指定 --date，使用最近日期 %s", trade_date.date())

        # 打印结果（原逻辑保留）
        for alias, picks in results.items():
            logger.info("")
            logger.info("============== 选股结果 [%s] ==============", alias)
            logger.info("交易日: %s", trade_date.date())

            if picks is None:
                logger.info("选股过程出错")
                continue

            logger.info("符合条件股票数: %d", len(picks))
            logger.info("%s", ", ".join(picks) if picks else "无符合条件股票")

    except Exception as e:
        logger.error("执行失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()