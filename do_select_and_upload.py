import fetch_kline
import select_stock
import requests
import json

from config import Config


def main():
    """
    1. 从数据目录中读取所有股票的k线数据
    2. 从配置文件中读取筛选条件
    3. 筛选符合条件的股票
    4. 上传符合条件的股票到数据库
    :return:
    """
    with open('bot_config.json', 'r') as f:
        config = json.load(f)
    # success = fetch_kline.run(
    #     datasource="tushare",
    #     frequency=4,
    #     min_mktcap=5e9,
    #     max_mktcap=float("inf"),
    #     start="20200101",
    #     end="today",
    #     out="./data",
    #     exclude_gem=False,
    #     workers=1,
    #     ts_token=config['tushareToken']  # 可选，根据数据源决定
    # )
    success = fetch_kline.run(
        datasource="aktools",
        frequency=4,
        min_mktcap=5e9,
        max_mktcap=float("inf"),
        start="20200101",
        end="today",
        out="./data",
        exclude_gem=False,
        workers=1,
        ts_token="your_actual_token"  # 可选，根据数据源决定
    )
    if not success:
        print("数据获取失败")
        return

    results, trade_date = select_stock.run(
        data_dir="./data",
        config="./configs.json",
        # 可选参数：指定日期或股票列表
        # date="2025-08-18",
        # tickers="600000,600036"
    )

    # 上传结果到服务器
    upload_results(results, trade_date)


def upload_results(results, trade_date):
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
            "stockSelectionResult": json.dumps(stock_list)
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