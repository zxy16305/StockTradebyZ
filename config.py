import os
from dotenv import load_dotenv

# 加载.env文件中的配置
load_dotenv()

class Config:
    """配置基类，包含所有配置项的默认值"""

    # 应用基本配置
    TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
    # 机器人配置
    QQ_APP_ID = os.getenv('QQ_APP_ID')
    QQ_APP_SECRET = os.getenv('QQ_APP_SECRET')
    # 接口配置
    AK_TOOLS_HOST = os.getenv('AK_TOOLS_HOST')
    # ms 接口地址
    MS_HOST = os.getenv('MS_HOST')


def get_config():
    """获取当前环境的配置实例"""
    return Config
