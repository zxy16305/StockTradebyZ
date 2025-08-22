FROM python:3.12-slim

# 关键：设置环境变量禁用debconf交互式前端
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

# 配置阿里云源并安装cron
# 安装tzdata（时区数据）和cron，此时不会出现交互提示
# 配置时区为上海（可选，解决时间差问题）
RUN rm -f /etc/apt/sources.list && \
    rm -rf /etc/apt/sources.list.d/* && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm main non-free contrib" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security/ bookworm-security main non-free contrib" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends tzdata cron && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com  --upgrade

# 复制脚本和配置文件
COPY . /app/

# 创建日志目录
RUN mkdir -p /app/logs

# 添加crontab任务: 每天15:05执行
RUN echo "5 15 * * * cd /app && /usr/local/bin/python3 /app/do_select_and_upload.py >> /app/logs/cron.log 2>&1" > /etc/cron.d/python-job
# RUN echo "17 21 * * * cd /app && /usr/local/bin/python3 /app/do_select_and_upload.py >> /app/logs/cron.log 2>&1" > /etc/cron.d/python-job

# 给予执行权限
RUN chmod 0644 /etc/cron.d/python-job

# 应用crontab
RUN crontab /etc/cron.d/python-job

# 启动命令（替代start.sh）
CMD touch /app/logs/cron.log && service cron start && python3 /app/tradebot.py >> /app/logs/bot.log 2>&1
