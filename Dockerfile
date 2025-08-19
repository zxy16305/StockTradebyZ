FROM python:3.12-slim

# 解决sources.list不存在的问题：直接创建阿里云源配置
# 清理原有源配置，仅保留阿里云源（避免多源冲突）
# 创建阿里云源配置（传统sources.list格式）
# 合并为单RUN指令，减少镜像层并安装cron
# 清理缓存，减小镜像体积
RUN rm -f /etc/apt/sources.list && \
    rm -rf /etc/apt/sources.list.d/* && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm main non-free contrib" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security/ bookworm-security main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-updates main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-backports main non-free contrib" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends cron && \
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
RUN mkdir -p /app/logs && touch /app/logs/cron.log &&  \
    chmod 777 /app/logs /app/logs/cron.log

# 添加crontab任务: 每天15:30执行
RUN echo "30 15 * * * python3 /app/do_select_and_upload.py >> /app/logs/cron.log 2>&1" > /etc/cron.d/python-job

# 给予执行权限
RUN chmod 0644 /etc/cron.d/python-job

# 应用crontab
RUN crontab /etc/cron.d/python-job

# 启动命令（替代start.sh）
CMD service cron start && tail -f /app/logs/cron.log