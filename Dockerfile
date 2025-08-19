FROM python:3.12-slim

# 关键优化：更换为国内Debian镜像源（加速apt操作）
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    sed -i s@/security.debian.org/@/mirrors.aliyun.com/debian-security/@g /etc/apt/sources.list

# 安装cron（添加--no-install-recommends减少无关依赖）
# 清理缓存，减小镜像体积
RUN apt-get update && \
    apt-get install -y --no-install-recommends cron && \
    rm -rf /var/lib/apt/lists/* \

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com  --upgrade

# 复制脚本和配置文件
COPY . /app/

# 创建日志目录
RUN mkdir -p /app/logs

# 添加crontab任务: 每天15:30执行
RUN echo "30 15 * * * python3 /app/do_select_and_upload.py >> /app/logs/cron.log 2>&1" > /etc/cron.d/python-job

# 给予执行权限
RUN chmod 0644 /etc/cron.d/python-job

# 应用crontab
RUN crontab /etc/cron.d/python-job

# 启动命令（替代start.sh）
CMD service cron start && tail -f /app/logs/cron.log