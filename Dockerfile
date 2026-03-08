


# 1. 使用官方 Python 3.9 轻量级镜像作为基础
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 安装系统级依赖 (修正版)
# libgl1-mesa-glx 已被废弃，改为 libgl1
# 增加 libglib2.0-0 是为了防止 Open3D/OpenCV 的其他依赖报错
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 复制依赖文件并安装
COPY requirements.txt .
# 使用清华源加速安装
# 使用阿里云镜像，并信任该主机
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# 5. 复制项目所有文件到容器
COPY . .

# 6. 创建数据存储目录
RUN mkdir -p server_data

# 7. 暴露端口 5000
EXPOSE 5000

# 8. 启动命令
CMD ["python", "app.py"]















