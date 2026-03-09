# 3D Point Cloud Processing Web App 🚗☁️

这是一个基于 Python (Flask + Open3D + Scikit-Learn) 构建的 3D 点云处理与可视化后端项目。它能够读取自动驾驶/激光雷达场景下的点云数据 (`.bin` 格式)，并自动进行降采样、去噪、地面分割、聚类以及 3D 最小包围盒 (OBB) 的计算，最终将结果提供给前端进行可视化。

## ✨ 核心功能 (Features)

*   **极速预处理**: 基于 Open3D 的体素降采样 (Voxel Downsampling) 与统计滤波去噪 (SOR)。
*   **地面分割**: 基于 RANSAC 算法的高效平面提取，智能分离地面与障碍物。
*   **对象聚类**: 支持 DBSCAN 与 KMeans (MiniBatch) 算法，自动分割独立物体。
*   **OBB 包围盒**: 基于主成分分析 (PCA) 计算带有真实旋转角度 (Yaw) 的 3D 有向包围盒。
*   **开箱即用**: 提供 Dockerfile，完美解决 Open3D 在 Linux 环境下的 `libgl1` 依赖问题，支持一键部署到云端 (如 Claw Cloud, Sealos 等)。

## 🛠️ 技术栈 (Tech Stack)

*   **后端框架**: Flask, Flask-CORS, Werkzeug
*   **点云处理引擎**: Open3D (C++ 底层加速)
*   **机器学习/几何计算**: Scikit-Learn (DBSCAN/KMeans/PCA), NumPy
*   **部署**: Docker

## 📁 项目结构 (Project Structure)

```text
/pointcloud-app
├── app.py                  # Flask 主程序与核心算法引擎
├── requirements.txt        # Python 依赖清单
├── Dockerfile              # Docker 镜像构建脚本
├── templates/              # 前端网页模板
│   └── index.html          # 可视化 UI 界面 (基于 Three.js 等)
├── static/                 # 前端静态资源 (CSS/JS)
└── server_data/            # 存放测试用例的点云数据
    ├── 000000.bin          # KITTI 格式点云 (x, y, z, intensity)
    └── ...
🚀 本地运行 (Local Development)
前置条件
建议使用 Python 3.8 或以上版本。
克隆/下载代码到本地
安装依赖
code
Bash
pip install -r requirements.txt
放入测试数据
确保 server_data/ 文件夹下至少有一个 .bin 文件。
启动服务
code
Bash
python app.py
访问应用
打开浏览器访问: http://127.0.0.1:5000
🐳 Docker 部署 (Docker Deployment)
本项目自带优化过的 Dockerfile，可直接打包并在任意云平台运行。
构建镜像
code
Bash
docker build -t your-username/pointcloud-app:latest .
本地测试镜像
code
Bash
docker run -d -p 5000:5000 your-username/pointcloud-app:latest
推送到镜像仓库
code
Bash
docker push your-username/pointcloud-app:latest
⚠️ 云端部署注意事项 (Claw Cloud / Sealos)
容器端口 (Container Port): 必须设置为 5000。
网络访问: 部署完成后，请使用平台提供的公网 URL (Public Access Domain) 访问，形如 https://xxx.claw.run，不要访问内网 .local 地址。
🔌 API 接口文档 (API Reference)
1. 获取文件列表
URL: /list_files
Method: GET
Response:
code
JSON
{
  "status": "success",
  "files":["000000.bin", "000001.bin"]
}
2. 处理点云数据
URL: /process
Method: POST
Headers: Content-Type: application/json
Body Example:
code
JSON
{
  "file": "000000.bin",
  "algorithm": "dbscan",
  "eps": 0.6,
  "min_samples": 10
}
Response Example:
code
JSON
{
  "status": "success",
  "meta": {
    "time_sec": 0.25,
    "total_points": 120000,
    "valid_points": 45000,
    "ground_points": 35000,
    "num_objects": 15
  },
  "data": {
    "ground": [[x, y, z], ...],
    "clusters": [[[x,y,z], ...], ...],
    "labels": ["0", "1", "2"],
    "bboxes": [
      {
        "center": [10.5, 2.1, 0.5],
        "dimensions": [4.2, 1.8, 1.5],
        "rotation_z": 1.57
      }
    ]
  }
}
📝 许可证 (License)
MIT License
