# 🚀 三维点云智能识别系统 V3.0

基于 Flask + Open3D + 机器学习的激光雷达点云处理平台，支持地面分割、目标聚类与智能分类。

## 📁 项目结构

```
/home/kali/point-cloud-3d/
├── app.py                    # Flask 主应用 (API + 后端处理)
├── train.py                  # 模型训练脚本 (生成仿真数据 + 随机森林分类器)
├── 3.py                      # 测试数据生成器 (街道场景仿真)
├── point_cloud_model.pkl     # 预训练分类模型
├── requirements.txt          # Python 依赖
├── Dockerfile                # 容器化配置
├── templates/
│   └── index.html            # 前端可视化界面 (Three.js)
└── server_data/              # 点云数据存储目录
```

## ⚡ 快速启动

### 方式一：一键启动脚本 (推荐)
```bash
/home/kali/3dpoint
```

### 方式二：手动启动
```bash
cd /home/kali/point-cloud-3d
python app.py
```

访问地址：**http://localhost:5000**

## 🔧 依赖安装

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 核心依赖
| 包名 | 用途 |
|------|------|
| flask | Web 框架 |
| flask-cors | 跨域支持 |
| open3d | 点云处理 (地面分割/聚类) |
| scikit-learn | 机器学习 (KMeans/GMM/随机森林) |
| numpy | 数值计算 |
| joblib | 模型序列化 |

## 🎯 核心功能

### 1. 地面分割
使用 Open3D RANSAC 算法分离地面点与非地面点
- 距离阈值：0.3 米
- 降级方案：按高度百分位切分

### 2. 聚类算法 (3 种可选)
| 算法 | 特点 | 适用场景 |
|------|------|----------|
| **DBSCAN (动态缩放)** | 自适应距离密度，专家级 | 默认推荐，处理近密远疏 |
| K-Means | 快速，需指定簇数 | 简单场景 |
| GMM | 概率模型，需指定组件数 | 复杂分布 |

### 3. 目标分类 (5 类)
基于 9 维特征的随机森林分类器：
- 🚶 **行人 (Person)** - 圆柱体，长宽比≈1
- 🚗 **车辆 (Car)** - L 形反射，长宽比>2
- 🌲 **植被 (Tree)** - 球状分布
- 🏢 **建筑 (Building)** - 极长薄墙
- 📍 **杆状物 (Pole)** - 细高圆柱

### 4. 特征工程 (V3 终极版)
```python
[线密度，平面度，散射度，高度，最大跨度，最小跨度，体积，密度，长宽比]
```

## 📊 API 接口

### POST /process
处理点云文件并返回识别结果

**请求参数:**
```json
{
  "file": "0000000000.bin",
  "algorithm": "dbscan",
  "mobile": false,
  "eps": 0.5,
  "min_samples": 10,
  "auto_k": true
}
```

**响应示例:**
```json
{
  "status": "success",
  "meta": {
    "processing_time": 1.23,
    "total_points": 50000,
    "clusters": 8
  },
  "data": {
    "ground": [[x,y,z], ...],
    "clusters": [[[x,y,z], ...], ...],
    "labels": ["car", "person", ...],
    "bboxes": [{"min": [...], "max": [...], "type": "car"}, ...]
  }
}
```

### GET /files
获取 server_data 目录下的点云文件列表

## 🎨 前端功能

- 🌐 三维点云可视化 (Three.js)
- 📱 移动端适配 (响应式面板)
- 🌓 深色/浅色主题切换
- 🎯 交互式参数调节
- 📦 识别框与标签显示

## 🧪 生成测试数据

```bash
python 3.py
```
生成 `server_data/street_scene_sim.bin` 包含：
- 5 辆车
- 4 个行人
- 5 栋建筑
- 8 棵树
- 地面网格

## 🤖 重新训练模型

```bash
python train.py
```
- 生成 600×5=3000 条仿真样本
- 训练 200 棵树的随机森林
- 输出 `point_cloud_model.pkl`

## ⚠️ 注意事项

1. **点云格式**: 支持 Nx3 (xyz) 或 Nx4 (xyz+intensity) 二进制文件
2. **性能优化**: 移动端自动下采样，超过 30 万点进一步降采样
3. **模型匹配**: 特征提取必须与 train.py 保持一致 (9 维)
4. **网络问题**: pip 安装失败时使用阿里云镜像源

## 📝 版本历史

| 版本 | 更新内容 |
|------|----------|
| V3.0 | 新增长宽比特征，L 形车辆仿真，动态缩放 DBSCAN |
| V2.0 | 添加 GMM 聚类，移动端适配 |
| V1.0 | 基础地面分割 + KMeans 聚类 |

## 📄 License

MIT License