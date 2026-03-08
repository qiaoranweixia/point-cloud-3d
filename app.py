import os
import time
import logging
import traceback
import numpy as np
import joblib
import open3d as o3d
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# --- 1. 系统配置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # 允许跨域请求

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = app.logger

# 路径配置
SERVER_DATA_DIR = os.path.abspath('./server_data')
os.makedirs(SERVER_DATA_DIR, exist_ok=True)
MODEL_PATH = 'point_cloud_model.pkl'

# --- 2. 加载机器学习模型 ---
CLF_MODEL = None
try:
    if os.path.exists(MODEL_PATH):
        CLF_MODEL = joblib.load(MODEL_PATH)
        logger.info(f"✅ 成功加载机器学习模型: {MODEL_PATH}")
    else:
        logger.warning("⚠️ 未找到模型文件，将仅使用硬规则进行分类。请运行 train_model_v3.py 生成模型。")
except Exception as e:
    logger.error(f"❌ 模型加载失败 (可能是特征维度不匹配，请重新训练): {e}")


# --- 3. 核心处理类 ---
class PointCloudProcessor:
    def __init__(self, data):
        # 确保只取前3列 (x,y,z)，忽略强度等
        self.original_data = data[:, :3]
        # 初始化 Open3D 对象
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.original_data)

    def ground_segmentation(self):
        """
        使用 Open3D RANSAC 进行地面分割
        """
        try:
            # distance_threshold: 0.3米内的点视为地面
            plane_model, inliers = self.pcd.segment_plane(distance_threshold=0.3,
                                                          ransac_n=3,
                                                          num_iterations=100)

            inlier_cloud = self.pcd.select_by_index(inliers)
            outlier_cloud = self.pcd.select_by_index(inliers, invert=True)

            return np.asarray(inlier_cloud.points), np.asarray(outlier_cloud.points)
        except Exception as e:
            logger.error(f"地面分割异常: {e}")
            # 降级方案：按高度切分
            z = self.original_data[:, 2]
            mask = z < np.percentile(z, 20)
            return self.original_data[mask], self.original_data[~mask]

    # --- 聚类算法 A: 自适应动态缩放 DBSCAN (专家级) ---
    def cluster_dynamic_scaling(self, points, base_eps=0.5, min_points=5):
        """
        通过坐标缩放解决激光雷达"近密远疏"问题
        """
        if len(points) < min_points: return np.array([])

        # 1. 计算水平距离
        dists = np.linalg.norm(points[:, :2], axis=1)

        # 2. 动态缩放因子：距离越远，压缩越厉害，使密度看起来均匀
        # 系数 0.015 意味着每 100米 坐标被压缩 2.5 倍
        scale_factors = 1.0 + 0.015 * dists
        scaled_points = points / scale_factors[:, np.newaxis]

        # 3. Open3D 聚类
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(scaled_points)

        return np.array(pcd_temp.cluster_dbscan(eps=base_eps, min_points=min_points, print_progress=False))

    # --- 聚类算法 B: K-Means ---
    def cluster_kmeans(self, points, n_clusters=5):
        if len(points) < n_clusters: return np.zeros(len(points))
        try:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(points[::2])
            return kmeans.predict(points)
        except:
            return np.zeros(len(points))

    # --- 聚类算法 C: GMM ---
    def cluster_gmm(self, points, n_components=5):
        if len(points) < n_components: return np.zeros(len(points))
        try:
            gmm = GaussianMixture(n_components=n_components, n_init=1).fit(points[::2])
            return gmm.predict(points)
        except:
            return np.zeros(len(points))

    def extract_features(self, points):
        """
        [V3 终极版] 提取 9 维特征
        必须与 train_model_v3.py 保持完全一致！
        """
        if len(points) < 5: return np.zeros(9)

        # 1. 几何特征 (PCA)
        cov_mat = np.cov(points[:, :3].T)
        eigen_vals = np.linalg.eigvalsh(cov_mat)
        idx = eigen_vals.argsort()[::-1]
        l1, l2, l3 = eigen_vals[idx]
        if l1 <= 0: l1 = 1e-6

        linearity = (l1 - l2) / l1
        planarity = (l2 - l3) / l1
        scattering = l3 / l1

        # 2. 物理尺寸特征
        min_p = np.min(points[:, :3], axis=0)
        max_p = np.max(points[:, :3], axis=0)

        dx = max_p[0] - min_p[0]
        dy = max_p[1] - min_p[1]
        dz = max_p[2] - min_p[2]  # 高度

        max_span = max(dx, dy)  # 最大水平跨度
        min_span = min(dx, dy)  # 最小水平跨度

        # [关键] 长宽比：区分圆柱体(人)和长方体(车)
        aspect_ratio = max_span / (min_span + 1e-6)

        volume = dx * dy * dz
        if volume <= 0: volume = 1e-6
        density = len(points) / volume

        return np.array([
            linearity, planarity, scattering,
            dz, max_span, min_span,
            volume, density, aspect_ratio
        ])

    def classify_object(self, points):
        """
        混合分类引擎：ML预测 + 硬规则修正
        """
        # 1. 提取特征
        feats = self.extract_features(points)

        # 解析关键指标用于硬规则
        height = feats[3]
        max_span = feats[4]
        aspect_ratio = feats[8]

        label_name = "未知"
        label_type = "unknown"

        # 2. 模型预测
        if CLF_MODEL:
            try:
                # 预测
                pred = CLF_MODEL.predict(feats.reshape(1, -1))[0]

                # 映射表
                mapping = {
                    "tree": ("植被", "tree"),
                    "person": ("行人", "person"),
                    "pole": ("杆状物", "person"),  # 前端同一颜色
                    "car": ("车辆", "car"),
                    "building": ("建筑", "building")
                }
                label_name, label_type = mapping.get(pred, ("未知", "unknown"))
            except Exception as e:
                pass  # 忽略预测错误，使用规则兜底

        # 3. [硬约束修正] 纠正模型的低级错误

        # 修正 A: 把"人"修正为"车/墙"
        if label_type == "person":
            if max_span > 1.2:  # 超过1.2米宽，绝不是单个人
                label_name, label_type = "车辆/物体", "car"
            if height < 0.5:  # 太矮
                label_name, label_type = "杂物", "unknown"

        # 修正 B: 把"车"修正为"人"
        if label_type == "car":
            if max_span < 1.5 and aspect_ratio < 1.8:
                # 又短又方，更像人或垃圾桶
                label_name, label_type = "行人/物体", "person"

        # 修正 C: 绝对尺寸过滤 (霸道规则)
        if height > 4.0:
            label_name, label_type = "高大建筑", "building"
        if max_span > 8.0:
            label_name, label_type = "长墙/建筑", "building"

        return label_name, label_type


# --- 4. 路由定义 ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """上传文件接口"""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "无文件"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"}), 400

    if file:
        filename = secure_filename(file.filename)
        # 时间戳防重名
        save_name = f"{int(time.time())}_{filename}"
        file.save(os.path.join(SERVER_DATA_DIR, save_name))
        return jsonify({"status": "success", "filename": save_name})


@app.route('/list_files')
def list_files():
    """获取文件列表，按时间倒序"""
    try:
        files = [f for f in os.listdir(SERVER_DATA_DIR) if f.endswith('.bin')]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(SERVER_DATA_DIR, x)), reverse=True)
        return jsonify({"status": "success", "files": files})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/process', methods=['POST'])
def process_pointcloud():
    """
    核心处理接口
    流程: 加载 -> 下采样 -> 地面分割 -> 聚类 -> 分类 -> 打包返回
    """
    start_time = time.time()
    try:
        req = request.get_json()
        filename = req.get('file')
        algorithm = req.get('algorithm', 'dbscan')
        is_mobile = req.get('mobile', False)

        filepath = os.path.join(SERVER_DATA_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("文件不存在")

        # 读取二进制点云
        raw_data = np.fromfile(filepath, dtype=np.float32)
        # 兼容 Nx3 或 Nx4 格式
        if raw_data.size % 4 == 0:
            points = raw_data.reshape(-1, 4)[:, :3]
        elif raw_data.size % 3 == 0:
            points = raw_data.reshape(-1, 3)
        else:
            raise ValueError("无法解析点云维度")

        # 下采样 (优化性能)
        step = 8 if is_mobile else 3
        if len(points) > 300000: step *= 2
        sampled_points = points[::step]

        # 初始化处理器
        processor = PointCloudProcessor(sampled_points)

        # 1. 地面分割
        ground, objects = processor.ground_segmentation()

        clusters = []
        bboxes = []
        labels_text = []

        if len(objects) > 0:
            # 2. 聚类
            if algorithm == 'kmeans':
                n_clusters = int(req.get('n_clusters', 5))
                if req.get('auto_k', True): n_clusters = 6  # 简化处理
                cluster_labels = processor.cluster_kmeans(objects, n_clusters)
            elif algorithm == 'gmm':
                n_components = int(req.get('n_components', 5))
                if req.get('auto_gmm', True): n_components = 6
                cluster_labels = processor.cluster_gmm(objects, n_components)
            else:
                # 默认: 自适应动态 DBSCAN
                eps = float(req.get('eps', 0.5))
                min_samples = int(req.get('min_samples', 10))
                cluster_labels = processor.cluster_dynamic_scaling(objects, base_eps=eps, min_points=min_samples)

            # 3. 分类与封装
            if len(cluster_labels) > 0:
                max_label = int(cluster_labels.max())
                for i in range(max_label + 1):
                    mask = cluster_labels == i
                    c_points = objects[mask]

                    if len(c_points) < 10: continue  # 忽略噪点

                    # 识别
                    label_name, label_type = processor.classify_object(c_points)

                    clusters.append(c_points)
                    labels_text.append(label_name)
                    bboxes.append({
                        "min": c_points.min(axis=0).tolist(),
                        "max": c_points.max(axis=0).tolist(),
                        "type": label_type  # 传给前端用于颜色显示
                    })

        # 再次下采样以减少 JSON 大小
        display_step = 2

        response = {
            "status": "success",
            "meta": {
                "processing_time": round(time.time() - start_time, 2),
                "total_points": len(points),
                "clusters": len(clusters)
            },
            "data": {
                "ground": ground[::display_step].tolist(),
                "clusters": [c[::2].tolist() for c in clusters],
                "labels": labels_text,
                "bboxes": bboxes
            }
        }

        logger.info(
            f"处理完成: {filename} | 耗时: {response['meta']['processing_time']}s | 识别: {len(clusters)} 个物体")
        return jsonify(response)

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    print("🚀 智能点云分析系统 V3.0 启动...")
    print(f"📂 数据存储: {SERVER_DATA_DIR}")
    print("🌍 访问地址: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)