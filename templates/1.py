from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os
import tempfile
import random
import logging
from sklearn.cluster import KMeans, DBSCAN

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 启用CORS支持

# 配置日志
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.DEBUG)


def read_velodyne_bin(path):
    """高效读取点云文件"""
    try:
        data = np.fromfile(path, dtype=np.float32)
        return data.reshape(-1, 4)[:, :3]  # 提取XYZ坐标
    except Exception as e:
        app.logger.error(f"文件读取失败: {str(e)}")
        raise


def ground_segmentation(data, max_iter=500, sigma=0.15):
    """改进的RANSAC地面分割"""
    best_inliers = -1
    best_model = (np.zeros(3), 0.0)

    for _ in range(max_iter):
        try:
            # 随机采样三个点
            samples = data[random.sample(range(len(data)), 3)]
            vec1 = samples[1] - samples[0]
            vec2 = samples[2] - samples[0]
            normal = np.cross(vec1, vec2)
            norm = np.linalg.norm(normal)

            if norm < 1e-6:
                continue  # 避免零向量

            normal /= norm
            d = -np.dot(normal, samples[0])
            distances = np.abs(np.dot(data, normal) + d)
            inliers = np.sum(distances < sigma)

            if inliers > best_inliers:
                best_inliers = inliers
                best_model = (normal, d)
        except Exception as e:
            app.logger.warning(f"迭代异常: {str(e)}")
            continue

    in_mask = np.dot(data, best_model[0]) + best_model[1] < sigma
    return data[in_mask], data[~in_mask]


@app.route('/process', methods=['POST'])
def process_pointcloud():
    """点云处理主端点"""
    app.logger.info("收到点云处理请求")

    # 验证文件存在
    if 'file' not in request.files:
        return jsonify(error="未上传文件"), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="无效文件名"), 400

    try:
        # 创建临时文件
        fd, temp_path = tempfile.mkstemp(suffix='.bin')
        os.close(fd)
        file.save(temp_path)
        app.logger.debug(f"临时文件保存至: {temp_path}")

        # 读取点云
        points = read_velodyne_bin(temp_path)
        app.logger.info(f"成功读取点云，点数: {len(points)}")

        # 地面分割
        ground, objects = ground_segmentation(points)
        app.logger.info(f"地面点: {len(ground)}, 目标点: {len(objects)}")

        # 获取算法参数
        algorithm = request.form.get('algorithm', 'kmeans')
        n_clusters = int(request.form.get('n_clusters', 5))
        eps = float(request.form.get('eps', 0.5))
        min_samples = int(request.form.get('min_samples', 10))

        # 执行聚类
        if algorithm == 'dbscan':
            cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(objects)
        else:
            cluster = KMeans(n_clusters=n_clusters).fit(objects)

        labels = cluster.labels_
        clusters = [objects[labels == i] for i in set(labels) if i != -1]
        app.logger.info(f"生成 {len(clusters)} 个有效聚类")

        # 构建响应
        return jsonify({
            "original": points[::10].tolist(),  # 原始点云降采样
            "ground": ground[::5].tolist(),  # 地面点降采样
            "clusters": [
                c[::2].tolist() for c in clusters
                if len(c) > 10  # 过滤小聚类
            ]
        })

    except Exception as e:
        app.logger.error(f"处理失败: {str(e)}", exc_info=True)
        return jsonify(error=str(e)), 500
    finally:
        # 清理临时文件
        if 'temp_path' in locals():
            try:
                os.remove(temp_path)
                app.logger.debug(f"已删除临时文件: {temp_path}")
            except Exception as e:
                app.logger.warning(f"文件删除失败: {str(e)}")


@app.route('/')
def index():
    """主页面路由"""
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """静态文件路由"""
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)