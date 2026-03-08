import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def extract_features_v3(points):
    """
    [V3 终极版] 特征提取
    新增: XY平面长宽比 (Aspect Ratio) - 区分圆柱(人)和长条(车)的神器
    """
    if len(points) < 5: return np.zeros(9)  # 维度变为 9

    # 1. 几何特征
    cov_mat = np.cov(points[:, :3].T)
    eigen_vals = np.linalg.eigvalsh(cov_mat)
    idx = eigen_vals.argsort()[::-1]
    l1, l2, l3 = eigen_vals[idx]
    if l1 <= 0: l1 = 1e-6

    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    scattering = l3 / l1

    # 2. 物理尺寸
    min_p = np.min(points[:, :3], axis=0)
    max_p = np.max(points[:, :3], axis=0)

    dx = max_p[0] - min_p[0]
    dy = max_p[1] - min_p[1]
    dz = max_p[2] - min_p[2]

    max_span = max(dx, dy)
    min_span = min(dx, dy)

    # 3. [新增核心特征] 长宽比 (Aspect Ratio)
    # 人: 长宽差不多，ratio 接近 1
    # 车: 车长 >> 车宽，ratio 通常 > 2
    # 墙: 极长，ratio 极大
    aspect_ratio = max_span / (min_span + 1e-6)

    volume = dx * dy * dz
    density = len(points) / (volume + 1e-6)

    return np.array([
        linearity, planarity, scattering,
        dz, max_span, min_span,
        volume, density, aspect_ratio  # 新增第9维
    ])


def generate_realistic_data(n_samples=600):
    X = []
    y = []

    # --- 1. 行人 (Person) ---
    # 特征：圆柱体，长宽比接近 1
    for _ in range(n_samples):
        h = np.random.uniform(1.5, 1.9)
        r = np.random.uniform(0.2, 0.35)  # 半径很小
        n_pts = np.random.randint(30, 80)

        theta = np.random.uniform(0, 2 * np.pi, n_pts)
        z = np.random.uniform(0, h, n_pts)
        x = r * np.cos(theta) + np.random.normal(0, 0.03, n_pts)
        y_coord = r * np.sin(theta) + np.random.normal(0, 0.03, n_pts)

        X.append(extract_features_v3(np.column_stack((x, y_coord, z))))
        y.append("person")

    # --- 2. 车辆 (Car) - 模拟真实雷达的"L"形视角 ---
    # 这是一个巨大的改进：不再生成实心盒子
    for _ in range(n_samples):
        l = np.random.uniform(3.5, 5.0)  # 车长
        w = np.random.uniform(1.6, 2.0)  # 车宽
        h = np.random.uniform(1.4, 1.7)  # 车高
        n_pts = np.random.randint(100, 400)

        # 模拟雷达只扫到车的两个面 (L形)
        # 50%的点在侧面，50%的点在后面
        side_pts = np.random.uniform(0, 1, (n_pts // 2, 3)) * [l, 0.1, h]  # 侧面
        back_pts = np.random.uniform(0, 1, (n_pts // 2, 3)) * [0.1, w, h]  # 后面
        # 拼接并偏移
        points = np.vstack([side_pts, back_pts + [0, 0, 0]])  # 简单的 L 拼接

        X.append(extract_features_v3(points))
        y.append("car")

    # --- 3. 杆状物 (Pole) ---
    for _ in range(n_samples):
        h = np.random.uniform(2.5, 6.0)
        r = np.random.uniform(0.05, 0.15)  # 极细
        n_pts = np.random.randint(20, 60)
        z = np.random.uniform(0, h, n_pts)
        x = np.random.normal(0, r, n_pts)
        y_coord = np.random.normal(0, r, n_pts)
        X.append(extract_features_v3(np.column_stack((x, y_coord, z))))
        y.append("pole")

    # --- 4. 植被 (Tree) ---
    for _ in range(n_samples):
        scale = np.random.uniform(1.5, 4.0)
        n_pts = np.random.randint(50, 200)
        pts = np.random.normal(0, scale / 2, (n_pts, 3))
        X.append(extract_features_v3(pts))
        y.append("tree")

    # --- 5. 墙/建筑 (Building) ---
    for _ in range(n_samples):
        l = np.random.uniform(6.0, 15.0)  # 极长
        w = np.random.uniform(0.1, 0.2)  # 极薄
        h = np.random.uniform(2.0, 5.0)
        pts = np.random.uniform(0, 1, (300, 3)) * [l, w, h]
        X.append(extract_features_v3(pts))
        y.append("building")

    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("生成 V3 真实感数据 (L形车辆, 圆柱行人)...")
    X, y = generate_realistic_data()
    print(f"特征维度: {X.shape} (含长宽比)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    clf.fit(X_train, y_train)

    print("准确率评估:", clf.score(X_test, y_test))
    joblib.dump(clf, 'point_cloud_model.pkl')
    print("✅ V3 模型已保存")