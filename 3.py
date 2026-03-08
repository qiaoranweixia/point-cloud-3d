import numpy as np
import os


def create_box(center, size, n_points=200):
    """生成长方体点云 (模拟车、楼)"""
    x = np.random.uniform(center[0] - size[0] / 2, center[0] + size[0] / 2, n_points)
    y = np.random.uniform(center[1] - size[1] / 2, center[1] + size[1] / 2, n_points)
    z = np.random.uniform(center[2] - size[2] / 2, center[2] + size[2] / 2, n_points)
    return np.column_stack((x, y, z))


def create_cylinder(center, radius, height, n_points=100):
    """生成圆柱体点云 (模拟人、树干)"""
    angle = np.random.uniform(0, 2 * np.pi, n_points)
    r = np.random.uniform(0, radius, n_points)
    x = center[0] + r * np.cos(angle)
    y = center[1] + r * np.sin(angle)
    z = np.random.uniform(center[2], center[2] + height, n_points)
    return np.column_stack((x, y, z))


def create_tree_crown(center, radius, n_points=300):
    """生成球体点云 (模拟树冠)"""
    vec = np.random.randn(n_points, 3)
    vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
    points = center + vec * (np.random.rand(n_points, 1) ** (1 / 3)) * radius
    return points


def generate_street_scene():
    points = []

    # 1. 地面 (Ground) - 50m x 50m 的马路
    x = np.linspace(-25, 25, 400)
    y = np.linspace(-25, 25, 400)
    xx, yy = np.meshgrid(x, y)
    ground_z = np.random.normal(0, 0.02, xx.shape)  # 加一点点噪声
    ground = np.column_stack((xx.ravel(), yy.ravel(), ground_z.ravel()))
    points.append(ground)

    # 2. 车辆 (Cars) - 尺寸: 4.5m x 1.8m x 1.5m
    car_positions = [
        (5, 5, 0.75), (12, 5, 0.75), (-8, 5, 0.75),  # 右车道
        (0, -5, 0.75), (-15, -5, 0.75)  # 左车道
    ]
    for pos in car_positions:
        # 车身
        car = create_box(pos, (4.5, 1.8, 1.5), 800)
        points.append(car)

    # 3. 行人 (People) - 尺寸: 0.5m x 0.5m x 1.7m
    person_positions = [
        (8, 10, 0), (9, 11, 0), (-5, 12, 0), (-5.5, 11.5, 0)
    ]
    for pos in person_positions:
        person = create_cylinder(pos, 0.25, 1.75, 150)
        points.append(person)

    # 4. 建筑物 (Buildings) - 尺寸巨大
    # 模拟街道两侧的高楼
    building_positions = [
        (-20, 15, 5), (0, 15, 8), (20, 15, 6),  # 上方一排楼
        (-15, -15, 7), (15, -15, 5)  # 下方一排楼
    ]
    for pos in building_positions:
        # 楼比较大，点要多一点
        w = np.random.uniform(8, 12)
        h = pos[2] * 2  # 高度
        building = create_box(pos, (w, 8, h), 3000)
        points.append(building)

    # 5. 行道树 (Trees)
    tree_x = np.linspace(-20, 20, 8)
    for x in tree_x:
        # 树干
        trunk = create_cylinder((x, 8, 0), 0.2, 2.0, 100)
        # 树冠
        crown = create_tree_crown((x, 8, 2.5), 1.5, 300)
        points.append(trunk)
        points.append(crown)

    # 合并数据
    all_points = np.vstack(points).astype(np.float32)

    # 保存
    os.makedirs('server_data', exist_ok=True)
    file_path = 'server_data/street_scene_sim.bin'

    # 补齐intensity列 (Nx3 -> Nx4)
    data_with_intensity = np.hstack([all_points, np.zeros((len(all_points), 1))]).astype(np.float32)
    data_with_intensity.tofile(file_path)

    print(f"✅ 生成成功: {file_path}")
    print(f"包含: {len(car_positions)}辆车, {len(person_positions)}个人, {len(building_positions)}栋楼")


if __name__ == '__main__':
    generate_street_scene()