import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
import time
import open3d as o3d

# 1. 读取PLY文件
def read_ply_file(filename):
    plydata = PlyData.read(filename)
    points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    return points

# 2. EAR算法
def ear_denoise(points, k=10, threshold=1.0):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    noise_indices = np.where(mean_distances > global_mean + threshold * global_std)[0]
    return noise_indices

# 3. WLOP算法
def wlop_denoise(points, r=0.1, iterations=10):
    tree = cKDTree(points)
    for _ in range(iterations):
        new_points = []
        for p in points:
            indices = tree.query_ball_point(p, r)
            if len(indices) > 1:
                neighbors = points[indices]
                weights = np.exp(-np.linalg.norm(neighbors - p, axis=1)**2 / (2 * (r / 3)**2))
                weighted_sum = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                new_p = weighted_sum / np.sum(weights)
                new_points.append(new_p)
            else:
                new_points.append(p)
        points = np.array(new_points)
    return points

# 4. Non-iterative算法
def non_iterative_denoise(points, r=0.1, min_neighbors=5):
    tree = cKDTree(points)
    noise_indices = [i for i, p in enumerate(points) if len(tree.query_ball_point(p, r)) < min_neighbors]
    return noise_indices

# 5. 引力特征函数算法
def calculate_centroid(points):
    return np.mean(points, axis=0)

def find_bounding_box(points):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    return min_coords, max_coords

def calculate_surface_area(min_coords, max_coords):
    l = max_coords[0] - min_coords[0]
    w = max_coords[1] - min_coords[1]
    h = max_coords[2] - min_coords[2]
    return 2 * (l * w + l * h + w * h)

def calculate_neighborhood_radius(S, n, k=1):
    V_avg = S / n
    r_avg = np.cbrt(3 * V_avg / (4 * np.pi))
    return k * r_avg

def find_neighborhoods(points, r):
    tree = cKDTree(points)
    return [len(tree.query_ball_point(p, r)) for p in points]

def calculate_distances(points, centroid):
    return np.linalg.norm(points - centroid, axis=1)

def calculate_gravity_values(neighborhoods, d_i, n, G=1):
    return [G * m * n / (d**2 + 1e-6) for m, d in zip(neighborhoods, d_i)]

def calculate_reference_values(average_m, d_i, n, G=1):
    return [G * average_m * n / (d**2 + 1e-6) for d in d_i]

def identify_sparse_outliers(F_i, F_ref_i, threshold_factor=2):
    diff = np.array(F_i) - np.array(F_ref_i)
    std_dev = np.std(diff)
    threshold = threshold_factor * std_dev
    return [i for i, f in enumerate(F_i) if f < F_ref_i[i] - threshold]

def gravity_denoise(points, k=1, threshold_factor=2):
    centroid = calculate_centroid(points)
    min_coords, max_coords = find_bounding_box(points)
    S = calculate_surface_area(min_coords, max_coords)
    n = len(points)
    r = calculate_neighborhood_radius(S, n, k)
    neighborhoods = find_neighborhoods(points, r)
    d_i = calculate_distances(points, centroid)
    F_i = calculate_gravity_values(neighborhoods, d_i, n)
    average_m = np.mean(neighborhoods)
    F_ref_i = calculate_reference_values(average_m, d_i, n)
    noise_indices = identify_sparse_outliers(F_i, F_ref_i, threshold_factor)
    return noise_indices

# 6. 可视化点云（使用Open3D）
def visualize_point_cloud(points, title, color=[0, 0, 1]):  # 默认蓝色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd], window_name=title)

def visualize_three_views(points, clean_points, noise_points):
    """
    分开显示三张图：
    - 去噪前：蓝色点云
    - 去噪后：蓝色点云
    - 噪声：红色点云
    """
    # 去噪前
    visualize_point_cloud(points, "Before Denoising", color=[0, 0, 1])  # 蓝色

    # 去噪后
    visualize_point_cloud(clean_points, "After Denoising", color=[0, 0, 1])  # 蓝色

    # 噪声
    if len(noise_points) > 0:
        visualize_point_cloud(noise_points, "Noise", color=[1, 0, 0])  # 红色
    else:
        print("没有噪声点")

# 7. GUI界面
class DenoiseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("点云去噪 - 可视化调节")
        self.points = None

        # 文件路径
        tk.Label(root, text="PLY文件路径:").grid(row=0, column=0, pady=5)
        self.file_entry = tk.Entry(root, width=50)
        self.file_entry.grid(row=0, column=1, pady=5)
        tk.Button(root, text="加载", command=self.load_file).grid(row=0, column=2, pady=5)

        # 算法选择
        tk.Label(root, text="选择算法:").grid(row=1, column=0, pady=5)
        self.algo_var = tk.StringVar()
        self.algo_combobox = ttk.Combobox(root, textvariable=self.algo_var, state='readonly')
        self.algo_combobox['values'] = ('EAR', 'WLOP', 'Non-iterative', 'Gravity Feature Function')
        self.algo_combobox.grid(row=1, column=1, pady=5)
        self.algo_combobox.bind('<<ComboboxSelected>>', self.update_params)

        # 参数面板
        self.param_frame = tk.Frame(root)
        self.param_frame.grid(row=2, column=0, columnspan=3, pady=10)

        # 运行按钮
        tk.Button(root, text="运行去噪", command=self.run_denoise).grid(row=3, column=1, pady=10)

    def load_file(self):
        filename = self.file_entry.get()
        try:
            self.points = read_ply_file(filename)
            print(f"从 {filename} 加载了 {len(self.points)} 个点")
        except Exception as e:
            print(f"加载文件失败: {e}")

    def update_params(self, event):
        algo = self.algo_var.get()
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        if algo == 'EAR':
            tk.Label(self.param_frame, text="k (邻居数):").grid(row=0, column=0)
            self.k_scale = tk.Scale(self.param_frame, from_=1, to=50, orient=tk.HORIZONTAL, resolution=1)
            self.k_scale.set(10)
            self.k_scale.grid(row=0, column=1)
            tk.Label(self.param_frame, text="threshold:").grid(row=1, column=0)
            self.threshold_scale = tk.Scale(self.param_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, resolution=0.1)
            self.threshold_scale.set(1.0)
            self.threshold_scale.grid(row=1, column=1)
        elif algo == 'WLOP':
            tk.Label(self.param_frame, text="r (半径):").grid(row=0, column=0)
            self.r_scale = tk.Scale(self.param_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL, resolution=0.01)
            self.r_scale.set(0.1)
            self.r_scale.grid(row=0, column=1)
            tk.Label(self.param_frame, text="iterations:").grid(row=1, column=0)
            self.iterations_scale = tk.Scale(self.param_frame, from_=1, to=50, orient=tk.HORIZONTAL, resolution=1)
            self.iterations_scale.set(10)
            self.iterations_scale.grid(row=1, column=1)
        elif algo == 'Non-iterative':
            tk.Label(self.param_frame, text="r (半径):").grid(row=0, column=0)
            self.r_scale = tk.Scale(self.param_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL, resolution=0.01)
            self.r_scale.set(0.1)
            self.r_scale.grid(row=0, column=1)
            tk.Label(self.param_frame, text="min_neighbors:").grid(row=1, column=0)
            self.min_neighbors_scale = tk.Scale(self.param_frame, from_=1, to=20, orient=tk.HORIZONTAL, resolution=1)
            self.min_neighbors_scale.set(5)
            self.min_neighbors_scale.grid(row=1, column=1)
        elif algo == 'Gravity Feature Function':
            tk.Label(self.param_frame, text="k (邻域半径系数):").grid(row=0, column=0)
            self.k_scale = tk.Scale(self.param_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, resolution=0.1)
            self.k_scale.set(1.0)
            self.k_scale.grid(row=0, column=1)
            tk.Label(self.param_frame, text="threshold_factor:").grid(row=1, column=0)
            self.threshold_factor_scale = tk.Scale(self.param_frame, from_=0.5, to=5.0, orient=tk.HORIZONTAL, resolution=0.1)
            self.threshold_factor_scale.set(2.0)
            self.threshold_factor_scale.grid(row=1, column=1)

    def run_denoise(self):
        if self.points is None:
            print("请先加载PLY文件")
            return

        algo = self.algo_var.get()
        start_time = time.time()

        # 执行去噪算法
        if algo == 'EAR':
            k = self.k_scale.get()
            threshold = self.threshold_scale.get()
            noise_indices = ear_denoise(self.points, k, threshold)
        elif algo == 'WLOP':
            r = self.r_scale.get()
            iterations = self.iterations_scale.get()
            points_denoised = wlop_denoise(self.points, r, iterations)
            noise_indices = []  # WLOP不直接提供噪声点
        elif algo == 'Non-iterative':
            r = self.r_scale.get()
            min_neighbors = self.min_neighbors_scale.get()
            noise_indices = non_iterative_denoise(self.points, r, min_neighbors)
        elif algo == 'Gravity Feature Function':
            k = self.k_scale.get()
            threshold_factor = self.threshold_factor_scale.get()
            noise_indices = gravity_denoise(self.points, k, threshold_factor)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # 准备三张图的数据
        if algo == 'WLOP':
            clean_points = points_denoised
            noise_points = np.array([])  # WLOP不提供噪声点
        else:
            clean_points = np.delete(self.points, noise_indices, axis=0)
            noise_points = self.points[noise_indices] if len(noise_indices) > 0 else np.array([])

        # 输出统计信息
        print(f"算法: {algo}")
        print(f"去噪前点数: {len(self.points)}")
        print(f"去噪后点数: {len(clean_points)}")
        print(f"运算时间: {elapsed_time:.4f} 秒")

        # 显示三张图
        visualize_three_views(self.points, clean_points, noise_points)

if __name__ == "__main__":
    root = tk.Tk()
    app = DenoiseGUI(root)
    root.mainloop()