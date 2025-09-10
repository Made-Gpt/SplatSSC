import torch
import matplotlib.pyplot as plt

# 假设 new_anchor_num = 10000，且 anchor.device 为 CPU（测试时）
new_anchor_num = 10000
anchor_device = torch.device("cpu")
anchor_xyz_noise = torch.randn(new_anchor_num, 3, dtype=torch.float, device=anchor_device) * 0.08  # 生成噪声
anchor_xyz_noise = torch.clamp(anchor_xyz_noise, -0.2, 0.2)  # 限制范围在 -0.1 到 0.1

# 将噪声数据展平并转换为 numpy 数组用于绘图
noise_np = anchor_xyz_noise.cpu().numpy().flatten()

plt.figure(figsize=(8, 4))
plt.hist(noise_np, bins=50, density=True, alpha=0.75, color='blue')
plt.title("Clamped Gaussian Noise Distribution")
plt.xlabel("Noise value")
plt.ylabel("Density")
plt.grid(True)
plt.show()

