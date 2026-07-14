import numpy as np
import matplotlib.pyplot as plt
import time

class EKF_Standard:
    def __init__(self, x0, P0, Q, R):
        self.x = x0.reshape(4, 1)  
        self.P = P0                
        self.Q = Q                 
        self.R = R                 
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.I = np.eye(4)

    def predict(self, u, dt):
        v, theta = self.x[2, 0], self.x[3, 0]
        a, w = u[0], u[1]

        self.x[0, 0] += v * np.cos(theta) * dt
        self.x[1, 0] += v * np.sin(theta) * dt
        self.x[2, 0] += a * dt
        self.x[3, 0] += w * dt

        F = np.eye(4)
        F[0, 2] = np.cos(theta) * dt
        F[0, 3] = -v * np.sin(theta) * dt
        F[1, 2] = np.sin(theta) * dt
        F[1, 3] = v * np.cos(theta) * dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z = z.reshape(2, 1)
        
        y = z - self.H @ self.x
        
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P


class EKF_FirmwareOptimized:
    def __init__(self, x0, P0, Q, R):
        self.x = x0.flatten()      
        self.P = P0                
        self.Q = Q
        
        self.R_x = R[0, 0]
        self.R_y = R[1, 1]
        
        self.SCALE = 1 << 16  

    def float_to_fp(self, val):
        return int(val * self.SCALE)

    def predict_fixed_point(self, u, dt):
        px, py, v, theta = self.float_to_fp(self.x[0]), self.float_to_fp(self.x[1]), \
                           self.float_to_fp(self.x[2]), self.float_to_fp(self.x[3])
        a = self.float_to_fp(u[0])
        w = self.float_to_fp(u[1])
        dt_fp = self.float_to_fp(dt)

        cos_theta = self.float_to_fp(np.cos(self.x[3])) 
        sin_theta = self.float_to_fp(np.sin(self.x[3]))

        v_dt = (v * dt_fp) >> 16
        px += (v_dt * cos_theta) >> 16
        py += (v_dt * sin_theta) >> 16
        v  += (a * dt_fp) >> 16
        theta += (w * dt_fp) >> 16

        self.x = np.array([px, py, v, theta], dtype=float) / self.SCALE

        v_float, theta_float = self.x[2], self.x[3]
        F = np.array([
            [1, 0, np.cos(theta_float)*dt, -v_float*np.sin(theta_float)*dt],
            [0, 1, np.sin(theta_float)*dt,  v_float*np.cos(theta_float)*dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.P = F @ self.P @ F.T + self.Q

    def update_optimized(self, z):
        y_x = z[0] - self.x[0]
        y_y = z[1] - self.x[1]

        S_x = self.P[0, 0] + self.R_x
        S_y = self.P[1, 1] + self.R_y

        inv_S_x = 1.0 / S_x
        inv_S_y = 1.0 / S_y

        K = np.zeros((4, 2))
        for i in range(4):
            K[i, 0] = self.P[i, 0] * inv_S_x
            K[i, 1] = self.P[i, 1] * inv_S_y

        for i in range(4):
            self.x[i] += K[i, 0] * y_x + K[i, 1] * y_y

        P_new = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                P_new[i, j] = self.P[i, j] - (K[i, 0] * self.P[0, j] + K[i, 1] * self.P[1, j])
        self.P = P_new

dt = 0.01          
time_steps = 2000  
t_space = np.arange(0, time_steps * dt, dt)

sigma_gps = 2.0    
sigma_acc = 0.5    
sigma_yaw = 0.1    

P0 = np.eye(4) * 1.0
Q = np.diag([0.1, 0.1, 0.5, 0.1]) * dt  
R = np.diag([sigma_gps**2, sigma_gps**2]) 
x0 = np.array([0, 0, 2.0, 0])             

ekf_std = EKF_Standard(x0.copy(), P0.copy(), Q, R)
ekf_opt = EKF_FirmwareOptimized(x0.copy(), P0.copy(), Q, R)

x_true_history = []
gps_measurements = []
u_measurements = []

x_curr = x0.copy().reshape(4, 1)
print("正在生成無人機 8 字型軌跡模擬資料...")
for i, t in enumerate(t_space):
    u_true = np.array([0.0, np.sin(t * 0.5)]) 
    
    v, theta = x_curr[2, 0], x_curr[3, 0]
    x_curr[0, 0] += v * np.cos(theta) * dt
    x_curr[1, 0] += v * np.sin(theta) * dt
    x_curr[2, 0] += u_true[0] * dt
    x_curr[3, 0] += u_true[1] * dt
    x_true_history.append(x_curr.flatten())
    
    u_noisy = u_true + np.array([np.random.normal(0, sigma_acc), 
                                 np.random.normal(0, sigma_yaw)])
    u_measurements.append(u_noisy)
    
    if i % 10 == 0:
        z_gps = np.array([x_curr[0,0] + np.random.normal(0, sigma_gps),
                          x_curr[1,0] + np.random.normal(0, sigma_gps)])
        gps_measurements.append((i, z_gps))

history_std = []
history_opt = []

time_std_total = 0.0
time_opt_total = 0.0

gps_idx = 0
print("正在執行 EKF 追蹤與效能分析...")
for i in range(time_steps):
    u = u_measurements[i]
    has_gps = (gps_idx < len(gps_measurements) and gps_measurements[gps_idx][0] == i)
    if has_gps:
        z = gps_measurements[gps_idx][1]
        gps_idx += 1

    t_start = time.perf_counter()
    ekf_std.predict(u, dt)
    if has_gps:
        ekf_std.update(z)
    time_std_total += (time.perf_counter() - t_start)
    history_std.append(ekf_std.x.flatten())

    t_start = time.perf_counter()
    ekf_opt.predict_fixed_point(u, dt)
    if has_gps:
        ekf_opt.update_optimized(z)
    time_opt_total += (time.perf_counter() - t_start)
    history_opt.append(ekf_opt.x.copy())

x_true_history = np.array(x_true_history)
history_std = np.array(history_std)
history_opt = np.array(history_opt)
gps_x = [z[1][0] for z in gps_measurements]
gps_y = [z[1][1] for z in gps_measurements]

print("繪製圖表中...")
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121)
ax1.plot(x_true_history[:, 0], x_true_history[:, 1], 'k--', label="Ground Truth", linewidth=2)
ax1.scatter(gps_x, gps_y, c='r', s=10, alpha=0.5, label="Noisy GPS")
ax1.plot(history_std[:, 0], history_std[:, 1], 'b-', alpha=0.7, label="Standard EKF", linewidth=4)
ax1.plot(history_opt[:, 0], history_opt[:, 1], 'y--', label="Optimized EKF", linewidth=2)
ax1.set_title("2D Trajectory Tracking: Standard vs Optimized", fontsize=14)
ax1.set_xlabel("X Position (m)")
ax1.set_ylabel("Y Position (m)")
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(122)
categories = ['Standard (Matrix + Inverse)', 'Optimized (Scalar + Bit-shift)']
times = [time_std_total * 1000, time_opt_total * 1000] 
bars = ax2.bar(categories, times, color=['blue', 'orange'], alpha=0.8)
ax2.set_title("Total Computation Time for 2000 Iterations", fontsize=14)
ax2.set_ylabel("Time (ms)")

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(times)*0.01), f'{yval:.2f} ms', ha='center', va='bottom', fontsize=12, fontweight='bold')

speedup = time_std_total / time_opt_total
ax2.text(0.5, max(times)*0.8, f"Speedup: {speedup:.2f}x Faster!", ha='center', fontsize=14, color='red', fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.show()