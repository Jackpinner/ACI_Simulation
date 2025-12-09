import numpy as np
import matplotlib.pyplot as plt

class ACIPaper_LongTermStable:
    def __init__(self):
        # 1. 仿真时间
        self.dt = 0.01
        self.T_max = 100.0 # 跑满 300秒
        self.steps = int(self.T_max / self.dt)
        
        # 2. 论文参数
        self.Q = np.diag([1.0, 1.0]) 
        self.R = np.array([[10.0]])   

        # 3. 学习率 (Alpha)
        # 为了长时稳定，Identifier 稍微降一点，Actor 保持稳健
        self.alpha_I = 5.0    
        self.alpha_C = 1.0     
        self.alpha_A = 0.05    

        # 4. 阻尼项
        # 依然保持 0，为了复现论文的"直线"效果
        # 我们靠"维稳噪声"来防止漂移，而不是靠阻尼
        self.Gamma_I = 0.0 
        self.Sigma_C = 0.0 
        self.F_A = 0.00001 

        # 5. 权重初始化
        np.random.seed(42)
        self.W_I = np.random.uniform(-0.1, 0.1, (5, 2))
        self.W_C = np.random.uniform(-0.1, 0.1, (5, 1))
        self.W_A = np.random.uniform(-0.1, 0.1, (5, 1))

        # 稳定矩阵
        self.A_I = np.diag([-5.0, -5.0])

    # ==========================================
    # 特征方程 (保持不变)
    # ==========================================
    def get_phi_I(self, x, u):
        vec = np.array([x[0], x[1], x[0]**2, x[1]**2, 0.0])
        phi = np.tanh(vec)
        phi[4] = u[0] 
        return phi.reshape(-1, 1)

    def get_phi_C(self, e, u):
        vec = np.array([e[0]**2, e[0]*e[1], e[1]**2, u[0]**2, e[0]])
        return vec.reshape(-1, 1)
        
    def get_grad_phi_C_e(self, e):
        jac = np.zeros((5, 2))
        jac[0, :] = [2*e[0], 0]
        jac[1, :] = [e[1], e[0]]
        jac[2, :] = [0, 2*e[1]]
        jac[3, :] = [0, 0]
        jac[4, :] = [1, 0]
        return jac

    def get_phi_A(self, e):
        vec = np.array([e[0], e[1], e[0]**2, e[1]**2, e[0]*e[1]])
        return vec.reshape(-1, 1)

    # ==========================================
    # 动力学
    # ==========================================
    def dynamics_leader(self, x):
        return np.array([x[1], - (x[0]**2) * x[1] - x[0]])

    def dynamics_follower(self, x, u):
        return np.array([x[1], -5 * x[0] - 0.5 * x[1] +  u[0]])

    # ==========================================
    # 主循环
    # ==========================================
    def run(self):
        # 初始化
        x0 = np.array([1.2971, 0.1138])
        e_init_paper = np.array([2.2970, 0.0886]) 
        x1 = x0 + e_init_paper
        e_hat_init_paper = np.array([0.4180, 0.3497])
        x_hat = x1 - e_hat_init_paper

        hist = {'t': [], 'x0': [], 'x1': [], 'u': [], 'ident_error': [], 'W_I': []}

        print("Simulation Start (Long Term Stable)...")

        for k in range(self.steps):
            t = k * self.dt
            e = x1 - x0
            
            # --- Actor ---
            phi_A = self.get_phi_A(e)
            u_base = np.dot(self.W_A.T, phi_A).flatten()[0]
            
            # --- 关键修改：噪声策略 ---
            if t < 15.0:
                # Phase 1: 强激励 (Burn-in)
                noise = np.sin(2*t) * 30.0 + np.cos(10*t) * 20.0
            
            elif 15.0 <= t < 50.0:
                # Phase 2: 线性衰减
                decay = (50.0 - t) / 35.0
                noise = (np.sin(5*t) * 10.0) * decay
                
            else:
                # Phase 3: 【维稳噪声】 (Tiny Maintenance Noise)
                # 不要设为 0.0！给一个极小的随机扰动。
                # 这个扰动肉眼看不见，但足够锁住 Identifier 的权重，防止漂移。
                noise = np.random.randn() * 0.2 

            u_val = u_base + noise
            u = np.clip(u_val, -100.0, 100.0)
            u_vec = np.array([u])
            
            # --- Physics ---
            dx0 = self.dynamics_leader(x0)
            dx1 = self.dynamics_follower(x1, u_vec)
            x0_new = x0 + dx0 * self.dt
            x1_new = x1 + dx1 * self.dt
            
            # --- Identifier ---
            phi_I = self.get_phi_I(x_hat, u_vec)
            dx_hat = np.dot(self.A_I, x_hat) + np.dot(self.W_I.T, phi_I).flatten()
            x_hat_new = x_hat + dx_hat * self.dt
            
            x_tilde = x1 - x_hat
            
            # Update W_I
            rho_i = np.linalg.norm(phi_I)
            norm_i = (rho_i**2 + 1)**2
            dW_I = self.alpha_I * np.outer(phi_I, x_tilde) / norm_i - self.Gamma_I * self.W_I
            self.W_I += dW_I * self.dt
            
            # --- Critic ---
            phi_I_real = self.get_phi_I(x1, u_vec)
            est_dx1 = np.dot(self.A_I, x1) + np.dot(self.W_I.T, phi_I_real).flatten()
            est_de = est_dx1 - dx0
            
            phi_C = self.get_phi_C(e, u_vec)
            dV_de = np.dot(self.W_C.T, self.get_grad_phi_C_e(e)).flatten()
            
            utility = np.dot(e.T, np.dot(self.Q, e)) + np.dot(u_vec.T, np.dot(self.R, u_vec))
            ec = np.dot(dV_de, est_de) + utility
            
            rho_c = np.linalg.norm(phi_C)
            norm_c = (rho_c**2 + 1)**2
            dW_C = -self.alpha_C * (phi_C * ec) / norm_c - self.Sigma_C * self.W_C
            self.W_C += np.clip(dW_C, -5.0, 5.0) * self.dt
            
            # --- Actor ---
            dU_du = 2 * self.R[0,0] * u
            dphiI_du = np.zeros((5, 1))
            dphiI_du[4] = 1.0 
            d_est_dx1_du = np.dot(self.W_I.T, dphiI_du).flatten()
            dH_du = dU_du + np.dot(dV_de, d_est_dx1_du)
            
            dW_A = -self.alpha_A * np.outer(phi_A, dH_du) - self.F_A * self.W_A
            self.W_A += np.clip(dW_A, -5.0, 5.0) * self.dt
            
            # State Update
            x0 = x0_new
            x1 = x1_new
            x_hat = x_hat_new
            
            hist['t'].append(t)
            hist['x0'].append(x0.copy())
            hist['x1'].append(x1.copy())
            hist['u'].append(u)
            hist['ident_error'].append(np.linalg.norm(x_tilde))
            hist['W_I'].append(self.W_I.flatten().copy())

        return hist

def plot_final(h):
    t = np.array(h['t'])
    x0 = np.array(h['x0'])
    x1 = np.array(h['x1'])
    W_I = np.array(h['W_I'])
    
    plt.figure(figsize=(10, 8))
    
    # 1. 位置跟随 (验证长时稳定性)
    plt.subplot(3, 1, 1)
    plt.plot(t, x0[:,0], 'b', linewidth=1.5, label='Leader')
    plt.plot(t, x1[:,0], 'r', linewidth=1.5, label='Agent 1')
    plt.title('Position Tracking (300s Stable Test)')
    plt.grid(True)
    plt.legend()

    # 2. 速度跟随
    plt.subplot(3, 1, 2)
    plt.plot(t, x0[:,1], 'b', linewidth=1.5, label='Leader')
    plt.plot(t, x1[:,1], 'r', linewidth=1.5, label='Agent 1')
    plt.title('Velocity Tracking')
    plt.grid(True)
    plt.legend()
    
    # 3. 权重收敛 (验证直线)
    plt.subplot(3, 1, 3)
    for i in range(W_I.shape[1]):
        plt.plot(t, W_I[:, i], linewidth=1.5, alpha=0.9)
    plt.title('Identifier Weights (Should be flat and stable)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sim = ACIPaper_LongTermStable()
    h = sim.run()
    plot_final(h)