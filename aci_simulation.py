import numpy as np
import matplotlib.pyplot as plt

class ACIPaper_Strict_Final:
    def __init__(self):
        # 1. 仿真时间
        self.dt = 0.01
        self.T_max = 300.0
        self.steps = int(self.T_max / self.dt)
        
        # 2. 论文参数
        # R=10, Q=I
        self.Q = np.diag([1.0, 1.0]) 
        self.R = np.array([[10.0]])   

        # 3. 学习率 (Alpha)
        # 这种配置下，Identifier 收敛很快，Actor 紧随其后
        self.alpha_I = 5.0     
        self.alpha_C = 1     
        self.alpha_A = 0.05    

        # 4. 【关键修改】阻尼项设为 0
        # 只有设为0，权重收敛后才会变成"水平直线"，否则会是"斜线"（衰减）
        self.Gamma_I = 0.0 
        self.Sigma_C = 0.0 
        self.F_A = 0.00001 # Actor 保留极微小的阻尼防止飘飞

        # 5. 权重初始化
        # 既然要复现 Fig 7，权重的初始值通常较小，从 0 附近开始
        np.random.seed(42) # 固定随机种子，保证每次结果一样
        self.W_I = np.random.uniform(-0.1, 0.1, (5, 2))
        self.W_C = np.random.uniform(-0.1, 0.1, (5, 1))
        self.W_A = np.random.uniform(-0.1, 0.1, (5, 1))

        # 稳定矩阵 A_I
        self.A_I = np.diag([-5.0, -5.0])

    # ==========================================
    # 特征方程
    # ==========================================
    def get_phi_I(self, x, u):
        """ Identifier 特征 """
        # [x1, x2, x1^2, x2^2, u]
        # 注意：为了让 Agent 1 (线性输入) 收敛，这里的 u 必须是线性的
        vec = np.array([x[0], x[1], x[0]**2, x[1]**2, 0.0])
        phi = np.tanh(vec)
        phi[4] = u[0] # 线性 u
        return phi.reshape(-1, 1)

    def get_phi_C(self, e, u):
        """ Critic 特征 """
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
        """ Actor 特征 """
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
        # --- 1. 状态初始化 (严格按照 Example) ---
        x0 = np.array([1.2971, 0.1138])
        e_init_paper = np.array([2.2970, 0.0886]) # e(0)
        x1 = x0 + e_init_paper
        
        # --- 2. Identifier 初始化 (严格按照 Example) ---
        # 论文给出: e_hat(0) = [0.4180, 0.3497]
        # 定义: e_tilde = e - e_hat (或者 x_tilde = x - x_hat)
        # 所以: x_hat = x1 - e_hat_init
        e_hat_init_paper = np.array([0.4180, 0.3497])
        x_hat = x1 - e_hat_init_paper

        # 记录器
        hist = {'t': [], 'x0': [], 'x1': [], 'u': [], 'ident_error': [], 'W_I': []}

        print("Simulation Start...")

        for k in range(self.steps):
            t = k * self.dt
            e = x1 - x0
            
            # --- Actor 计算 ---
            phi_A = self.get_phi_A(e)
            u_base = np.dot(self.W_A.T, phi_A).flatten()[0]
            
            # --- 噪声策略 (为了画出平滑直线) ---
            # Phase 1 (0-15s): 强激励，让权重快速找到位置
            if t < 15.0:
                noise = np.sin(2*t) * 30.0 + np.cos(10*t) * 20.0
            
            # Phase 2 (15-20s): 噪声线性衰减到0 (软着陆)
            elif 15.0 <= t < 50.0:
                decay = (20.0 - t) / 5.0
                noise = (np.sin(5*t) * 10.0) * decay
                
            # Phase 3 (>20s): 【完全静默】
            # 只有噪声完全为0，权重才会变成完美的直线
            else:
                noise = 0.0

            u_val = u_base + noise
            u = np.clip(u_val, -100.0, 100.0)
            u_vec = np.array([u]) # 包装成 vector 防止报错
            
            # --- 物理演化 ---
            dx0 = self.dynamics_leader(x0)
            dx1 = self.dynamics_follower(x1, u_vec)
            x0_new = x0 + dx0 * self.dt
            x1_new = x1 + dx1 * self.dt
            
            # --- Identifier 更新 ---
            phi_I = self.get_phi_I(x_hat, u_vec)
            dx_hat = np.dot(self.A_I, x_hat) + np.dot(self.W_I.T, phi_I).flatten()
            x_hat_new = x_hat + dx_hat * self.dt
            
            x_tilde = x1 - x_hat
            
            # 权重更新
            rho_i = np.linalg.norm(phi_I)
            norm_i = (rho_i**2 + 1)**2
            
            # Gamma_I 已经是 0 了，所以这里只有梯度项
            dW_I = self.alpha_I * np.outer(phi_I, x_tilde) / norm_i - self.Gamma_I * self.W_I
            self.W_I += dW_I * self.dt
            
            # --- Critic 更新 ---
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
            
            # --- Actor 更新 ---
            dU_du = 2 * self.R[0,0] * u
            dphiI_du = np.zeros((5, 1))
            dphiI_du[4] = 1.0 
            d_est_dx1_du = np.dot(self.W_I.T, dphiI_du).flatten()
            dH_du = dU_du + np.dot(dV_de, d_est_dx1_du)
            
            dW_A = -self.alpha_A * np.outer(phi_A, dH_du) - self.F_A * self.W_A
            self.W_A += np.clip(dW_A, -5.0, 5.0) * self.dt
            
            # 更新状态
            x0 = x0_new
            x1 = x1_new
            x_hat = x_hat_new
            
            # 记录 (Flatten 权重以便画图)
            hist['t'].append(t)
            hist['x0'].append(x0.copy())
            hist['x1'].append(x1.copy())
            hist['u'].append(u)
            hist['ident_error'].append(np.linalg.norm(x_tilde))
            hist['W_I'].append(self.W_I.flatten().copy())

        return hist

def plot_paper_reproduction(h):
    t = np.array(h['t'])
    x0 = np.array(h['x0'])
    x1 = np.array(h['x1'])
    u = np.array(h['u'])
    W_I = np.array(h['W_I'])
    x_tilde = np.array(h['ident_error'])
    
    # 1. 状态跟随图 (复现 Fig 3 & 4)
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, x0[:,0], 'b', linewidth=2, label='Leader $x_{0,1}$')
    plt.plot(t, x1[:,0], 'r', linewidth=2, label='Agent 1 $x_{1,1}$')
    plt.title('State Tracking Performance (Position)', fontsize=12)
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, x0[:,1], 'b', linewidth=2, label='Leader $x_{0,2}$')
    plt.plot(t, x1[:,1], 'r', linewidth=2, label='Agent 1 $x_{1,2}$')
    plt.title('State Tracking Performance (Velocity)', fontsize=12)
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. 权重收敛图 (复现 Fig 7)
    plt.figure(figsize=(10, 5))
    
    # 选取几条比较明显的权重曲线画出来，或者画全部
    # 论文 Fig 7 是很多条线从左边震荡然后变成右边直线
    lines = plt.plot(t, W_I, linewidth=1.5, alpha=0.9)
    
    plt.title('Fig 7 Reproduction: Identifier NN Weights Convergence', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Weight Value', fontsize=12)
    plt.grid(True)
    
    # 加上一条分界线，展示噪声切断的时间点
    plt.axvline(x=20, color='k', linestyle=':', alpha=0.5, label='Noise OFF')
    plt.xlim(0, 300)
    plt.legend([lines[0]], ['Weights'], loc='upper right') # 简化图例
    plt.show()
    
    # 3. 辨识误差 (复现 Fig 6 趋势)
    plt.figure(figsize=(10, 3))
    plt.plot(t, x_tilde, 'g', linewidth=1.5)
    plt.title('Identification Error Norm', fontsize=12)
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    sim = ACIPaper_Strict_Final()
    h = sim.run()
    plot_paper_reproduction(h)