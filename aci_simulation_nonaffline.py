import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 通用 ACI 智能体类 (封装了算法核心)
# ==========================================
class ACI_Node:
    def __init__(self, name, x0_init, e_init, e_hat_init, dynamics_func, input_gain=1.0):
        self.name = name
        self.dt = 0.01
        
        # 1. 物理参数
        self.input_gain = input_gain # 控制输入增益 (Agent1=0.01, Agent2=1.0)
        self.dynamics_func = dynamics_func # 自身的动力学函数
        
        # 2. 状态初始化
        self.x = x0_init + e_init
        self.x_hat = self.x - e_hat_init # 根据 e_hat 反推 x_hat
        
        # 3. 学习参数 (通用配置)
        self.Q = np.diag([1.0, 1.0])
        # 如果增益大(Agent2)，R可以大一点；增益小(Agent1)，R大就是找死
        # 这里为了统一对比，我们根据 gain 自动调整 R，保证 Actor 敢输出
        if input_gain < 0.1:
            self.R = np.array([[10.0]])   # Agent 1 (困难模式)
            self.alpha_A = 0.05
        else:
            self.R = np.array([[1.0]])    # Agent 2 (正常模式)
            self.alpha_A = 0.05           # 增益大，学习率要小，防止震荡

        self.alpha_I = 5.0
        self.alpha_C = 1.0
        
        # 阻尼项 (设为0以复现直线收敛)
        self.Gamma_I = 0.0
        self.Sigma_C = 0.0
        self.F_A = 0.001

        # 4. 权重初始化
        np.random.seed(42 if name=='Agent1' else 2023) # 不同的随机种子
        self.W_I = np.random.uniform(-0.1, 0.1, (5, 2))
        self.W_C = np.random.uniform(-0.1, 0.1, (5, 1))
        self.W_A = np.random.uniform(-0.1, 0.1, (5, 1))
        
        self.A_I = np.diag([-5.0, -5.0])

    # --- 特征函数 (通用) ---
    def get_phi_I(self, x, u):
        # [x1, x2, x1^2, x2^2, u]
        vec = np.array([x[0], x[1], x[0]**2, x[1]**2, 0.0])
        phi = np.tanh(vec)
        phi[4] = u[0] # 线性 u
        return phi.reshape(-1, 1)

    def get_phi_C(self, e, u):
        vec = np.array([e[0]**2, e[0]*e[1], e[1]**2, u[0]**2, e[0]])
        return vec.reshape(-1, 1)
    
    def get_grad_phi_C_e(self, e):
        jac = np.zeros((5, 2))
        jac[0,:]=[2*e[0],0]; jac[1,:]=[e[1],e[0]]; jac[2,:]=[0,2*e[1]]; jac[3,:]=[0,0]; jac[4,:]=[1,0]
        return jac

    def get_phi_A(self, e):
        vec = np.array([e[0], e[1], e[0]**2, e[1]**2, e[0]*e[1]])
        return vec.reshape(-1, 1)

    # --- 单步更新 (One Step Update) ---
    def update(self, t, x_leader, dx_leader):
        e = self.x - x_leader
        
        # 1. Actor 计算 u
        phi_A = self.get_phi_A(e)
        u_base = np.dot(self.W_A.T, phi_A).flatten()[0]
        
        # 噪声策略 (Burn-in + Maintenance)
        if t < 15.0:
            # 前期强激励
            noise = np.sin(2*t)*30.0 + np.cos(10*t)*20.0
            # Agent 2 增益大，噪声不需要那么大，稍微缩放一下
            if self.input_gain > 0.5: noise *= 0.2 
        elif 15.0 <= t < 40.0:
            decay = (40.0 - t) / 25.0
            noise = (np.sin(5*t) * 10.0) * decay
            if self.input_gain > 0.5: noise *= 0.2
        else:
            # 维稳噪声
            noise = np.random.randn() * 0.05
        
        u_val = u_base + noise
        u = np.clip(u_val, -100.0, 100.0)
        u_vec = np.array([u])

        # 2. 物理演化
        dx = self.dynamics_func(self.x, u_vec) # 使用传入的动力学函数
        self.x = self.x + dx * self.dt
        
        # 3. Identifier 更新
        phi_I = self.get_phi_I(self.x_hat, u_vec)
        dx_hat = np.dot(self.A_I, self.x_hat) + np.dot(self.W_I.T, phi_I).flatten()
        self.x_hat = self.x_hat + dx_hat * self.dt
        
        x_tilde = self.x - self.x_hat
        rho_i = np.linalg.norm(phi_I)
        norm_i = (rho_i**2 + 1)**2
        dW_I = self.alpha_I * np.outer(phi_I, x_tilde) / norm_i - self.Gamma_I * self.W_I
        self.W_I += dW_I * self.dt
        
        # 4. Critic 更新
        phi_I_real = self.get_phi_I(self.x, u_vec)
        est_dx = np.dot(self.A_I, self.x) + np.dot(self.W_I.T, phi_I_real).flatten()
        est_de = est_dx - dx_leader
        
        phi_C = self.get_phi_C(e, u_vec)
        dV_de = np.dot(self.W_C.T, self.get_grad_phi_C_e(e)).flatten()
        utility = np.dot(e.T, np.dot(self.Q, e)) + np.dot(u_vec.T, np.dot(self.R, u_vec))
        ec = np.dot(dV_de, est_de) + utility
        
        rho_c = np.linalg.norm(phi_C)
        norm_c = (rho_c**2 + 1)**2
        dW_C = -self.alpha_C * (phi_C * ec) / norm_c - self.Sigma_C * self.W_C
        self.W_C += np.clip(dW_C, -5, 5) * self.dt
        
        # 5. Actor 更新
        dU_du = 2 * self.R[0,0] * u
        dphiI_du = np.zeros((5, 1)); dphiI_du[4] = 1.0
        d_est_dx_du = np.dot(self.W_I.T, dphiI_du).flatten()
        dH_du = dU_du + np.dot(dV_de, d_est_dx_du)
        
        dW_A = -self.alpha_A * np.outer(phi_A, dH_du) - self.F_A * self.W_A
        self.W_A += np.clip(dW_A, -5, 5) * self.dt

        return self.x.copy(), u, self.W_I.flatten().copy()

# ==========================================
# 仿真主程序
# ==========================================
class MultiAgentSim:
    def __init__(self):
        self.dt = 0.01
        self.T_max = 75.0
        self.steps = int(self.T_max / self.dt)
        
        # Leader Initial (Shared)
        self.x0 = np.array([1.2971, 0.1138])

        # --- Agent 1 配置 (论文原版困难户) ---
        # 0.01u 输入, 初始误差大
        def dyn_agent1(x, u):
            # dot_x1 = x2
            # dot_x2 = -5x1 - 0.5x2 + 0.01u
            return np.array([x[1], -5*x[0] - 0.5*x[1] + 1.0*u[0]])
            
        self.agent1 = ACI_Node(
            name="Agent 1",
            x0_init=self.x0, 
            e_init=np.array([2.2970, 0.0886]), 
            e_hat_init=np.array([0.4180, 0.3497]), # 反推 x_hat 用
            dynamics_func=dyn_agent1,
            input_gain=0.01 # 弱增益
        )

        # --- Agent 2 配置 (论文参考 + 你的修改) ---
        # 1.0u 输入 (去掉0.01), 初始误差小, 动力学不同
        def dyn_agent2(x, u):
            # dot_x1 = 2 * x2 (注意这里系数是 2, 参考论文 Agent 2 结构)
            # dot_x2 = -5x1 - 0.5x2 + 1.0*u (去掉了 0.01)
            return np.array([2.0*x[1], -5*x[0] - 0.5*x[1] + 1.0*u[0]])
            
        self.agent2 = ACI_Node(
            name="Agent 2",
            x0_init=self.x0,
            e_init=np.array([0.0330, 0.5104]), # 论文 Page 7 Agent 2 数据
            e_hat_init=np.array([0.4892, 0.5274]), # 论文 Page 7 Agent 2 数据
            dynamics_func=dyn_agent2,
            input_gain=1.0 # 强增益
        )

    def dynamics_leader(self, x):
        return np.array([x[1], - (x[0]**2) * x[1] - x[0]])

    def run(self):
        # 记录器
        hist = {
            't': [], 
            'x0': [], 
            'x1': [], 'u1': [], 'w1': [],
            'x2': [], 'u2': [], 'w2': []
        }
        
        print("Starting Parallel Simulation (Agent 1 vs Agent 2)...")
        
        x0_curr = self.x0.copy()
        
        for k in range(self.steps):
            t = k * self.dt
            
            # 1. Leader Update
            dx0 = self.dynamics_leader(x0_curr)
            x0_next = x0_curr + dx0 * self.dt
            
            # 2. Agent 1 Update (Weak Input)
            x1_new, u1, w1 = self.agent1.update(t, x0_curr, dx0)
            
            # 3. Agent 2 Update (Strong Input)
            x2_new, u2, w2 = self.agent2.update(t, x0_curr, dx0)
            
            # Record
            hist['t'].append(t)
            hist['x0'].append(x0_curr.copy())
            hist['x1'].append(x1_new)
            hist['u1'].append(u1)
            hist['w1'].append(w1)
            hist['x2'].append(x2_new)
            hist['u2'].append(u2)
            hist['w2'].append(w2)
            
            # Step
            x0_curr = x0_next
            
        return hist

def plot_comparison(h):
    t = np.array(h['t'])
    x0 = np.array(h['x0'])
    x1 = np.array(h['x1'])
    x2 = np.array(h['x2'])
    u1 = np.array(h['u1'])
    u2 = np.array(h['u2'])
    
    # 1. 位置跟随对比
    plt.figure(figsize=(12, 5))
    plt.plot(t, x0[:,0], 'k', linewidth=2, alpha=0.6, label='Leader $x_0$')
    plt.plot(t, x1[:,0], 'r--', linewidth=1.5, label='Agent 1 ')
    plt.plot(t, x2[:,0], 'b-.', linewidth=1.5, label='Agent 2 ')
    plt.title('Position Tracking Comparison', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. 控制量对比 (注意量级差异)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, u1, 'r')
    plt.title('Agent 1 Control Input (Weak Gain)')
    plt.ylabel('u1 (Large Magnitude)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(t, u2, 'b')
    plt.title('Agent 2 Control Input (Strong Gain)')
    plt.ylabel('u2 (Small Magnitude)')
    plt.grid(True)
    plt.show()

    # 3. 权重收敛对比
    w1 = np.array(h['w1'])
    w2 = np.array(h['w2'])
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(w1.shape[1]): plt.plot(t, w1[:,i])
    plt.title('Agent 1 Identifier Weights')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for i in range(w2.shape[1]): plt.plot(t, w2[:,i])
    plt.title('Agent 2 Identifier Weights')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    sim = MultiAgentSim()
    h = sim.run()
    plot_comparison(h)