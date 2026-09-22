import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# ==========================================
# 1. 入力パラメータ
# ==========================================
Vs = 1500.0            # S波速度 (m/s)
Vp = 3000.0            # P波速度 (m/s)
nu = 0.33             # ポアソン比 (Vp, Vsから整合をとる場合は注意)
gamma = 18.0          # 単位体積重量 (kN/m^3)
Lx = 80.0             # 基礎幅 縦 (m)
Ly = 80.0             # 基礎幅 横 (m)
Nx = 20               # 縦方向分割数
Ny = 20               # 横方向分割数
contact_pressure_type = "rigid"  # "uniform" (一様) または "rigid" (剛版分布)
freq_max = 20.0       # 解析最大振動数 (Hz)
num_freqs = 40        # 振動数ステップ数

# ==========================================
# 2. 地盤定数の算出
# ==========================================
g = 9.81              # 重力加速度 (m/s^2)
rho = (gamma * 1e3) / g   # 密度 (kg/m^3)
G = rho * (Vs ** 2)       # せん断弾性係数 (N/m^2)

# ==========================================
# 3. メッシュ分割と座標幾何
# ==========================================
dx = Lx / Nx
dy = Ly / Ny
dA = dx * dy

x = np.linspace(-Lx / 2.0 + dx / 2.0, Lx / 2.0 - dx / 2.0, Nx)
y = np.linspace(-Ly / 2.0 + dy / 2.0, Ly / 2.0 - dy / 2.0, Ny)
X, Y = np.meshgrid(x, y)
X_flat = X.ravel()
Y_flat = Y.ravel()
num_elements = Nx * Ny

# 要素間距離行列 R_ij および方向ベクトル
dX = X_flat[:, np.newaxis] - X_flat[np.newaxis, :]
dY = Y_flat[:, np.newaxis] - Y_flat[np.newaxis, :]
R = np.sqrt(dX**2 + dY**2)

# ==========================================
# 4. 各モードの接地圧/反力モード分布の定義
# ==========================================
eps = 1e-4
xi = np.clip(np.abs(2.0 * X_flat / Lx), 0, 1.0 - eps)
eta = np.clip(np.abs(2.0 * Y_flat / Ly), 0, 1.0 - eps)

if contact_pressure_type == "uniform":
    # 鉛直・水平：一様
    w_base = np.ones(num_elements)
    # 回転（y軸周りモーメント・x方向線形傾斜）：x
    w_rock = X_flat.copy()
elif contact_pressure_type == "rigid":
    # 剛版特異性
    base_singularity = 1.0 / (np.sqrt(1.0 - xi**2) * np.sqrt(1.0 - eta**2))
    w_base = base_singularity
    w_rock = X_flat * base_singularity
else:
    raise ValueError("contact_pressure_type must be 'uniform' or 'rigid'")

# 1) 鉛直 (Vertical): 全力 = 1.0 N
p_vert = (w_base / np.sum(w_base * dA)) * dA

# 2) 水平 (Horizontal: x方向加振): 全力 = 1.0 N
p_horiz = (w_base / np.sum(w_base * dA)) * dA

# 3) 回転 (Rocking: y軸周りモーメント M_y): 全モーメント = 1.0 N・m
#    モーメント M = sum(p_elem_i * x_i) = 1.0
p_rock = (w_rock / np.sum(w_rock * X_flat * dA)) * dA

# ==========================================
# 5. 静的 Green 関数マトリクスの構築
# ==========================================
with np.errstate(divide='ignore', invalid='ignore'):
    cos2_theta = (dX / R)**2
    cos2_theta[np.isnan(cos2_theta)] = 0.0

    # 鉛直 (Boussinesq解)
    G_stat_v = (1.0 - nu) / (2.0 * np.pi * G * R)

    # 水平 x方向加振・x方向変位 (Cerruti解)
    G_stat_h = (1.0 / (4.0 * np.pi * G * R)) * (2.0 * (1.0 - nu) + 2.0 * nu * cos2_theta)

# 対角項（自己要素の特異点積分平均）
diag_v = (1.0 - nu) / (G * np.sqrt(np.pi * dA))
diag_h = (2.0 - nu) / (2.0 * G * np.sqrt(np.pi * dA))
np.fill_diagonal(G_stat_v, diag_v)
np.fill_diagonal(G_stat_h, diag_h)

# ==========================================
# 6. 動的インピーダンスの計算ループ
# ==========================================
freqs = np.linspace(0.001, freq_max, num_freqs)

# 結果格納配列 [モード名] = (K1, K2)
results = {
    'Vertical':   {'real': np.zeros(num_freqs), 'imag': np.zeros(num_freqs), 'unit': 'MN/m'},
    'Horizontal': {'real': np.zeros(num_freqs), 'imag': np.zeros(num_freqs), 'unit': 'MN/m'},
    'Rocking':    {'real': np.zeros(num_freqs), 'imag': np.zeros(num_freqs), 'unit': 'GN・m/rad'}
}

for idx, f in enumerate(freqs):
    omega = 2.0 * np.pi * f
    ks = omega / Vs
    phase = np.exp(-1j * ks * R)

    # 1) 鉛直
    F_v = G_stat_v * phase
    w_elems = F_v @ p_vert
    w_avg = np.sum(w_elems * p_vert)
    K_v = 1.0 / w_avg
    results['Vertical']['real'][idx] = K_v.real
    results['Vertical']['imag'][idx] = K_v.imag

    # 2) 水平
    F_h = G_stat_h * phase
    u_elems = F_h @ p_horiz
    u_avg = np.sum(u_elems * p_horiz)
    K_h = 1.0 / u_avg
    results['Horizontal']['real'][idx] = K_h.real
    results['Horizontal']['imag'][idx] = K_h.imag

    # 3) 回転 (鉛直柔軟度マトリクスを使用し、回転角 theta を評価)
    #    各要素の鉛直変位 w = F_v * p_rock
    #    剛体傾転角 theta_y = sum(w_elems * x_i * dA) / I_geom (仮想仕事重み平均)
    w_rock_elems = F_v @ p_rock
    theta_avg = np.sum(w_rock_elems * p_rock)  # 単位モーメントに対する回転角
    K_r = 1.0 / theta_avg
    results['Rocking']['real'][idx] = K_r.real
    results['Rocking']['imag'][idx] = K_r.imag

# ==========================================
# 7. 計算値のコンソール出力（一部抜粋）
# ==========================================
header = f"{'Freq(Hz)':>8} | {'KV_real':>10} {'KV_imag':>10} | {'KH_real':>10} {'KH_imag':>10} | {'KR_real':>10} {'KR_imag':>10}"
print(header)
print("-" * len(header))
for i in range(0, num_freqs):
    print(f"{freqs[i]:8.2f} | "
          f"{results['Vertical']['real'][i]/1e6:10.2f} {results['Vertical']['imag'][i]/1e6:10.2f} | "
          f"{results['Horizontal']['real'][i]/1e6:10.2f} {results['Horizontal']['imag'][i]/1e6:10.2f} | "
          f"{results['Rocking']['real'][i]/1e9:10.2f} {results['Rocking']['imag'][i]/1e9:10.2f}")
    
# ==========================================
# 8. グラフ描画（3枚、左単一軸共通）
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

modes = [
    ('Vertical', '鉛直動的インピーダンス $K_V$', 1e6),
    ('Horizontal', '水平動的インピーダンス $K_H$', 1e6),
    ('Rocking', '回転動的インピーダンス $K_R$', 1e9)
]

for ax, (key, title, scale) in zip(axes, modes):
    k1 = results[key]['real'] / scale
    k2 = results[key]['imag'] / scale
    unit_str = results[key]['unit']

    ax.plot(freqs, k1, color='tab:blue', linewidth=2.0, label='実部 $K_1$ (剛性)')
    ax.plot(freqs, k2, color='tab:red', linewidth=2.0, linestyle='--', label='虚部 $K_2$ (減衰)')

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel('振動数 Frequency [Hz]', fontsize=11)
    ax.set_ylabel(f'インピーダンス [{unit_str}]', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
plt.show()

