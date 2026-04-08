import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres
from tqdm import tqdm

# ===== Загрузка ===== 
save_dir = 'compressed_data555'
unique_M = np.load(f'{save_dir}/unique_M.npy')
pair_map = np.load(f'{save_dir}/pair_map.npy')
flux = np.load(f'{save_dir}/flux.npy')
with open(f'{save_dir}/params.json') as f: params = json.load(f)
R, L, C = params['R'], params['L'], params['C']
all_freqs = np.array(params['freqs'])[:1000]
N = pair_map.shape[0]

RES_MIN, RES_MAX = 38e6, 42e6
res_freqs = np.sort(all_freqs[(all_freqs >= RES_MIN) & (all_freqs <= RES_MAX)])

def build_Z(omega):
    Z = np.zeros((N, N), dtype=np.complex128)
    Z0 = R - 1j*omega*L - 1/(1j*omega*C)
    np.fill_diagonal(Z, Z0)
    for i in range(N):
        for j in range(i+1, N):
            idx = pair_map[i, j]
            if idx >= 0: Z[i, j] = Z[j, i] = -1j*omega*unique_M[idx]
    return Z

iteration_count = 0

def solve_ZI_v(Z, v, rtol=1e-6):
    global iteration_count
    iteration_count = 0
    def callback(rk):
        global iteration_count
        iteration_count += 1
        print(f"Итерация {iteration_count}, норма остатка: {rk:.6e}")
    I, info = gmres(Z, v, callback=callback, restart=1000000000, rtol=rtol)
    residual = np.linalg.norm(v - Z @ I) / np.linalg.norm(v)
    print(f"  -> info = {info}, невязка = {residual:.2e}")
    return iteration_count, info, I


iters, infos = [], []

for omega in tqdm(res_freqs, desc="Частота"):
    Z = build_Z(omega)
    v = -1j * omega * flux
    n_iter, info, _ = solve_ZI_v(Z, v, rtol=1e-6)
    iters.append(n_iter); infos.append(info); 

iters, infos = np.array(iters), np.array(infos)

plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.plot(res_freqs/1e6, iters, marker='.', ms=3, lw=1)
plt.axhline(np.mean(iters), color='r', ls='--', label=f'Среднее: {np.mean(iters):.1f}')
plt.xlabel('Частота (МГц)'); plt.ylabel('Итерации'); plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.hist(iters, bins=20, color='gray', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(iters), color='r', ls='--')
plt.xlabel('Итерации'); plt.ylabel('Количество'); plt.grid(axis='y', alpha=0.3)

plt.show()

