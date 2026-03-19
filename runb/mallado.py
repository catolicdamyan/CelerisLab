#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import re
import os

plt.style.use('Solarize_Light2')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 5)))

x_target = 30
h = 0.2
carpetas = ["0p16", "0p08", "0p05.","0p04", "0p02", "0p01"]

def sort_files_numerically(files):
    return sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(1)))

def load_bin(filename, ny, nx):
    with open(filename, 'rb') as f:
        return np.fromfile(f, dtype=np.float32).reshape((ny, nx))

def load_simulation_data(carpeta, x_target):
    with open(os.path.join(carpeta, 'config.json'), 'r') as f:
        config = json.load(f)
    nx = config['WIDTH']
    ny = config['HEIGHT']
    dx = config['dx']
    x = np.arange(0, nx) * dx
    y = np.arange(0, ny) * config['dy']
    j_target = np.argmin(np.abs(y - 1.5))
    i_target = np.argmin(np.abs(x - x_target))
    elev_files = sort_files_numerically(glob.glob(os.path.join(carpeta, 'elev_*.bin')))
    time_files = sort_files_numerically(glob.glob(os.path.join(carpeta, 'time_*.txt')))
    min_len = min(len(elev_files), len(time_files))
    times_sim = np.array([ float(np.loadtxt(tf)) for tf in time_files[:min_len]])
    sort_idx = np.argsort(times_sim)
    times_sim = times_sim[sort_idx]
    elev_files = [elev_files[i] for i in sort_idx]
    y_sim = np.array([load_bin(elev_files[i], ny, nx)[j_target, i_target] for i in range(len(elev_files))])
    return times_sim, y_sim, dx

datos = {carpeta: load_simulation_data(carpeta, x_target) for carpeta in carpetas}

dx_adim_values = []
rmse_adim_values = []

for i in range(1, len(carpetas)):
    carpeta_anterior = carpetas[i-1]
    carpeta_actual = carpetas[i]
    times_ref, y_ref, _ = datos[carpeta_anterior]
    times_curr, y_curr, dx_curr = datos[carpeta_actual]
    t_min = max(times_ref.min(), times_curr.min())
    t_max = min(times_ref.max(), times_curr.max())
    t_common = np.linspace(t_min, t_max, min(len(times_ref), len(times_curr)))
    y_ref_interp = np.interp(t_common, times_ref, y_ref)
    y_curr_interp = np.interp(t_common, times_curr, y_curr)
    rmse = np.mean(np.sqrt((y_curr_interp - y_ref_interp)**2))
    dx_adim_values.append(dx_curr / h)
    rmse_adim_values.append(rmse / h)

sort_idx = np.argsort(dx_adim_values)
dx_adim_values = np.array(dx_adim_values)[sort_idx]
rmse_adim_values = np.array(rmse_adim_values)[sort_idx]

plt.figure(1, figsize=(8, 4), dpi=150)
plt.plot(dx_adim_values, rmse_adim_values, 'o-', linewidth=2.5, markersize=10, 
         label='Error Cuadrático Medio (RMSE)')
plt.xlabel(r'$dx/h_0$')
plt.ylabel(r'$RMSE$')
plt.title(f'X = {x_target} m, Y = 1.5 m')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('estudio_mallado.png', dpi=300, bbox_inches='tight')
plt.show()
