#!/usr/bin/env python3
"""
Análisis de error espacial: Compara la serie de tiempo del solitón sintético
con las series simuladas a lo largo de la línea media del canal (y = 1.5 m)
para identificar dónde se reproduce mejor la condición de borde impuesta.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import re

plt.style.use('Solarize_Light2')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 5)))

# ================== CARGAR SERIE DE TIEMPO DEL SOLITÓN SINTÉTICO ==================
data_file ="18_09_15_09.001(F).txt"

raw = np.loadtxt(data_file)
t_original = raw[:, 0].astype(float)
y_original = raw[:, 1].astype(float)

# Ordenar y eliminar duplicados
ordr = np.argsort(t_original)
t_original = t_original[ordr]
y_original = y_original[ordr]
t_orig_u, idx = np.unique(t_original, return_index=True)
y_orig_u = y_original[idx]

print("=" * 60)
print("SERIE DE TIEMPO DEL SOLITÓN CARGADA")
print("=" * 60)
print(f"Número de puntos: {len(t_orig_u)}")
print(f"Rango temporal: [{t_orig_u[0]:.2f}, {t_orig_u[-1]:.2f}] s")
print("=" * 60)

# ================== CARGAR CONFIGURACIÓN SIMULACIÓN ==================
with open('config.json', 'r') as f:
    config = json.load(f)

nx = config['WIDTH']
ny = config['HEIGHT']
dx = config['dx']
dy = config['dy']

# Crear coordenadas
x = np.arange(0, nx) * dx
y = np.arange(0, ny) * dy

# Encontrar el índice j más cercano a y = 1.5 m
y_target = 1.5
j_target = np.argmin(np.abs(y - y_target))
y_actual = y[j_target]

print(f"\nDominio de simulación: {nx} × {ny} puntos")
print(f"Resolución espacial: dx={dx:.4f} m, dy={dy:.4f} m")
print(f"Analizando línea media: y = {y_actual:.4f} m (índice j = {j_target})")
print("=" * 60)

# ================== CARGAR DATOS DE SIMULACIÓN ==================
def sort_files_numerically(files):
    return sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(1)))

def load_bin(filename):
    with open(filename, 'rb') as f:
        return np.fromfile(f, dtype=np.float32).reshape((ny, nx))

elev_files = sort_files_numerically(glob.glob('elev_*.bin'))
time_files = sort_files_numerically(glob.glob('time_*.txt'))
min_len = min(len(elev_files), len(time_files))
elev_files = elev_files[:min_len]
time_files = time_files[:min_len]

# Cargar tiempos de simulación
times_sim = np.array([float(np.loadtxt(tf)) for tf in time_files])
print(f"\nSimulación: {len(times_sim)} pasos temporales")
print(f"Rango temporal simulación: [{times_sim[0]:.2f}, {times_sim[-1]:.2f}] s")

# ================== INTERPOLAR SERIE ORIGINAL A TIEMPOS DE SIMULACIÓN ==================
t_min = max(t_orig_u[0], times_sim[0])
t_max = min(t_orig_u[-1], times_sim[-1])

mask_sim = (times_sim >= t_min) & (times_sim <= t_max)
times_common = times_sim[mask_sim]
elev_files_common = [elev_files[i] for i in range(len(elev_files)) if mask_sim[i]]

print(f"\nRango temporal común: [{t_min:.2f}, {t_max:.2f}] s")
print(f"Pasos temporales en común: {len(times_common)}")

# Interpolar serie del solitón a los tiempos de simulación
y_orig_interp = np.interp(times_common, t_orig_u, y_orig_u)

# ================== CALCULAR ERROR SOLO EN LA LÍNEA MEDIA (y = 1.5 m) ==================
# Vector para almacenar el error RMS en cada punto x a lo largo de y = 1.5 m
error_line = np.zeros(nx)

print("\nCalculando error a lo largo de la línea media (y = 1.5 m)...")
for i in range(nx):
    if i % 500 == 0:
        print(f"  Procesando x[{i}]/{nx}...")
    
    # Extraer serie temporal en el punto (j_target, i)
    y_sim = np.zeros(len(times_common))
    
    for t_idx, ef in enumerate(elev_files_common):
        eta = load_bin(ef)
        y_sim[t_idx] = eta[j_target, i]
    
    # Calcular error RMS entre la serie simulada y la original
    diff = y_sim - y_orig_interp
    rms_error = np.sqrt(np.mean(diff**2))
    error_line[i] = rms_error

print("Cálculo completado.\n")

# Encontrar el punto x con menor error
min_error_idx = np.argmin(error_line)
min_error_x = x[min_error_idx]
min_error_val = error_line[min_error_idx]

print("=" * 60)
print("PUNTO CON MENOR ERROR")
print("=" * 60)
print(f"Posición: X = {min_error_x:.4f} m, Y = {y_actual:.4f} m")
print(f"Índices: i = {min_error_idx}, j = {j_target}")
print(f"Error RMS: {min_error_val:.6f} m")
print("=" * 60)

# ================== GRÁFICO 1: MAPA DE CALOR DEL ERROR A LO LARGO DE X ==================
plt.figure(figsize=(14, 6), dpi=150)

# Crear heatmap 1D del error (usando imshow para efecto de calor)
error_2d = error_line.reshape(1, -1)  # Convertir a 2D para imshow
im = plt.imshow(error_2d, 
                extent=[x[0], x[-1], 0, 1], 
                origin='lower',
                cmap='jet',
                aspect='auto',
                interpolation='bilinear')

cbar = plt.colorbar(im, label='Error RMS [m]')
plt.xlabel('X [m]', fontsize=12)
plt.ylabel('')
plt.title(f'Error RMS a lo largo de la línea media del canal (y = {y_actual:.2f} m)', fontsize=13)
plt.yticks([])  # Ocultar ticks del eje y

# Marcar el punto con menor error
plt.axvline(x=min_error_x, color='white', linestyle='--', linewidth=2, 
            label=f'Mejor ajuste: X = {min_error_x:.2f} m')
plt.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('mapa_error_linea_media.png', dpi=300, bbox_inches='tight')
plt.show()

# ================== GRÁFICO 2: COMPARACIÓN DE SERIES EN PUNTO ÓPTIMO ==================
# Extraer serie temporal en el punto de mínimo error
y_sim_best = np.zeros(len(times_common))
for t_idx, ef in enumerate(elev_files_common):
    eta = load_bin(ef)
    y_sim_best[t_idx] = eta[j_target, min_error_idx]
#%
plt.figure(figsize=(8, 4), dpi=150)

plt.plot(times_common, y_orig_interp, 'k-', linewidth=2.5, label='Real time series', alpha=0.8)
plt.plot(times_common, y_sim_best, 'r--', linewidth=2, label=f'Celeris WebGPU (X = {min_error_x:.2f} m)', alpha=0.8)
plt.xlabel('Time [s]')
plt.ylabel(r'$\eta$ [m]')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('comparacion_series_tiempo.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnálisis completado. Gráficos guardados:")
print("  - mapa_error_linea_media.png")
print("  - comparacion_series_tiempo.png")
