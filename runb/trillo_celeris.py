#!/usr/bin/e4nv python3
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
import os

plt.style.use('Solarize_Light2')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 5)))

# ================== CARGAR SERIE DE TIEMPO DESDE waves2.txt ==================
waves_file = "waves2.txt"
carpeta_sim = "0p01"

# Intervalo de tiempo para comparación
t_min_comp = 30.0  # Tiempo mínimo [s]
t_max_comp = 40.0  # Tiempo máximo [s]

# Leer archivo de ondas
with open(waves_file, 'r') as f:
    lines = f.readlines()

# Buscar número de ondas
n_waves = 0
for i, line in enumerate(lines):
    if '[NumberOfWaves]' in line:
        n_waves = int(line.split()[1])
        data_start = i + 2  # Saltar línea de separación
        break

# Leer parámetros de ondas
waves_data = []
for i in range(data_start, len(lines)):
    line = lines[i].strip()
    if line and not line.startswith('='):
        parts = line.split()
        if len(parts) >= 4:
            amp = float(parts[0])
            period = float(parts[1])
            direction = float(parts[2])
            phase = float(parts[3])
            waves_data.append((amp, period, phase))

print("=" * 60)
print("PARÁMETROS DE ONDAS CARGADOS")
print("=" * 60)
print(f"Número de ondas: {len(waves_data)}")
for i, (amp, period, phase) in enumerate(waves_data):
    print(f"  Onda {i+1}: A={amp:.6f} m, T={period:.2f} s, φ={phase:.6f} rad")
print("=" * 60)

# ================== CARGAR CONFIGURACIÓN SIMULACIÓN ==================
with open(os.path.join(carpeta_sim, 'config.json'), 'r') as f:
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

def load_bin(filename, ny, nx):
    with open(filename, 'rb') as f:
        return np.fromfile(f, dtype=np.float32).reshape((ny, nx))

elev_files = sort_files_numerically(glob.glob(os.path.join(carpeta_sim, 'elev_*.bin')))
time_files = sort_files_numerically(glob.glob(os.path.join(carpeta_sim, 'time_*.txt')))
min_len = min(len(elev_files), len(time_files))

# Cargar tiempos de simulación
times_sim = np.array([float(np.loadtxt(tf)) for tf in time_files[:min_len]])
sort_idx = np.argsort(times_sim)
times_sim = times_sim[sort_idx]
elev_files = [elev_files[i] for i in sort_idx]

print(f"\nSimulación: {len(times_sim)} pasos temporales")
print(f"Rango temporal simulación: [{times_sim[0]:.2f}, {times_sim[-1]:.2f}] s")

# ================== GENERAR SERIE DE TIEMPO A PARTIR DE PARÁMETROS DE ONDAS ==================
# Generar señal como suma de componentes de onda
y_orig_interp = np.zeros_like(times_sim)
for amp, period, phase in waves_data:
    y_orig_interp += -amp * np.cos(2.0 * np.pi * times_sim / period + phase)

print(f"\nSerie de tiempo generada: {len(times_sim)} puntos")
print(f"Rango temporal: [{times_sim[0]:.2f}, {times_sim[-1]:.2f}] s")

# ================== CALCULAR ERROR SOLO EN LA LÍNEA MEDIA (y = 1.5 m) ==================
# Filtrar datos en el intervalo de tiempo especificado
mask_time = (times_sim >= t_min_comp) & (times_sim <= t_max_comp)
times_comp = times_sim[mask_time]
y_orig_comp = y_orig_interp[mask_time]
elev_files_comp = [elev_files[i] for i in range(len(elev_files)) if mask_time[i]]

print(f"\nIntervalo de comparación: [{t_min_comp:.2f}, {t_max_comp:.2f}] s")
print(f"Puntos totales: {len(times_sim)}, Puntos en intervalo: {len(times_comp)}")
print(f"Archivos totales: {len(elev_files)}, Archivos en intervalo: {len(elev_files_comp)}")

# Verificar que las longitudes coincidan
if len(y_orig_comp) != len(elev_files_comp):
    print(f"ERROR: Longitudes no coinciden! y_orig_comp={len(y_orig_comp)}, elev_files_comp={len(elev_files_comp)}")

# Vector para almacenar el error RMS en cada punto x a lo largo de y = 1.5 m
error_line = np.zeros(nx)

print("\nCalculando error a lo largo de la línea media (y = 1.5 m)...")
for i in range(nx):
    if i % 500 == 0:
        print(f"  Procesando x[{i}]/{nx}...")
    
    # Extraer serie temporal en el punto (j_target, i) solo en el intervalo
    y_sim_comp = np.array([load_bin(ef, ny, nx)[j_target, i] for ef in elev_files_comp])
    
    # Verificar longitudes
    if len(y_sim_comp) != len(y_orig_comp):
        print(f"ERROR en x[{i}]: y_sim_comp={len(y_sim_comp)}, y_orig_comp={len(y_orig_comp)}")
    
    # Calcular error RMS entre la serie simulada y la original solo en el intervalo
    diff = y_sim_comp - y_orig_comp
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
plt.grid(False)
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
y_sim_best = np.array([load_bin(ef, ny, nx)[j_target, min_error_idx] for ef in elev_files])

plt.figure(figsize=(8, 4), dpi=150)

plt.plot(times_sim, y_orig_interp, 'k-', linewidth=2.5, label='Real wave file')
plt.plot(times_sim, y_sim_best, 'r-', linewidth=2, label=f'Celeris WebGPU (X = {min_error_x:.2f} m)')
plt.xlabel('Time [s]')
plt.ylabel(r'$\eta$ [m]')
plt.title('Comparación de series de tiempo en el punto de mejor ajuste', fontsize=13)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('comparacion_series_tiempo.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnálisis completado. Gráficos guardados:")
print("  - mapa_error_linea_media.png")
print("  - comparacion_series_tiempo.png")
