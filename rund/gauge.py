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
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 9)))

# ================== CARGAR CONFIGURACIÓN SIMULACIÓN ==================
carpeta_sim = "data"
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


# ================== CARGAR DATOS DE SIMULACIÓN ==================
def sort_files_numerically(files):
    return sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(1)))

def load_bin(filename, ny, nx):
    with open(filename, 'rb') as f:
        return np.fromfile(f, dtype=np.float32).reshape((ny, nx))

elev_files = sort_files_numerically(glob.glob(os.path.join(carpeta_sim, 'elev_*.bin')))
time_files = sort_files_numerically(glob.glob(os.path.join(carpeta_sim, 'time_*.txt')))
min_len = min(len(elev_files), len(time_files))
elev_files = elev_files[:min_len]
time_files = time_files[:min_len]

# Cargar tiempos de simulación
times_sim = np.array([float(np.loadtxt(tf)) for tf in time_files])

print(f"\nSimulación: {len(times_sim)} pasos temporales")
print(f"Rango temporal simulación: [{times_sim[0]:.2f}, {times_sim[-1]:.2f}] s")

# ================== FILTRAR RANGO TEMPORAL COMÚN (igual que trillo_celeris.py) ==================
# Cargar serie de tiempo original para determinar rango común
data_file = "18_09_15_09.001(F).txt"
raw = np.loadtxt(data_file)
t_original = raw[:, 0].astype(float)
y_original = raw[:, 1].astype(float)
ordr = np.argsort(t_original)
t_original = t_original[ordr]
y_original = y_original[ordr]
t_orig_u, idx = np.unique(t_original, return_index=True)

t_min = max(t_orig_u[0], times_sim[0])
t_max = min(t_orig_u[-1], times_sim[-1])

mask_sim = (times_sim >= t_min) & (times_sim <= t_max)
times_common = times_sim[mask_sim]
elev_files_common = [elev_files[i] for i in range(len(elev_files)) if mask_sim[i]]

print(f"\nRango temporal común: [{t_min:.2f}, {t_max:.2f}] s")
print(f"Pasos temporales en común: {len(times_common)}")

# ================== DEFINIR POSICIONES DE BOYAS (GAUGES) ==================
# Definir posiciones x donde guardar series de tiempo
#x_gauge_positions = [0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0]
x_gauge_positions = [0.0, 5.01891793, 14.93725575, 24.85559356 ,34.89342942 ,44.81176724, 54.73010505, 64.76794092 ,74.68627873]
y_gauge = y_actual  # Usar la misma y que se está analizando

# Encontrar índices más cercanos para cada posición x (mismo método simple que antes)
gauge_indices = []
gauge_x_actual = []
for x_pos in x_gauge_positions:
    i_gauge = np.argmin(np.abs(x - x_pos))
    gauge_indices.append(i_gauge)
    gauge_x_actual.append(x[i_gauge])
    print(f"  Boya en X={x_pos:.2f} m: usando índice {i_gauge} (X={x[i_gauge]:.4f} m)")

# Interpolar serie del solitón a los tiempos de simulación (para el gráfico de comparación)
y_orig_u = y_original[idx]  # Ya tenemos t_orig_u de antes
y_orig_interp = np.interp(times_common, t_orig_u, y_orig_u)


# ================== EXTRAER SERIES DE TIEMPO EN POSICIONES DE BOYAS ==================
gauge_time_series = []
for i_gauge in gauge_indices:
    y_sim_gauge = np.array([load_bin(ef, ny, nx)[j_target, i_gauge] for ef in elev_files_common])
    gauge_time_series.append(y_sim_gauge)
    if len(gauge_time_series) % 2 == 0:
        print(f"  Procesadas {len(gauge_time_series)}/{len(gauge_indices)} boyas...")

# ================== GRÁFICO: SERIES DE TIEMPO EN SUBPLOTS APILADOS ==================
# y_orig_interp ya está calculado arriba (línea 80)


# ================== GRÁFICO: COMPARACIÓN PRIMERA BOYA CON SERIE ORIGINAL ==================
plt.figure(figsize=(8, 4), dpi=150)

plt.plot(times_common, y_orig_interp, 'k-', linewidth=2.5, label='Real time series', alpha=0.8)
plt.plot(times_common, gauge_time_series[0], 'r--', linewidth=2, label=f'Celeris WebGPU (X = {gauge_x_actual[0]:.2f} m)', alpha=0.8)
plt.xlabel('Time [s]')
plt.ylabel(r'$\eta$ [m]')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('comparacion_primera_boya_antes_guardar.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGràficos generados antes de guardar:")
print("  - series_tiempo_boyas.png")
print("  - comparacion_primera_boya_antes_guardar.png")

# ================== GUARDAR POSICIONES Y SERIES DE TIEMPO EN ARCHIVO ==================
output_file = "gauge_time_series.txt"

with open(output_file, 'w') as f:
    # Primera sección: Posiciones de boyas (formato similar a time_series_locations.txt)
    f.write("# Posiciones de boyas (X, Y) [m]\n")
    f.write("# Formato: X\tY\n")
    for x_actual in gauge_x_actual:
        f.write(f"{x_actual:.6f}\t{y_gauge:.6f}\n")
    
    # Separador
    f.write("\n# Series de tiempo\n")
    f.write("# Formato: Time[s]")
    for idx in range(len(gauge_indices)):
        f.write(f"\tEta{idx+1}[m]")
    f.write("\n")
    
    # Segunda sección: Series de tiempo (tiempo, eta1, eta2, eta3, ...)
    # Guardar con máxima precisión (float32 tiene ~7 dígitos significativos)
    for t_idx, t in enumerate(times_common):
        f.write(f"{t:.10f}")
        for series in gauge_time_series:
            f.write(f"\t{series[t_idx]:.10f}")
        f.write("\n")

print("ARCHIVO GUARDADO")

#% ================== LEER ARCHIVO GUARDADO ==================
print("\nLeyendo archivo guardado para verificación...")
with open(output_file, 'r') as f:
    lines = f.readlines()

# Leer posiciones
positions_loaded = []
reading_positions = True
data_start_line = 0
for i, line in enumerate(lines):
    line = line.strip()
    if line.startswith('#'):
        if 'Series de tiempo' in line:
            reading_positions = False
            data_start_line = i + 2  # Saltar encabezado de columnas
        continue
    if not line:
        continue
    if reading_positions:
        parts = line.split()
        if len(parts) >= 2:
            x_loaded = float(parts[0])
            y_loaded = float(parts[1])
            positions_loaded.append((x_loaded, y_loaded))

# Leer series de tiempo
times_loaded = []
etas_loaded = [[] for _ in range(len(positions_loaded))]
for i in range(data_start_line, len(lines)):
    line = lines[i].strip()
    if line and not line.startswith('#'):
        parts = line.split()
        if len(parts) >= len(positions_loaded) + 1:
            times_loaded.append(float(parts[0]))
            for j in range(len(positions_loaded)):
                etas_loaded[j].append(float(parts[j+1]))

times_loaded = np.array(times_loaded)
etas_loaded = [np.array(eta) for eta in etas_loaded]

# ================== GRÁFICO: SERIES DE TIEMPO EN SUBPLOTS APILADOS ==================
n_gauges = len(gauge_x_actual)
fig, axes = plt.subplots(n_gauges, 1, figsize=(8, 8), dpi=150, sharex=True)

for idx, (ax, x_actual, series) in enumerate(zip(axes, gauge_x_actual, gauge_time_series)):
    ax.plot(times_common, series, linewidth=2, color=f'C{idx}')
    ax.set_ylabel(r'$\eta$ [m]')
    ax.grid(True)
    ax.set_xlim(times_common[0], times_common[-1])

axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig('series_tiempo_boyas.png', dpi=300, bbox_inches='tight')
plt.show()
