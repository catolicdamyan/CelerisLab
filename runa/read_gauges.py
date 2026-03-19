#!/usr/bin/e4nv python3
"""
Lee las series de tiempo guardadas en gauge_time_series.txt y las grafica.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('Solarize_Light2')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 8)))

# ================== CARGAR SERIE DE TIEMPO DEL SOLITÓN SINTÉTICO ==================
data_file = "16_09_15_14.001(F).txt"

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

# ================== LEER ARCHIVO GUARDADO ==================
output_file = "gauge_time_series.txt"

if not os.path.exists(output_file):
    raise FileNotFoundError(f"No se encontró el archivo: {output_file}")

print(f"Leyendo archivo: {output_file}")
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

# Extraer posiciones X para los títulos
gauge_x_actual = [pos[0] for pos in positions_loaded]

# ================== INTERPOLAR SERIE ORIGINAL A TIEMPOS DE LA PRIMERA BOYA ==================
# Usar la primera boya (X = 0.06 m) que es la de menor error según trillo_celeris.py
times_first_buoy = times_loaded
eta_first_buoy = etas_loaded[0]

# Encontrar rango temporal común
t_min = max(t_orig_u[0], times_first_buoy[0])
t_max = min(t_orig_u[-1], times_first_buoy[-1])

mask_common = (times_first_buoy >= t_min) & (times_first_buoy <= t_max)
times_common = times_first_buoy[mask_common]
eta_first_buoy_common = eta_first_buoy[mask_common]

# Interpolar serie del solitón a los tiempos de la primera boya
y_orig_interp = np.interp(times_common, t_orig_u, y_orig_u)

print(f"\nRango temporal común: [{t_min:.2f}, {t_max:.2f}] s")
print(f"Pasos temporales en común: {len(times_common)}")

# ================== GRÁFICO: SERIES DE TIEMPO EN SUBPLOTS APILADOS ==================
n_gauges = len(gauge_x_actual)
fig, axes = plt.subplots(n_gauges, 1, figsize=(8, 8), dpi=150, sharex=True)

for idx, (ax, x_actual, series) in enumerate(zip(axes, gauge_x_actual, etas_loaded)):
    ax.plot(times_loaded, series, linewidth=2, color=f'C{idx}')
    ax.set_ylabel(r'$\eta$ [m]')
    ax.grid(True)
    ax.set_xlim(times_loaded[0], times_loaded[-1])

axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig('series_tiempo_boyas.png', dpi=300, bbox_inches='tight')
plt.show()

# ================== GRÁFICO: COMPARACIÓN PRIMERA BOYA CON SERIE ORIGINAL ==================
plt.figure(figsize=(8, 4), dpi=150)

plt.plot(times_common, y_orig_interp, 'k-', linewidth=2.5, label='Real time series', alpha=0.8)
plt.plot(times_common, eta_first_buoy_common, 'r--', linewidth=2, label=f'Celeris WebGPU (X = {gauge_x_actual[0]:.2f} m)', alpha=0.8)
plt.xlabel('Time [s]')
plt.ylabel(r'$\eta$ [m]')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('comparacion_primera_boya.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnálisis completado. Gráficos guardados:")
print("  - series_tiempo_boyas.png")
print("  - comparacion_primera_boya.png")