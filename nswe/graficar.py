import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import re
import imageio
import os
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from matplotlib import animation
from matplotlib.animation import PillowWriter
plt.style.use('Solarize_Light2')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 3)))
#carpeta = r"C:\Programas\Tesis\Celeris\prueba2\\"
carpeta = r"C:\Users\Catolic Damyan\Downloads\Celeris\prueba2\\"

# === Archivos de entrada ===
data_file = "time_series_data.txt"
loc_file = "time_series_locations.txt"

T = 14.1

# === Leer datos con el tiempo como índice ===
df = pd.read_csv(data_file, sep="\t", index_col=0)
df.rename_axis("Time", inplace=True)

# === Leer ubicaciones y ordenar por posición X ===
locs = pd.read_csv(loc_file, sep="\t", header=None, names=["x", "y"])
locs["id"] = np.arange(1, len(locs) + 1)   # número de serie (1,2,3…)
locs = locs.sort_values(by="x").reset_index(drop=True)

#% === Graficar ===
plt.figure(1,figsize=(15, 8), dpi=300)

num_series = len(locs)
for i in range(num_series):
    col_eta = f"Eta{locs.loc[i,'id']}"   # usar el id original ligado a cada boya
    if col_eta in df.columns:
        x_pos = locs.loc[i, "x"]  # posición X de la boya
        plt.figure(1)
        plt.subplot(8, 1, i+1)
        plt.plot(df.index, df[col_eta], label=str(i*10+5)+' (m)', linewidth=1.5)
        #plt.xlabel("Time (s)")
        #plt.ylabel("Elevation (m)")
        plt.legend(loc = 'center right')
        plt.grid(True)
        #plt.xlim(0,100)
        plt.tight_layout()


#% !/usr/bin/env python3
# Lee una serie de tiempo desde "18_09_15_09.001(F).txt" (col 1 = tiempo [s], col 2 = señal),
# grafica la serie, realiza FFT, selecciona componentes, compara reconstrucción con original,
# grafica el espectro usado y genera el archivo de entrada "waves_fft.txt".
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('Solarize_Light2')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 3)))

# ================== PARÁMETROS DE EXPORTACIÓN ==================
data_file = "18_09_15_09.001(F).txt"  # archivo de entrada (2 columnas: tiempo, señal)
out_file = "waves_fft.txt"            # archivo de salida
M = 200                                # número objetivo de componentes a exportar
T_min = 2.0                            # periodo mínimo [s] permitido (filtra HF)
match_Hs = True                        # igualar Hs de la suma al Hs de la serie usada
scale_factor = 1.0                     # factor manual si el solver satura (0.9, 0.8, etc.)
phase_sign_export = "+"                # "+" usa cos(ωt+φ), "-" exporta φ invertida (cos(ωt-φ))
phase_units = "rad"                    # "rad" (por defecto) o "deg"
fig_prefix = "timeseries_fft"          # prefijo para figuras

# ================== LECTURA DE DATOS ==================
if not os.path.exists(data_file):
    raise FileNotFoundError(f"No se encontró el archivo: {data_file}")

# Dos columnas separadas por espacios/tab: tiempo [s], señal (elevación u otra magnitud)
raw = np.loadtxt(data_file)
if raw.ndim != 2 or raw.shape[1] < 2:
    raise ValueError("El archivo debe tener al menos 2 columnas: tiempo y señal.")

t_in = raw[:, 0].astype(float)
y_in = raw[:, 1].astype(float)

# Ordenar por tiempo y eliminar duplicados
ordr = np.argsort(t_in)
t_in = t_in[ordr]
y_in = y_in[ordr]
t_u, idx = np.unique(t_in, return_index=True)
y_u = y_in[idx]

# Graficar serie original leída (sin re-muestreo)
plt.figure(1)
plt.subplot(8, 1, 1)
plt.plot(t_u, y_u, '--b', lw=1)
plt.show()



plt.savefig(carpeta + 'trillo.png')

plt.show()



