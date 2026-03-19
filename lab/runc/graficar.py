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
        plt.plot(df.index, df[col_eta]*100, label=str(i*10+5)+' (m)', linewidth=1.5)
        #plt.xlabel("Time (s)")
        #plt.ylabel("Elevation (m)")
        plt.legend(loc = 'center right')
        plt.grid(True)
        plt.xlim(0,100)
        plt.tight_layout()
#%
df = pd.read_csv(carpeta+'trillo1.csv', header=None)
df.columns = ['X', 'Y']
plt.figure(1)
plt.subplot(8, 1, 1)
offset = (max(df['Y'])+min(df['Y']))/2*0
plt.plot(df['X'], df['Y']-offset,'-', linewidth = 1,label='Trillo')
#plt.title('(a) 5m')
#plt.xlim(min(df['X']),max(df['X']))
plt.tight_layout()
#plt.legend()
plt.grid(True)
df = pd.read_csv(carpeta+'/trillo2.csv', header=None)
df.columns = ['X', 'Y']
plt.figure(1)
plt.subplot(8, 1, 5)
plt.plot(df['X'], df['Y']-offset,'-', linewidth = 1,label='Trillo')
#plt.title('(e) 45m')
#plt.xlim(min(df['X']),max(df['X']))
plt.tight_layout()
#plt.legend()
df = pd.read_csv(carpeta+'/trillo3.csv', header=None)
df.columns = ['X', 'Y']
plt.figure(1)
plt.subplot(8, 1, 2)
plt.plot(df['X'], df['Y']-offset,'-', linewidth = 1,label='Trillo')
#plt.title('(b) 15m')
#plt.xlim(min(df['X']),max(df['X']))
plt.tight_layout()
#plt.legend()
df = pd.read_csv(carpeta+'/trillo4.csv', header=None)
df.columns = ['X', 'Y']
plt.figure(1)
plt.subplot(8, 1, 6)
plt.plot(df['X'], df['Y']-offset,'-', linewidth = 1,label='Trillo')
#plt.title('(f) 55m')
#plt.xlim(min(df['X']),max(df['X']))
plt.tight_layout()
#plt.legend()
df = pd.read_csv(carpeta+'/trillo5.csv', header=None)
df.columns = ['X', 'Y']
plt.figure(1)
plt.subplot(8, 1, 3)
plt.plot(df['X'], df['Y']-offset,'-', linewidth = 1,label='Trillo')
#plt.title('(c) 25m')
#plt.xlim(min(df['X']),max(df['X']))
plt.tight_layout()
#plt.legend()
df = pd.read_csv(carpeta+'/trillo6.csv', header=None)
df.columns = ['X', 'Y']
plt.figure(1)
plt.subplot(8, 1, 7)
plt.plot(df['X'], df['Y']-offset,'-', linewidth = 1,label='Trillo')
#plt.title('(g) 65m')
#plt.xlim(min(df['X']),max(df['X']))
plt.tight_layout()
#plt.legend()
df = pd.read_csv(carpeta+'/trillo7.csv', header=None)
df.columns = ['X', 'Y']
plt.figure(1)
plt.subplot(8, 1, 4)
plt.plot(df['X'], df['Y']-offset,'-', linewidth = 1,label='Trillo')
#plt.title('(d) 35m')
#plt.xlim(min(df['X']),max(df['X']))
plt.tight_layout()
#plt.legend()
df = pd.read_csv(carpeta+'/trillo8.csv', header=None)
df.columns = ['X', 'Y']
plt.figure(1)
plt.subplot(8, 1, 8)
plt.plot(df['X'], df['Y']-offset,'-', linewidth = 1,label='Trillo')
#plt.title('(h) 75m')
#plt.xlim(min(df['X']),max(df['X']))
#plt.legend()
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
data_file = "17_09_15_16.001(F).txt"  # archivo de entrada (2 columnas: tiempo, señal)
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
plt.plot(t_u, y_u*100, '--b', lw=1)
plt.show()



plt.savefig(carpeta + 'trillo.png')

plt.show()


#%% signal
import numpy as np
import matplotlib.pyplot as plt

# ================== PARÁMETROS DE EXPORTACIÓN ==================
M = 200                    # número objetivo de componentes a exportar
T_min = 2.0               # periodo mínimo [s] permitido en el archivo (evita HF que revientan el solver)
match_Hs = True           # igualar Hs de la suma a Hs de la serie usada
scale_factor = 1.0        # factor manual si tu solver satura (0.9, 0.8, etc.)
phase_sign_export = "+"   # "+" usa cos(ωt+φ), "-" exporta φ invertida (cos(ωt-φ))
phase_units = "rad"       # "rad" (por defecto) o "deg"
out_file = "waves_fft.txt"

# ================== TU LÓGICA ORIGINAL DE SEGMENTO Y GRÁFICOS ==================
# Requiere un DataFrame 'df' con columnas "X" (tiempo) y "Y" (elevación en cm)

# ---------- Segmento igual al del gráfico ----------
n = len(df['X'])
o = 25
i0 = n//3 + o
i1 = 2*n//3 + o//10

x = df['X'].to_numpy(dtype=float)
y_cm = df['Y'].to_numpy(dtype=float)

# Offset global como en tu gráfico original (sobre todo Y)
offset_cm = (y_cm.max() + y_cm.min())/2.0

# Extraer segmento y convertir a metros
x_seg = x[i0:i1]
y_seg_m = (y_cm[i0:i1] - offset_cm)/100.0  # cm -> m

# Tiempo relativo del segmento
t_raw = x_seg - x_seg.min()

# Ordenar y eliminar tiempos duplicados
ordr = np.argsort(t_raw)
t_raw = t_raw[ordr]
y_seg_m = y_seg_m[ordr]
t_u, idx = np.unique(t_raw, return_index=True)
y_u = y_seg_m[idx]

# Re-muestreo uniforme con suficientes puntos para M bins positivos
M_req = M
Nres = max(2*M_req + 4, t_u.size)   # garantiza >= M frecuencias positivas
t = np.linspace(t_u[0], t_u[-1], Nres)
y = np.interp(t, t_u, y_u)

# ---------- FFT lado positivo ----------
dt = float(t[1] - t[0])
Nsig = t.size
Y = np.fft.rfft(y)
f = np.fft.rfftfreq(Nsig, d=dt)

# Amplitudes y fases de un solo lado (para cosenos)
amps_full = 2.0*np.abs(Y)/Nsig
phases_full = np.angle(Y)  # para cos(2π f t + φ)

# Corrección Nyquist si N es par (ese bin no se duplica)
if Nsig % 2 == 0 and amps_full.size > 1:
    amps_full[-1] *= 0.5

# Excluir DC
f_pos = f[1:]
a_pos = amps_full[1:]
ph_pos = phases_full[1:]

# ---------- FILTRO DE BANDA: PERIODO MÍNIMO ----------
# Evita periodos demasiado cortos (altas frecuencias) que provocan NaN en el solver
fmax_band = 1.0 / T_min
mask_band = f_pos <= fmax_band
f_pos = f_pos[mask_band]
a_pos = a_pos[mask_band]
ph_pos = ph_pos[mask_band]

if f_pos.size == 0:
    raise RuntimeError(f"No quedan componentes después de aplicar T_min={T_min}s. Baja T_min o usa un segmento más largo.")

# Tomar exactamente M componentes de menor frecuencia dentro de la banda
if f_pos.size < M_req:
    M_eff = f_pos.size
else:
    M_eff = M_req

f_sel = f_pos[:M_eff]
a_sel = a_pos[:M_eff]
ph_sel = ph_pos[:M_eff]

# Normalizar fase a [0, 2π)
ph_sel = np.mod(ph_sel, 2.0*np.pi)

# Orden por periodo descendente (como tu JONSWAP)
T_sel = 1.0/f_sel
order = np.argsort(T_sel)[::-1]
T_sel = T_sel[order]
a_sel = a_sel[order]
ph_sel = ph_sel[order]

# Escala manual opcional
a_sel = a_sel * scale_factor

# ---------- Ajuste de energía (opcional) ----------
if match_Hs:
    Hs_meas = 4.0 * np.std(y)
    m0_comp = 0.5 * np.sum(a_sel**2)
    if m0_comp > 0:
        Hs_comp = 4.0 * np.sqrt(m0_comp)
        gain = Hs_meas / Hs_comp
        a_sel = a_sel * gain
    else:
        gain = 1.0
else:
    gain = 1.0
    Hs_meas = 4.0 * np.std(y)

# ---------- Signo y unidades de fase solo para EXPORTAR ----------
if phase_sign_export == "-":
    ph_out = (-ph_sel) % (2.0*np.pi)
else:
    ph_out = ph_sel

if phase_units.lower().startswith("deg"):
    ph_out_file = np.degrees(ph_out)
else:
    ph_out_file = ph_out

direc = np.zeros_like(a_sel)

# Comprobaciones de finitud
if not (np.all(np.isfinite(a_sel)) and np.all(np.isfinite(T_sel)) and np.all(np.isfinite(ph_out_file))):
    raise RuntimeError("Se detectaron valores no finitos en las componentes (amplitud/periodo/fase). Revisa el filtrado y el segmento.")

# ---------- Guardar waves_fft.txt ----------
with open(out_file, "w") as fh:
    fh.write(f"[NumberOfWaves] {a_sel.size}\n")
    fh.write("=================================\n")
    for ai, Ti, di, ph in zip(a_sel, T_sel, direc, ph_out_file):
        fh.write(f"{ai:.6f}\t{Ti:.6f}\t{di:.6f}\t{ph:.6f}\n")

# ---------- Gráficos (tus mismos) ----------
# Señal usada
plt.figure(figsize=(9,3))
plt.plot(t, y, 'k', lw=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Elevación [m]")
plt.title("Señal usada para la FFT (segmento en m)")
plt.grid(True)
plt.tight_layout()

# Espectro
plt.figure(figsize=(9,3))
plt.plot(f[1:], amps_full[1:], color='0.6', lw=1, label="Espectro (sin DC)")
f_mark = 1.0/T_sel[::-1]
# a_mark antes de gains para comparar con la curva gris:
if match_Hs and (gain != 0):
    a_mark = (a_sel/(gain*scale_factor))[::-1]
else:
    a_mark = (a_sel/(scale_factor))[::-1]
plt.plot(f_mark, a_mark, 'C1o', label=f"{a_sel.size} seleccionadas (T ≥ {T_min:.2f}s)")
plt.vlines(f_mark, 0, a_mark, colors='C1', lw=0.8)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [m]")
plt.title("Espectro (lado positivo)")
plt.xlim(0,2)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Reconstrucción
eta_rec = np.zeros_like(t)
for ai, Ti, ph in zip(a_sel, T_sel, ph_sel):
    eta_rec += ai*np.cos(2*np.pi*(1.0/Ti)*t + ph)

plt.figure(figsize=(9,3))
plt.plot(t, y, 'k', lw=1, label="Original (m)")
plt.plot(t, eta_rec, 'C2', lw=1, alpha=0.9, label=f"Reconstrucción {a_sel.size} términos")
plt.xlabel("Tiempo [s]")
plt.ylabel("Elevación [m]")
plt.title("Reconstrucción vs original")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# ---------- Info útil ----------
info = {
    "N_out": int(a_sel.size),
    "T_min_export": float(T_min),
    "min_T_sel": float(np.min(T_sel)),
    "max_T_sel": float(np.max(T_sel)),
    "min_f_sel": float(np.min(1.0/T_sel)),
    "max_f_sel": float(np.max(1.0/T_sel)),
    "Hs_meas": float(Hs_meas),
    "Hs_comp_antes_gain": float(4*np.sqrt(0.5*np.sum((a_sel/(gain if match_Hs else 1.0))**2))) if a_sel.size else 0.0,
    "gain_aplicado": float(gain),
}
print(f"{out_file} generado con {a_sel.size} componentes.")
print("Resumen:", info)
