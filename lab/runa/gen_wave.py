# remuestreo!!!!!!!!!!!!!!!!!
#!/usr/bin/env python3
# Lee una serie de tiempo desde "18_09_15_09.001(F).txt" (col 1 = tiempo [s], col 2 = señal),
# grafica la serie, realiza FFT, selecciona componentes, compara reconstrucción con original,
# grafica el espectro usado y genera el archivo de entrada "waves_fft.txt".
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('Solarize_Light2')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 3)))

# ================== PARÁMETROS DE EXPORTACIÓN ==================
data_file = "16_09_15_14.001(F).txt"  # archivo de entrada (2 columnas: tiempo, señal)
#data_file = "runb_short.txt"
out_file = "waves_fft.txt"            # archivo de salida
M = 200                                # número objetivo de componentes a exportar
T_min = 2.0                            # periodo mínimo [s] permitido (filtra HF)
match_Hs = True                        # igualar Hs de la suma al Hs de la serie usada
scale_factor = -1.0                     # factor manual si el solver satura (0.9, 0.8, etc.)
phase_sign_export = "+"                # "+" usa sin(ωt+φ), "-" exporta φ invertida (sin(ωt-φ))
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
plt.figure(figsize=(10, 3), dpi=150)
plt.plot(t_u, y_u, 'k', lw=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Señal")
plt.title(f"Serie de tiempo leída: {data_file}")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{fig_prefix}_serie_original.png", dpi=200)
plt.show()

# ================== PREPARACIÓN PARA FFT ==================
# Re-muestreo uniforme (necesario para FFT robusta)
# Definimos un número de muestras que garantice >= M frecuencias positivas disponibles
M_req = M
Nres = max(2*M_req + 8, t_u.size)  # algo mayor para margen
t = np.linspace(t_u[0], t_u[-1], Nres)

# Interpolación lineal
y = np.interp(t, t_u, y_u)

# Remover media (DC) para enfocarnos en oscilaciones
y = y 

# ================== FFT (lado positivo) ==================
dt = float(t[1] - t[0])
Nsig = t.size
Y = np.fft.rfft(y)
f = np.fft.rfftfreq(Nsig, d=dt)

# Amplitudes y fases para senos (k>0 duplicado x2)
amps_full = 2.0 * np.abs(Y) / Nsig
phases_full = np.angle(Y) - np.pi/2.0  # Convertir de coseno a seno: sin(ωt+φ) = cos(ωt+φ-π/2)

# Corrección Nyquist si N es par (ese bin no se duplica)
if Nsig % 2 == 0 and amps_full.size > 1:
    amps_full[-1] *= 0.5

# Excluir DC
f_pos = f[1:]
a_pos = amps_full[1:]
ph_pos = phases_full[1:]

# ================== FILTRO DE BANDA: PERIODO MÍNIMO ==================
# Evita periodos demasiado cortos (altas frecuencias)
fmax_band = 1.0 / T_min
mask_band = f_pos <= fmax_band
f_pos = f_pos[mask_band]
a_pos = a_pos[mask_band]
ph_pos = ph_pos[mask_band]

if f_pos.size == 0:
    raise RuntimeError(f"No quedan componentes después de aplicar T_min={T_min}s. "
                       "Baja T_min o usa una serie más larga.")

# Seleccionar hasta M componentes de menor frecuencia (más energéticas en mar típico)
M_eff = min(M_req, f_pos.size)
f_sel = f_pos[:M_eff]
a_sel = a_pos[:M_eff]
ph_sel = ph_pos[:M_eff]

# Normalizar fase a [0, 2π)
ph_sel = np.mod(ph_sel, 2.0*np.pi)

# Ordenar por periodo descendente (útil para algunos solvers/lecturas)
T_sel = 1.0 / f_sel
order = np.argsort(T_sel)[::-1]  # de mayor periodo a menor
T_sel = T_sel[order]
a_sel = a_sel[order]
ph_sel = ph_sel[order]

# Escala manual opcional
a_sel = a_sel * scale_factor

# ================== AJUSTE DE ENERGÍA (opcional) ==================
if match_Hs:
    Hs_meas = 4.0 * np.std(y)
    m0_comp = 0.5 * np.sum(a_sel**2)
    if m0_comp > 0:
        Hs_comp = 4.0 * np.sqrt(m0_comp)
        gain = Hs_meas / Hs_comp
        a_sel = a_sel * gain
    else:
        gain = 1.0
        Hs_comp = 0.0
else:
    gain = 1.0
    Hs_meas = 4.0 * np.std(y)
    Hs_comp = 4.0 * np.sqrt(0.5*np.sum(a_sel**2)) if a_sel.size else 0.0

# ================== EXPORTACIÓN: fases y unidades ==================
if phase_sign_export == "-":
    ph_out = (-ph_sel) % (2.0*np.pi)
else:
    ph_out = ph_sel

if phase_units.lower().startswith("deg"):
    ph_out_file = np.degrees(ph_out)
else:
    ph_out_file = ph_out

direc = np.zeros_like(a_sel)  # sin dirección (1D)

# Comprobaciones
if not (np.all(np.isfinite(a_sel)) and np.all(np.isfinite(T_sel)) and np.all(np.isfinite(ph_out_file))):
    raise RuntimeError("Valores no finitos en amplitud/periodo/fase tras el procesado.")

# ================== GUARDAR ARCHIVO waves_fft.txt ==================
with open(out_file, "w") as fh:
    fh.write(f"[NumberOfWaves] {a_sel.size}\n")
    fh.write("=================================\n")
    for ai, Ti, di, ph in zip(a_sel, T_sel, direc, ph_out_file):
        fh.write(f"{ai:.6f}\t{Ti:.6f}\t{di:.6f}\t{ph:.6f}\n")

print(f"{out_file} generado con {a_sel.size} componentes.")
print({
    "N_out": int(a_sel.size),
    "T_min_export": float(T_min),
    "min_T_sel": float(np.min(T_sel)),
    "max_T_sel": float(np.max(T_sel)),
    "min_f_sel": float(np.min(1.0/T_sel)),
    "max_f_sel": float(np.max(1.0/T_sel)),
    "Hs_meas": float(Hs_meas),
    "Hs_comp_final": float(4*np.sqrt(0.5*np.sum(a_sel**2))) if a_sel.size else 0.0,
    "gain_aplicado": float(gain),
    "scale_factor": float(scale_factor),
})

# ================== GRÁFICOS DE ANÁLISIS ==================
# Señal usada para la FFT (re-muestreada y de-mean)
plt.figure(figsize=(10, 3), dpi=150)
plt.plot(t, y, 'k', lw=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Elevación (centrada)")
plt.title("Señal usada para la FFT (re-muestreada, sin DC)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{fig_prefix}_senal_usada.png", dpi=200)
plt.show()
#%
# Espectro (lado positivo) y componentes usadas
plt.figure(figsize=(10, 3), dpi=150)
plt.plot(f[1:], amps_full[1:], color='0.6', lw=1, label="Espectro (sin DC)")
# Para comparar puntos seleccionados con la curva gris previa al escalado total:
if match_Hs and (gain != 0):
    a_mark = (a_sel/(gain*scale_factor))[::-1]  # deshacer escalas para marcar
else:
    a_mark = (a_sel/(scale_factor))[::-1]
f_mark = 1.0 / T_sel[::-1]
plt.plot(f_mark, a_mark, 'C1o', label=f"{a_sel.size} seleccionadas (T ≥ {T_min:.2f}s)")
plt.vlines(f_mark, 0, a_mark, colors='C1', lw=0.8)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [u]")
plt.title("Espectro (lado positivo) y componentes usadas")
plt.xlim(left=0)
plt.xlim(0,0.5)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{fig_prefix}_espectro.png", dpi=200)
plt.show()
#%
# Reconstrucción vs original (sobre la malla re-muestreada)
eta_rec = np.zeros_like(t)
for ai, Ti, ph in zip(a_sel, T_sel, ph_sel):
    eta_rec += ai*np.sin(2*np.pi*(1.0/Ti)*t + ph)

plt.figure(figsize=(10, 3), dpi=150)
plt.plot(t, y, 'k', lw=1, label="Original (re-muestreada)")
plt.plot(t, eta_rec, '--r', lw=1, alpha=1, label=f"Reconstrucción ({a_sel.size} términos)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Elevación (centrada)")
plt.title("Reconstrucción vs serie original")
plt.legend()
#plt.xlim(0,20)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{fig_prefix}_reconstruccion.png", dpi=200)
plt.show()

