'''
fig 1
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from matplotlib import animation
from matplotlib.animation import PillowWriter
from kdev_solver import kdev_solver
plt.style.use('Solarize_Light2')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 4)))
carpeta = r"C:\Users\damya\OneDrive - Universidad Católica de Chile\Escritorio\Escritorio\Proyectos\仕事\究明\Civil\Tesis\solver\CI\\"
#%% Parámetros de la simulación
h = 0.2#1.39
g = 9.8
c = np.sqrt(g*h)#0.185
alpha = 3/(2*c*h)
beta = h**2/(6*c**3)
mu = beta/(c**2*(1-c**2)) #0.4, 0.06
eps = alpha/(1-c**2) #0.2
ds = 0.07
dt = 0.1
L = 14.1
T = 9
mx = int(L/ds)
t_max = int(T/dt)   # Tiempo máximo equivalente al anterior
time_steps = t_max
# Definir la onda inicial
def onda_inicial(x):
    return 0.0184*np.cos(x * 2 * np.pi / (mx * ds))
def deriv_central(s, ds):
    ux = np.gradient(s, ds, axis=0)  # Derivada respecto al espacio (centrada)
    return ux
# Ejecutar la simulación
spl, dt, ds, m = kdev_solver(mu, eps, ds, mx, t_max, dt, time_steps, onda_inicial)
# Preparar datos para gráfica
x = np.linspace(0, (mx - 1) * ds, mx)  # Escala de posición en el espacio
y = np.linspace(0, (time_steps - 1) * dt, time_steps)  # Escala de tiempo
X, Y = np.meshgrid(x, y)
# Crear figura y ejes 3D
fig = plt.figure(2, dpi=200)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(65, 80)
# Graficar la onda en el espacio y tiempo reales
ax.plot_wireframe(-X, -Y, spl.T, rstride=5, cstride=5, linewidth=1,color = 'black')
# Etiquetas de los ejes
ax.set_xlabel('Position (x)')
ax.set_ylabel('Time (t)')
ax.set_zlabel('Disturbance (u)')
ax.grid(False)
ax.axis(False)
fig.savefig(carpeta + 'joydivision_1.png')
fig.savefig(carpeta + 'joydivision_1.svg')
plt.show()
#% Gráfica de calor
plt.figure(3, dpi=200)
plt.imshow(spl, aspect='auto', extent=[0, (time_steps * dt), 0, mx*ds], origin='lower', cmap='viridis')
plt.colorbar(label=r'$u$')
plt.grid(False)
plt.ylabel(r'$x$ [m]')
plt.xlabel(r'$t$ [s]')
plt.savefig(carpeta + 'calor_1.svg')
plt.savefig(carpeta + 'calor_1.png')
plt.show()
#%%
ux = deriv_central(spl, ds)/ds
#% Cálculo de energía total por tiempo
energia_por_tiempo = np.sum(spl**2 + mu**2 * ux**2, axis=0) * ds
#% Normalización de la energía
energia_normalizada = energia_por_tiempo / energia_por_tiempo[0]
# Graficar energía normalizada a lo largo de x (promediada)
plt.figure(dpi=400)
plt.plot(np.arange(0, len(energia_normalizada)) * dt, energia_normalizada, label='Energía normalizada',color=plt.cm.viridis(0), linewidth=5)
plt.xlabel('Tiempo')
plt.ylabel('Energía normalizada')
plt.grid(True)
plt.ylim(0,1.2)
plt.legend()
plt.savefig(carpeta + 'energia_normalizada_1.png')
plt.show()
#% Espacio de fase promediado
plt.figure(dpi=200)
for i in range(mx):  # Graficar el espacio de fase para cada x
    ut_fijo = np.gradient(spl[i, :], dt)  # Derivada temporal en x
    u_fijo = spl[i, :]
    plt.plot(u_fijo, ut_fijo, linewidth=2.5, color=plt.cm.viridis(0.6))  # Dibujar cada curva de fase individualmente
plt.xlabel('u')
plt.ylabel("u'")
plt.grid(True)
plt.tight_layout()
plt.title('Espacio de fase a lo largo de x')
plt.savefig(carpeta + 'espacio_fase_todo_x_1.png')
plt.show()
#%
plt.figure(4, dpi=200)
for i in (0, round(len(spl[0, :])/3.67), len(spl[0, :])-1):
    plt.plot(x/(ds*mx), spl[:, i], label=r'$t = '+str(round((i * dt), 1))+' [s]$', linewidth=5)
    plt.title('Soliton en el espacio')
    plt.tight_layout()
    plt.xlabel(r'Normalized distance')
    plt.ylabel(r'')
    plt.grid(True)
plt.legend()
plt.savefig(carpeta + 'soliton_1.svg')
plt.savefig(carpeta + 'soliton_1.png')
plt.show()

#%fourier
plt.figure(5, dpi=400)
for i in (0, round(len(spl[0, :])/3.67), len(spl[0, :])-1): 
    u_x = spl[:, i]   # Eliminar el valor medio
    fft_u = np.fft.fft(u_x) / len(u_x)  # FFT sin shift
    k = (np.fft.fftfreq(len(x), ds)) * 10*np.pi  # Números de onda k
    pos_mask = k >= 0  # Tomar solo valores positivos de k
    plt.plot(k[pos_mask], np.abs(fft_u[pos_mask]), label=r'$t = '+str(round((i * dt), 1))+' [s]$', linewidth=5)

plt.xlabel(r'wave number $k$')
plt.ylabel('Amplitud')
plt.title('FFT')
plt.xlim(0,21)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(carpeta + 'fourier_espacial_soliton.png')
plt.savefig(carpeta + 'fourier_espacial_soliton.svg')
plt.show()
#%Función de dominancia espectral
def dominancia_espectral(fft_u):
    P_k = np.abs(fft_u) ** 2
    return np.max(P_k) / np.sum(P_k)

# Configuración de la figura
plt.figure(6, dpi=400)

# Iteración sobre todos los tiempos
for i in range(len(spl[0, :])):
    u_x = spl[:, i]  # Eliminar el valor medio (si es necesario)
    fft_u = np.fft.fft(u_x) / len(u_x)  # FFT sin shift
    k = (np.fft.fftfreq(len(x), ds)) * 10 * np.pi  # Números de onda k (ajustados para tu caso)
    
    pos_mask = k >= 0  # Tomar solo valores positivos de k
    k_pos = k[pos_mask]
    fft_pos = np.abs(fft_u[pos_mask])
    
    # Calcular la dominancia espectral
    dominancia = dominancia_espectral(fft_u)

    # Graficar la dominancia espectral vs. tiempo
    plt.plot(i * dt, dominancia, 'd', linewidth=2, color=plt.cm.viridis(0.1))  # Tiempo vs Dominancia espectral

# Etiquetas y título
plt.xlabel(r'Tiempo $t$')
plt.ylabel(r'Índice de dominancia espectral')
plt.title('Evolución de la dominancia espectral')
plt.grid(True)
plt.tight_layout()

# Guardar imagen
plt.savefig(carpeta + 'dominancia_espectral_soliton.png')
plt.savefig(carpeta + 'dominancia_espectral_soliton.svg')
plt.show()
# % el GIF
def plot_gif(spl, dt, ds, nombre='soliton_evolution.gif'):
    fig, ax = plt.subplots(figsize=(8, 6))
    ln, = plt.plot([], [], '-', color='orange', linewidth=5)
    ax.set_xlim(0, mx * ds)
    ax.set_ylim(np.min(spl), np.max(spl))
    ax.set_title('Soliton Evolution')
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$u$ [m]')
    ax.grid(True)

    def init():
        ln.set_data([], [])
        return ln,

    def animate(i):
        ln.set_data(x, spl[:, i])
        return ln,

    ani = animation.FuncAnimation(fig, animate, frames=t_max, init_func=init, blit=True, interval=50)
    ani.save(nombre, writer='ffmpeg', fps=15)

plot_gif(spl, dt, ds, carpeta + 'soliton_evolution_1.gif')
#%% comparacion
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from matplotlib import animation
from matplotlib.animation import PillowWriter
from kdev_solver import kdev_solver
plt.style.use('Solarize_Light2')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 4)))
carpeta = r"C:\Users\damya\OneDrive - Universidad Católica de Chile\Escritorio\Escritorio\Proyectos\仕事\究明\Civil\Tesis\solver\CI\\"


#%simulacion
plt.figure(8, dpi = 300, figsize=(8, 10))
n = 0
idem = [1,3,5,7,2,4,6,8]
for i in np.arange(0,t_max,t_max//8+1):
    plt.subplot(4,2,idem[n])
    p1, = plt.plot(x+L, 100*spl[:, i],'-' , linewidth = 2.5,label='KdV')
    plt.plot(x+2*L, 100*spl[:, i],'-' ,color = p1.get_color(), linewidth = 2.5,label='_no_legend_')
    plt.xlim(min(x+L),max(x+2*L))
    n = n + 1
offx = 4.3
df = pd.read_csv(carpeta+'/trillo1.csv', header=None)
df.columns = ['X', 'Y']
plt.subplot(4,2,1)
offset = (max(df['Y'])+min(df['Y']))/2
plt.plot(df['X']-offx, df['Y']-offset,'-', linewidth = 1,label='Trillo')
plt.title('(a) 5m')
plt.tight_layout()
plt.legend()
plt.grid(True)
df = pd.read_csv(carpeta+'/trillo2.csv', header=None)
df.columns = ['X', 'Y']
plt.subplot(4,2,2)
plt.plot(df['X']-3*T-offx, df['Y']-offset,'-', linewidth = 1,label='Trillo')
plt.title('(e) 45m')
plt.tight_layout()
plt.legend()
df = pd.read_csv(carpeta+'/trillo3.csv', header=None)
df.columns = ['X', 'Y']
plt.subplot(4,2,3)
plt.plot(df['X']-T-0.5*offx, df['Y']-offset,'-', linewidth = 1,label='Trillo')
plt.title('(b) 15m')
plt.tight_layout()
plt.legend()
df = pd.read_csv(carpeta+'/trillo4.csv', header=None)
df.columns = ['X', 'Y']
plt.subplot(4,2,4)
plt.plot(df['X']-4*T-offx, df['Y']-offset,'-', linewidth = 1,label='Trillo')
plt.title('(f) 55m')
plt.tight_layout()
plt.legend()
df = pd.read_csv(carpeta+'/trillo5.csv', header=None)
df.columns = ['X', 'Y']
plt.subplot(4,2,5)
plt.plot(df['X']-4*offx, df['Y']-offset,'-', linewidth = 1,label='Trillo')
plt.title('(c) 25m')
plt.tight_layout()
plt.legend()
df = pd.read_csv(carpeta+'/trillo6.csv', header=None)
df.columns = ['X', 'Y']
plt.subplot(4,2,6)
plt.plot(df['X']-3.5*T, df['Y']-offset,'-', linewidth = 1,label='Trillo')
plt.title('(g) 65m')
plt.tight_layout()
plt.legend()
df = pd.read_csv(carpeta+'/trillo7.csv', header=None)
df.columns = ['X', 'Y']
plt.subplot(4,2,7)
plt.plot(df['X']-3*T, df['Y']-offset,'-', linewidth = 1,label='Trillo')
plt.title('(d) 35m')
plt.tight_layout()
plt.legend()
df = pd.read_csv(carpeta+'/trillo8.csv', header=None)
df.columns = ['X', 'Y']
plt.subplot(4,2,8)
plt.plot(df['X']-4*T-offx, df['Y']-offset,'-', linewidth = 1,label='Trillo')
plt.title('(h) 75m')
plt.legend()
plt.tight_layout()

plt.savefig(carpeta + 'trillo.png')

plt.show()

