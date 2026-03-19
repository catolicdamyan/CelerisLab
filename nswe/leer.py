import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import re
import imageio
import os
from scipy import signal
plt.style.use('Solarize_Light2')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 5)))

# Cargar configuración
with open('C:/Users/Catolic Damyan/Downloads/Celeris/prueba2/config.json', 'r') as f:
    config = json.load(f)

# Parámetros de la simulación
nx = config['WIDTH']
ny = config['HEIGHT']
dx = config['dx']
dy = config['dy']

# Cargar batimetría
bathy_data = np.loadtxt('C:/Users/Catolic Damyan/Downloads/Celeris/prueba2/bathy.txt')
bathy = bathy_data.reshape((ny, nx))

# Crear coordenadas
x = np.arange(0, nx) * dx
y = np.arange(0, ny) * dy
X, Y = np.meshgrid(x, y)

# Función para ordenar archivos numéricamente
def sort_files_numerically(files):
    return sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(1)))

# Encontrar y ordenar archivos
elev_files = sort_files_numerically(glob.glob('elev_*.bin'))
time_files = sort_files_numerically(glob.glob('time_*.txt'))
min_len = min(len(elev_files), len(time_files))
elev_files = elev_files[:min_len]
time_files = time_files[:min_len]

# Función para cargar binarios
def load_bin(filename):
    with open(filename, 'rb') as f:
        return np.fromfile(f, dtype=np.float32).reshape((ny, nx))

# Cargar todos los tiempos
times = [float(np.loadtxt(tf)) for tf in time_files]

# ==============================================
# GRÁFICO DE CALOR: Elevación a lo largo de X para todos los tiempos
# ==============================================
# Seleccionar una línea en el centro de Y para visualizar
center_y = ny // 2

# Crear matriz para el heatmap (tiempos x posiciones X)
elevation_matrix = np.zeros((len(times), nx))

# Llenar la matriz con datos de elevación
for idx in range(len(times)):
    # Cargar datos de elevación
    eta = load_bin(elev_files[idx])
    
    # Obtener la elevación a lo largo de X en el centro de Y
    elevation_matrix[idx, :] = eta[center_y, :]

# Crear figura para el gráfico de calor
plt.figure( dpi=400)

# Crear heatmap
heatmap = plt.imshow(elevation_matrix, 
                    aspect='auto', 
                    extent=[x[0], x[-1], times[0], times[-1]], 
                    origin='lower',
                    cmap='viridis')

plt.colorbar(heatmap, label=r'$eta$ [m]')
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$t$ [s]')
plt.grid(False)
plt.tight_layout()
plt.savefig('heatmap_elevacion_x_tiempo.png', bbox_inches='tight', dpi=300)
plt.show()
#%% ==============================================
# TERCER GRÁFICO: Curtosis espacial en función del tiempo
# ==============================================
# Función para identificar olas basado en cruces por cero
def identify_waves(elevation_data, mean_level=0.0):
    # Encontrar cruces por cero
    zero_crossings = np.where(np.diff(np.sign(elevation_data - mean_level)))[0]
    
    # Identificar crestas y valles entre cruces por cero
    wave_heights = []
    for i in range(len(zero_crossings)-1):
        segment = elevation_data[zero_crossings[i]:zero_crossings[i+1]]
        crest = np.max(segment)
        trough = np.min(segment)
        wave_height = crest - trough
        wave_heights.append(wave_height)
    
    return np.array(wave_heights)

# Calcular altura de olas y curtosis para cada tiempo
spatial_kurtosis = []

for i, ef in enumerate(elev_files):
    eta = load_bin(ef)
    
    # Calcular altura de olas a lo largo de la línea central
    elevation_center = eta[center_y, :]
    wave_heights = identify_waves(elevation_center)
    
    if len(wave_heights) > 0:
        # Calcular curtosis de las alturas de ola
        mean_val = np.mean(wave_heights)
        std_val = np.std(wave_heights)
        m4 = np.mean((wave_heights - mean_val)**4)
        kurtosis_val = m4 / (std_val**4)
        spatial_kurtosis.append(kurtosis_val)
    else:
        spatial_kurtosis.append(0.0)

# Crear gráfico de curtosis
plt.figure(figsize=(12, 6))
plt.plot(times, spatial_kurtosis, 'r-', linewidth=2)
plt.xlabel('Tiempo (s)')
plt.ylabel('Curtosis de alturas de ola')
plt.title('Evolución de la curtosis de alturas de ola')
plt.grid(True)
plt.tight_layout()
plt.savefig('curtosis_alturas_ola.png', bbox_inches='tight')
plt.show()

# %%==============================================
# CUARTO: Crear GIF de la evolución temporal
# ==============================================
# Crear directorio temporal para las imágenes
if not os.path.exists('temp_frames'):
    os.makedirs('temp_frames')

# Crear frames para el GIF
frames = []
for i, ef in enumerate(elev_files):
    # Cargar datos de elevación
    eta = load_bin(ef)
    
    # Obtener la elevación a lo largo de X en el centro de Y
    elevation_along_x = eta[center_y, :]
    
    # Crear figura
    plt.figure(figsize=(10, 6))
    plt.plot(x, elevation_along_x, '-', color='black', linewidth=5)
    plt.xlabel('Distancia X (m)')
    plt.ylabel('Elevación (m)')
    plt.title(f'Tiempo = {times[i]:.2f} s')
    plt.grid(True)
    plt.xlim(0, max(x))
    plt.ylim(-0.05, 0.05)
    # Guardar frame temporal
    frame_path = f'temp_frames/frame_{i:04d}.png'
    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    frames.append(frame_path)

# Crear GIF
with imageio.get_writer('evolucion_superficie_libre.gif', mode='I', fps=25) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

# Limpiar archivos temporales
for frame in frames:
    os.remove(frame)
os.rmdir('temp_frames')