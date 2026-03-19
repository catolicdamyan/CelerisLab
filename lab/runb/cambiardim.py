import numpy as np

def redimensionar_batimetria(archivo_entrada, archivo_salida, nueva_cols, nueva_filas=None):
    # Leer archivo
    with open(archivo_entrada, 'r') as file:
        lineas = [l.strip() for l in file if l.strip()]
    
    # Convertir a matriz numpy (float)
    matriz = np.array([list(map(float, l.split())) for l in lineas])
    
    filas, cols = matriz.shape
    if nueva_filas is None:
        nueva_filas = filas  # Mantener filas iguales si no se especifica
    
    # Crear nuevo grid
    x_original = np.linspace(0, 1, cols)
    x_nuevo = np.linspace(0, 1, nueva_cols)
    
    matriz_nueva = np.zeros((nueva_filas, nueva_cols))
    
    for i in range(min(filas, nueva_filas)):
        matriz_nueva[i, :] = np.interp(x_nuevo, x_original, matriz[i, :])
    
    # Si hay más filas nuevas que originales → repetir últimas
    if nueva_filas > filas:
        for i in range(filas, nueva_filas):
            matriz_nueva[i, :] = matriz[-1, :]
    
    # Guardar archivo
    with open(archivo_salida, 'w') as file:
        for fila in matriz_nueva:
            linea = " ".join(f"{v:.6f}" for v in fila)
            file.write(linea + "\n")

# Configuración
archivo_original = "bathy.txt"
archivo_modificado = "bathy.txt"
nueva_cols = 4500 # nuevas columnas
nueva_filas = 7    # mantener 10 filas

redimensionar_batimetria(archivo_original, archivo_modificado, nueva_cols, nueva_filas)
