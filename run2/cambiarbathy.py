def modificar_batimetria(archivo_entrada, archivo_salida, nuevo_valor):
    with open(archivo_entrada, 'r') as file:
        lineas = file.readlines()
    
    with open(archivo_salida, 'w') as file:
        for linea in lineas:
            # Si la línea está vacía, conservarla tal cual
            if not linea.strip():
                file.write(linea)
                continue
                
            # Dividir los valores preservando los espacios
            valores = linea.split()
            # Crear nuevos valores con el formato correcto
            nuevos_valores = [f"{nuevo_valor:.6f}" for _ in valores]
            # Reconstruir la línea exactamente como estaba
            nueva_linea = " ".join(nuevos_valores)
            
            # Conservar el salto de línea original
            if linea.endswith('\n'):
                nueva_linea += '\n'
                
            file.write(nueva_linea)

# Configuración
archivo_original = "bathy.txt"
archivo_modificado = "bathy.txt"
nueva_profundidad = -0.02578  # Cambia este valor al que necesites

modificar_batimetria(archivo_original, archivo_modificado, nueva_profundidad)
