import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Función para cargar imagen y CSV desde rutas
def cargar_datos(ruta_imagen, ruta_csv):
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Cargar CSV
    datos = pd.read_csv(ruta_csv)

    return imagen, datos

# Función para generar el Heatmap
def generar_heatmap(imagen, datos, titulo='Heatmap'):
    plt.figure(figsize=(12, 8))
    plt.imshow(imagen)

    sns.kdeplot(
        x=datos['x'],
        y=datos['y'],
        cmap='plasma',  # o cualquier otro colormap
        fill=True,
        alpha=0.5,
        thresh=0.05,
        levels=100
    )

    plt.axis('off')
    plt.title(titulo)
    plt.show()



# ------------------------- EJECUTAR -------------------------
# Especifica las rutas de los archivos
ruta_mockup_bueno = 'mockup_bueno.png'
ruta_datos_bueno = 'datos_mockup_bueno_realista(1).csv'

ruta_mockup_malo = 'mockup_malo.png'
ruta_datos_malo = 'datos_mockup_malo_realista(1).csv'

# Cargar datos
imagen_bueno, datos_bueno = cargar_datos(ruta_mockup_bueno, ruta_datos_bueno)
imagen_malo, datos_malo = cargar_datos(ruta_mockup_malo, ruta_datos_malo)

# Generar Heatmaps
generar_heatmap(imagen_bueno, datos_bueno, titulo='Heatmap - Mockup Bueno')
generar_heatmap(imagen_malo, datos_malo, titulo='Heatmap - Mockup Malo')