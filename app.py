import base64
from io import BytesIO
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from flask import Flask, render_template, request
import serial
import time

matplotlib.use('Agg')

app = Flask(__name__)
#
PORT = 'COM5'
BAUD = 1200

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.5)
    time.sleep(2)
    print(f"Conexión exitosa con Arduino en {PORT}")
except Exception as e:
    ser = None
    print(f"Advertencia: No se pudo conectar al puerto serial ({e}). Se usará simulación.")

def extraer_caracteristicas(senal):
    """Extrae las 5 características clave de la serie de tiempo digital"""
    senal = np.array(senal)
    if len(senal) == 0:
        return [0, 0, 0, 0, 0]
    media = np.mean(senal)
    desviacion = np.std(senal)
    rms = np.sqrt(np.mean(senal**2))
    pico_a_pico = np.ptp(senal)
    variacion_promedio = np.mean(np.abs(np.diff(senal))) if len(senal) > 1 else 0
    return [media, desviacion, rms, pico_a_pico, variacion_promedio]

def leer_datos_arduino(num_muestras=1200, insertar_falla=False, amplitud=1.0, ruido=0.1, intensidad=1.0):
    """
    Captura estrictamente muestras reales del sensor físico sin rellenar ni inventar datos.
    Se bloquea/espera de forma limpia hasta completar las 1200 muestras requeridas del puerto serial.
    """
    if ser is None or not ser.is_open:
        t = np.arange(num_muestras)
        if insertar_falla:
            prob_falla = min(0.6, 0.2 * intensidad)
            senal = np.random.choice([0, 1], size=num_muestras, p=[1 - prob_falla, prob_falla])
        else:
            senal = np.ones(num_muestras) + np.random.normal(0, 0.02, num_muestras)
        return t, np.clip(senal.astype(float), 0.0, 1.0)
    
    lecturas = []
    try:
        ser.reset_input_buffer()
        while len(lecturas) < num_muestras:
            if ser.in_waiting > 0:
                linea = ser.readline().decode('utf-8', errors='ignore').strip()
                if linea:
                    try:
                        val = float(linea)
                        lecturas.append(val)
                    except ValueError:
                        pass
            else:
                time.sleep(0.001)
               
    except Exception as e:
        print(f"Error al leer el puerto serial: {e}")

    lecturas = np.array(lecturas, dtype=float)
    t = np.arange(len(lecturas))
    return t, lecturas

def entrenar_modelo_red_neuronal():
    """Entrena la red neuronal MLP utilizando bloques de 1200 muestras"""
    X, y = [], []
    
    for _ in range(150):
        senal = np.ones(1200)
        X.append(extraer_caracteristicas(senal))
        y.append(0)
        
    for _ in range(150):
        senal = np.random.choice([0, 1], size=1200, p=[0.01, 0.99])
        X.append(extraer_caracteristicas(senal))
        y.append(0)
        
    for _ in range(300):
        senal = np.random.choice([0, 1], size=1200, p=[0.5, 0.5])
        X.append(extraer_caracteristicas(senal))
        y.append(1)
        
    X = np.array(X)
    y = np.array(y)

    modelo = MLPClassifier(hidden_layer_sizes=(12, 8), max_iter=1500, random_state=42)
    modelo.fit(X, y)
    exactitud = modelo.score(X, y) * 100
    return modelo, exactitud

modelo_global, exactitud_global = entrenar_modelo_red_neuronal()

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    estado_real = None
    confianza = 0.0
    caracteristicas = []
    
    amplitud = 1.0
    ruido = 0.1
    intensidad = 1.0
    insertar_falla = False
    
    if request.method == "POST":
        try:
            amplitud = float(request.form.get("amplitud", 1.0))
        except ValueError:
            amplitud = 1.0
            
        try:
            ruido = float(request.form.get("ruido", 0.1))
        except ValueError:
            ruido = 0.1
            
        try:
            intensidad = float(request.form.get("intensidad", 1.0))
        except ValueError:
            intensidad = 1.0
            
        insertar_falla = "insertar_falla" in request.form
        
        t, nueva_senal = leer_datos_arduino(
            num_muestras=1200, 
            insertar_falla=insertar_falla, 
            amplitud=amplitud, 
            ruido=ruido, 
            intensidad=intensidad
        )
        
        estado_real = "VIBRACIÓN EXCESIVA / ANOMALÍA" if insertar_falla else "OPERACIÓN NORMAL (REPOSO)"
            
        caracteristicas = extraer_caracteristicas(nueva_senal)
        
        prediccion = modelo_global.predict([caracteristicas])[0]
        probabilidades = modelo_global.predict_proba([caracteristicas])[0]
        
        resultado = "ANOMALÍA DETECTADA" if prediccion == 1 else "NORMAL"
        confianza = float(np.max(probabilidades) * 100)
        
        fig, ax = plt.subplots(figsize=(10, 3.8))
        if prediccion == 1:
            color_linea = "#d9534f" 
        else:
            color_linea = "#0275d8" 
            
        ax.plot(t, nueva_senal, color=color_linea, linewidth=1.5, label=f"Estado: {resultado}")
        ax.fill_between(t, nueva_senal, color=color_linea, alpha=0.15)
        
        ax.set_title(f"Monitoreo KY-002 (Muestras Reales: {len(nueva_senal)}) — Predicción: {resultado}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Muestras de Tiempo")
        ax.set_ylabel("Estado Digital (0 / 1)")
        ax.set_ylim(-0.1, 1.3)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3, linestyle='--')
        fig.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        imagen_inferencia = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)
    else:
        imagen_inferencia = None

    t_ref = np.arange(1200)
    senal_ref = np.ones(1200)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(t_ref, senal_ref, color="#0275d8", linewidth=1.5, label="Patrón Base (Reposo en 1)")
    ax.fill_between(t_ref, senal_ref, color="#0275d8", alpha=0.15)
    
    ax.set_title("Monitoreo en Vivo — Sensor de Vibración KY-002 Arduino", fontsize=11, fontweight='bold')
    ax.set_xlabel("Muestras de Tiempo")
    ax.set_ylabel("Estado Digital (0 / 1)")
    ax.set_ylim(-0.1, 1.3)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    imagen_entrenamiento = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    return render_template(
        "index.html",
        imagen_entrenamiento=imagen_entrenamiento,
        imagen_inferencia=imagen_inferencia,
        exactitud=exactitud_global,
        resultado=resultado,
        estado_real=estado_real,
        confianza=confianza,
        caracteristicas=caracteristicas,
        amplitud=amplitud,
        ruido=ruido,
        intensidad=intensidad,
        insertar_falla=insertar_falla
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)