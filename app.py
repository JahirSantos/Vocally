from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy import signal
from scipy.fft import fft
import os
from pydub import AudioSegment

app = Flask(__name__, static_folder='static')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/why')
def why():
    return render_template('why.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/Acustico')
def acustico():
    return render_template('acustico.html')

# Ruta para manejar la solicitud de análisis de audio
@app.route('/analizar_audio', methods=['POST'])
def analizar_audio():
    fs_37, audio_37 = wavfile.read('static/audios/recording.wav')

    #Umbral por porcentaje de la Amplitud Máxima está entre 10 y 20%
    umbral_amplitud = 0.01 * np.max(np.abs(audio_37))

    # Encontrar segmentos activos
    segmentos_activos = []
    segmento_actual = []
    tiempo_fonacion_total = 0.0

    for i in range(len(audio_37)):
        if np.abs(audio_37[i]) > umbral_amplitud:
            segmento_actual.append(i)
        elif len(segmento_actual) > 0:
            duracion_segmento = len(segmento_actual) / fs_37
            segmentos_activos.append((segmento_actual[0] / fs_37, (segmento_actual[-1] + 1) / fs_37))
            tiempo_fonacion_total += duracion_segmento
            segmento_actual = []

    # Verificar si el último segmento es activo
    if len(segmento_actual) > 0:
        duracion_segmento = len(segmento_actual) / fs_37
        segmentos_activos.append((segmento_actual[0] / fs_37, (segmento_actual[-1] + 1) / fs_37))
        tiempo_fonacion_total += duracion_segmento

    # Imprimir el tiempo de fonación total y los segmentos activos si es necesario
    print(f"Tiempo total de fonación: {tiempo_fonacion_total} segundos")


    # Vector de tiempo inicializado
    tiempo_audio = np.zeros_like(audio_37, dtype=float)

    # Conversión de muestras a tiempo
    for i in range(1, len(audio_37)):
        tiempo_audio[i] = tiempo_audio[i - 1] + 1 / fs_37

    # Encontrar el índice más cercano al segundo 2 y 3 en el vector de tiempo
    indice_segundo_2 = np.argmin(np.abs(tiempo_audio - 2))
    indice_segundo_3 = np.argmin(np.abs(tiempo_audio - 3))

    # Recortar la señal de audio utilizando los índices del segundo 2 y 3
    audio_37 = audio_37[indice_segundo_2:indice_segundo_3]
    tiempo_recortado = tiempo_audio[indice_segundo_2:indice_segundo_3]

    # Encontrar picos en la señal de audio
    pospic_37, _ = find_peaks(audio_37, distance=460, prominence=0.018)

    # Ajustar el vector de tiempo para que comience en 0 segundos
    tiempo_ajustado = tiempo_recortado - tiempo_recortado[0]

    # Invertir la señal de audio
    audio_invertida = -audio_37

    # Encontrar picos en la señal invertida
    pospic_37i, _ = find_peaks(audio_invertida, distance=460, prominence=0.018)

    # Vector de tiempo para la señal invertida
    tiempo_invertido = np.arange(len(audio_invertida)) / fs_37

    # Convertir posiciones de picos a tiempo
    tiempo_picos = pospic_37 / fs_37
    dife_tiempo = np.diff(tiempo_picos)
    N = len(dife_tiempo)


    # Calculo de jitta y jitt
    titi_1 = np.zeros(N)
    for k in range(1, N):
        titi_1[k] = abs(dife_tiempo[k] - dife_tiempo[k - 1])

    suma_ti = np.sum(titi_1)
    jitta = suma_ti / (N - 1)
    sumati = np.sum(dife_tiempo)
    jitt = (jitta / ((1 / N) * sumati)) * 100

    # Calculo de Tn
    dif_tn = np.zeros(N - 2)
    for p in range(1, N - 1):
        tn = (dife_tiempo[p - 1] + dife_tiempo[p] + dife_tiempo[p + 1]) / 3
        dif_tn[p - 1] = abs(dife_tiempo[p] - tn)

    sumasuma = np.sum(dif_tn)
    sumati = np.sum(dife_tiempo)
    numerador_rap = (1 / (N - 1)) * sumasuma
    denominador_rap = (1 / N) * sumati
    rap = (numerador_rap / denominador_rap) * 100

    # Calculo de shimmer
    Apico_37 = audio_37[pospic_37]  # Obtener valores de los picos
    Ai_Ait1 = np.abs(np.diff(Apico_37))

    sumatoria_Ais = np.sum(Ai_Ait1)
    numerador_shim = sumatoria_Ais / (N - 1)
    sumatoria_Ai = np.sum(Apico_37)
    denominador_shim = (sumatoria_Ai / N)
    shimmer = (numerador_shim / denominador_shim) * 100

    # Imprimir valores de jitt, jitta, rap y shimmer para la señal 
    print(f"jitt : {jitt}%")
    print(f"jitta : {jitta}%")
    print(f"rap : {rap}%")
    print(f"shimmer : {shimmer}%")

    # Convertir posiciones de picos a tiempo
    tiempo_invertido_picos = pospic_37i / fs_37
    dife_tiempo_invertido = np.diff(tiempo_invertido_picos)
    N_invertido = len(dife_tiempo_invertido)

    # Cálculo de jitta y jitt para la señal invertida
    titi_1_invertido = np.zeros(N_invertido)
    for k in range(1, N_invertido):
        titi_1_invertido[k] = abs(dife_tiempo_invertido[k] - dife_tiempo_invertido[k - 1])

    suma_ti_invertido = np.sum(titi_1_invertido)
    jitta_invertido = suma_ti_invertido / (N_invertido - 1)
    sumati_invertido = np.sum(dife_tiempo_invertido)
    jitt_invertido = (jitta_invertido / ((1 / N_invertido) * sumati_invertido)) * 100

    # Cálculo de Tn para la señal invertida
    dif_tn_invertido = np.zeros(N_invertido - 2)
    for p in range(1, N_invertido - 1):
        tn_invertido = (dife_tiempo_invertido[p - 1] + dife_tiempo_invertido[p] + dife_tiempo_invertido[p + 1]) / 3
        dif_tn_invertido[p - 1] = abs(dife_tiempo_invertido[p] - tn_invertido)

    sumasuma_invertido = np.sum(dif_tn_invertido)
    sumati_invertido = np.sum(dife_tiempo_invertido)
    numerador_rap_invertido = (1 / (N_invertido - 1)) * sumasuma_invertido
    denominador_rap_invertido = (1 / N_invertido) * sumati_invertido
    rap_invertido = (numerador_rap_invertido / denominador_rap_invertido) * 100

    # Calculo de shimmer para la señal invertida
    Apico_37i = audio_37[pospic_37i]  # Obtener valores de los picos
    Ai_Ait1_invertido = np.abs(np.diff(Apico_37i))

    sumatoria_Ais_invertido = np.sum(Ai_Ait1_invertido)
    numerador_shim_invertido = sumatoria_Ais_invertido / (N_invertido - 1)
    sumatoria_Ai_invertido = np.sum(Apico_37i)
    denominador_shim_invertido = (sumatoria_Ai_invertido / N_invertido)
    shimmer_invertido = (numerador_shim_invertido / denominador_shim_invertido) * 100

    # Imprimir valores de jitt, jitta, rap y shimmer para la señal invertida
    print(f"jitt (señal invertida): {jitt_invertido}")
    print(f"jitta (señal invertida): {jitta_invertido}")
    print(f"rap (señal invertida): {rap_invertido}")
    print(f"shimmer (señal invertida): {shimmer_invertido}")

    #CALCULO DE PPQ5 = 5* desviacion estandar/ media / cuestionable
    std_audio = np.std(audio_37)
    mean_audio = np.mean(audio_37)
    ppq5 = (5 * std_audio) / mean_audio
    print(f"mean: {mean_audio}")
    print(f"PPQ5: {ppq5}")

    # calculo de PPQ3 = 3* desviacion estandar/ media 
    ppq3= (3*std_audio)/ mean_audio
    print(f"PPQ3: {ppq3}")

    # Cálculo de la desviación estándar y la media de la señal de audio
    std_audio = np.std(audio_37)
    mean_audio = np.mean(audio_37)

    # Cálculo de la relación señal-ruido
    ruido = np.random.normal(0, 1, len(audio_37))
    relacion_senal_ruido = 10 * np.log10(np.sum(audio_37 ** 2) / np.sum(ruido ** 2))
    print(f"Relación señal-ruido: {relacion_senal_ruido:.2f} dB")

    # Calcular la transformada de Fourier
    fourier = fft(audio_37)
    print(f"trans fourier: {fourier}")

    # Calcular la magnitud de la transformada de Fourier
    magnitud = np.abs(fourier)

    # Cálculo de la frecuencia fundamental y los armónicos
    _, espectro = signal.welch(audio_37, fs_37, nperseg=770)
    frecuencias = signal.find_peaks(espectro, height=0.1)[0]
    frecuencia_fundamental = frecuencias[0]
    armónicos = frecuencias[1:]
    print(f"Frecuencia fundamental: {frecuencia_fundamental:.2f} Hz")
    print(f"Armónicos: {armónicos}")

    # Calcular la frecuencia correspondiente a cada punto de la transformada de Fourier
    frecuencia = np.linspace(0, fs_37, len(magnitud))

    # verificar si los armónicos están equilibrados y son armónicos de la frecuencia fundamental
    if all(h % frecuencia_fundamental == 0 for h in armónicos):
        print("La señal de audio tiene un buen timbre.")
    else:
        print("La señal de audio tiene un timbre pobre o desagradable.")

    # Cálculo de la distorsión armónica
    distorsión_armónica = np.sum(espectro[armónicos]) / np.sum(espectro)
    print(f"Distorsión armónica: {distorsión_armónica:.2f}")

    # Cálculo de la potencia
    potencia = np.sum(audio_37 ** 2) / len(audio_37)
    print(f"Potencia: {potencia:.2f}")

    # Cálculo de la frecuencia de muestreo y la profundidad de bits
    info_audio = wavfile.read('static/audios/recording.wav')
    frecuencia_muestreo = info_audio[0]
    profundidad_bits = info_audio[1]
    print(f"Frecuencia de muestreo: {frecuencia_muestreo} Hz")
    print(f"Profundidad de bits: {profundidad_bits} bits")
    
    # Después de realizar el análisis, devuelve los resultados como un objeto JSON
    resultados = {
        'jitt': jitt,
        'jitta': jitta,
        'rap': rap,
        'shimmer': shimmer,
        'ppq5': ppq5,
        'ppq3':ppq3,
        #'relacion_senal_ruido' : relacion_senal_ruido,
        'distorsion_armonica': distorsión_armónica,
        'potencia' : potencia,
        'frecuenciamuestreo' : frecuencia_muestreo,
        'mean':mean_audio,
        # Agrega aquí otros resultados que desees devolver
    }

    return jsonify(resultados)

@app.route('/guardar_audio', methods=['POST'])
def guardar_audio():
    archivo_audio = request.files['audio']
    archivo_audio.save(os.path.join(app.static_folder, 'audios', 'recording.webm'))
    return jsonify({'mensaje': 'Archivo de audio guardado correctamente'})

def convert_webm_to_wav(input_file, output_file):
    # Carga el archivo WebM
    audio = AudioSegment.from_file(input_file, format="webm")

    # Guarda el archivo como WAV
    audio.export(output_file, format="wav")

@app.route('/convertir_webm_a_wav', methods=['POST'])
def convertir_webm_a_wav():
    input_file = os.path.join(app.static_folder, 'audios', 'recording.webm')  # Ruta del archivo de entrada WebM
    output_file = os.path.join(app.static_folder, 'audios', 'recording.wav')  # Ruta del archivo de salida WAV

    convert_webm_to_wav(input_file, output_file)

    return jsonify({'mensaje': 'Conversión de WebM a WAV exitosa'})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=os.getenv("PORT",default=5000))