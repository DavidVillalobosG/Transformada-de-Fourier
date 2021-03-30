import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.fft import ifft

#Inicio código de referencia para señal ruidosa

# Parámetros
tasa_muestreo = 1024
deltaT = 1

# Tamaño del arreglo de muestras
nPuntos = deltaT*tasa_muestreo

puntos_tiempo = np.linspace(0, deltaT, nPuntos)

frec_1 = 75
magnitud_1 = 25

frec_2 = 250
magnitud_2 = 40

# Señales
señal_1 = magnitud_1*np.sin(2*np.pi*frec_1*puntos_tiempo)
señal_2 = magnitud_2*np.sin(2*np.pi*frec_2*puntos_tiempo)


# Ruido para la señal
ruido = np.random.normal(0, 13, nPuntos)

señal_pura = señal_1 + señal_2
señal_ruidosa = señal_1 + señal_2 + ruido

fig, (ax1, ax2) = plt.subplots(1, 2, dpi=120, sharey= True)
ax1.plot(puntos_tiempo[0:50], señal_pura[0:50])
ax1.set_title('Señal original')
ax1.set_xlabel('Tiempo')
ax1.set_ylabel('Amplitud')

ax2.plot(puntos_tiempo[1:50], señal_ruidosa[1:50])
ax2.set_title('Señal ruidosa')
ax2.set_xlabel('Tiempo')

plt.show()

#Fin código de referencia para señal ruidosa



#Incio transformada de Fourier

frecuencias = np.linspace(0,512,512)

tranformada = fft(señal_ruidosa)
amplitudes = (1/len(frecuencias))*np.abs(tranformada)

#Se grafica la frecuencia vs la amplitud

fig, ax = plt.subplots(dpi=120)
ax.plot(frecuencias, amplitudes[0:len(frecuencias)])
ax.set_title('Señal en el dominio de la frecuencia')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud')
ax.set_xticks(np.arange(0,501,50))
plt.show()

#Fin transformada de Fourier


#Inicio de limpieza de señal

umbral = 5

def Filtrar_Señal(amplitudes, umbral):
    """
        Elimina el ruido eliminado de las amplitudes menores a un umbral previamente definido.
    """
    for i in range (0, len(amplitudes)):
        if amplitudes[i] < umbral:
            amplitudes[i] = 0
    return amplitudes

amplitudes_limpias = Filtrar_Señal(amplitudes , umbral)

#Se grafica las frecuencias filtradas por el umbral vs la amplitud

fig, ax = plt.subplots(dpi=120)
ax.plot(frecuencias, amplitudes_limpias[0:len(frecuencias)])
ax.set_title('Señal en el dominio de la frecuencia Corregida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud Corregida')
ax.set_xticks(np.arange(0,501,50))
plt.show()

def limpiar_señal (amplitud,transformada,umbral):
    """
        Elimina los elementos de la transformada de fourier que producen el ruido.
    """
    for i in range (0, len(amplitudes)):
        if amplitud[i] < umbral:
            transformada[i] = 0
    return transformada

señal_filtrada = limpiar_señal(amplitudes,tranformada,umbral)

tranformada_inversa = ifft(señal_filtrada)
mod_sl = tranformada_inversa.real

#Grafica la señal original sin ruido y la señal obtenida por la tranformada inversa eliminando el ruido

fig, (ax1, ax2) = plt.subplots(1, 2, dpi=120, sharey= True)
ax1.plot(puntos_tiempo[1:50], mod_sl[1:50])
ax1.set(xlabel='Tiempo', ylabel='Amplitud',
       title='Transformada Inversa')
ax2.plot(puntos_tiempo[0:50], señal_pura[0:50])
ax2.set_title('Señal original')
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Amplitud')
plt.show()




