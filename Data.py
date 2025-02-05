# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:35:24 2025

@author: User
"""

# len() = es una funcion que devuelve la longitud de un objeto
# mean() = calcula la media de la señal
# std() = calcula la desviacion estandar
# percentile()


import numpy as np #Operaciones matematicas 
import matplotlib.pyplot as plt #Graficas y dibujos
import wfdb  #Para carga de señales fisiológicas y manipularlos 
from scipy.io import loadmat #para archivos .mat
from scipy import stats #Para funciones estadisticas



x=loadmat ('100m.mat')
ecg=x['val']
ecg=np.transpose(ecg) #corrige las filas y columnas de la matriz

señal, datos = wfdb.rdsamp('100')
record = wfdb.rdrecord('100')


n = datos['sig_len']#llamar longitud de verctor datos
c = señal[:,1] #canal que vamos a usar 2
fr = datos ['fs'] #Frecuencia
#t = np.linspace(0,d, len(señal)) #crea un vector tiempo, empieza en 0, termina en d y se separa en puntos uniformes


#funcion de probabilidad
kde = stats.gaussian_kde(c)
d = np.linspace(min(c),max(c),100)
p = kde(d)



tm=1/fr #tiempo de muestreo
t= np.linspace(0,np.size(señal),np.size(señal))*tm
ts= np.linspace(0,np.size(ecg),np.size(ecg))*tm
s=señal.flatten() #para que la señal este unidimencional, para que no salga en forma de matrix

media = np.mean(señal)
mediap = sum(s)/np.size(s)
print("Media Programada: ",mediap)
print("Media Predefinida: ",media)
d_e = np.std(señal) #desviacion estandar
d_ep= np.sqrt(sum((x - mediap) ** 2 for x in s) / len(s))
print("Desviación estándar Programada:", d_ep)
print("Desviación Predefinida:",d_e)
cv = d_e/media
print("Coeficiente de variación:", cv)



plt.hist(señal, bins=100)
plt.xlabel("tiempo [s]")
plt.ylabel("voltaje [mV]")
plt.title("Histograma")
plt.show()

plt.plot(ts,ecg)
plt.xlabel("tiempo [s]")
plt.ylabel("voltaje [mV]")
plt.show()

plt.plot(d,p)
plt.xlabel("tiempo [s]")
plt.ylabel("voltaje [mV]")
plt.title("Funcion probabilidad")
plt.show()









