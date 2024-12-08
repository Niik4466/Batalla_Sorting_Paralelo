# Cargamos las librerias necesarias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


### A Tiempo(y) vs n(x) y F Speedup(y) vs n(x)
# Cargamos los archivos CSV en Dataframes de pandas
mpdm_data = pd.read_csv("CSV/for_A_and_F/MPDMSort_Time.csv", header=None, names=["n", "threads", "time"])
merge_data = pd.read_csv("CSV/for_A_and_F/ParallelMergeSort_Time.csv", header=None, names=["n", "blocks", "time"])
std_data = pd.read_csv("CSV/for_A_and_F/STDSort_Time.csv", header=None, names=["n", "time"])

# Filtrar datos para los tamaños de n
n_values = np.unique(mpdm_data["n"])

# Promediar los tiempos por tamaño de n
mpdm_avg_time = [mpdm_data[mpdm_data["n"] == n]["time"].mean() for n in n_values]
merge_avg_time = [merge_data[merge_data["n"] == n]["time"].mean() for n in n_values]
std_avg_time = [std_data[std_data["n"] == n]["time"].mean() for n in n_values]

# Grafico para A
plt.figure(figsize=(10, 6))
plt.plot(n_values, mpdm_avg_time, label="MPDM Sort", marker='o')
plt.plot(n_values, merge_avg_time, label="Parallel Merge Sort", marker='x')
plt.xscale('log', base=2)  # Usar escala logarítmica en x
plt.xlabel("n (Tamano de los datos)")
plt.ylabel("Tiempo (segundos)")
plt.title("Tiempo vs n\npara los algoritmos MPDMSort y ParallelMergeSort")
plt.legend()
plt.grid(True)
plt.savefig('graphs/Graph_A.png')

# Grafico para F

mpdm_speedup = [std / mpdm for std, mpdm in zip(std_avg_time, mpdm_avg_time)]
merge_speedup = [std / merge for std, merge in zip (std_avg_time, merge_avg_time)]

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(n_values, mpdm_speedup, label="MPDM Sort", marker='o')
plt.plot(n_values, merge_speedup, label="Parallel Merge Sort", marker='x')
plt.xscale('log', base=2)  # Escala logarítmica en X
plt.xlabel("n (Tamaño de los datos)")
plt.ylabel("Speedup")
plt.title("Speedup vs n para MPDM y Parallel Merge Sort respecto a la STL")
plt.legend()
plt.grid(True)
plt.savefig('graphs/Graph_F.png')

### [Solo CPU] B Speedup(y) vs num-threads(x) y C Eficiencia paralela(y) vs num-bloques(x)

# Cargar los datos
mpdm_data = pd.read_csv("CSV/for_B_and_C/MPDMSort_Time.csv", header=None, names=["n", "nt", "time"])

# Eliminamos la columna n ya que no se utiliza
del mpdm_data['n']

# Calcular el tiempo promedio por cada número de hilos (nt)
mean_times_mpdm = mpdm_data.groupby("nt")["time"].mean().reset_index()

# Calcular el tiempo promedio para 1 hilo (nt=1)
mpdm_1thread_speed = mean_times_mpdm[mean_times_mpdm["nt"] == 1]["time"].values[0]

# Calcular el speedup para cada número de hilos
mean_times_mpdm["speedup"] = mpdm_1thread_speed / mean_times_mpdm["time"]

# Calcular la eficiencia paralela
mean_times_mpdm["efficiency"] = mean_times_mpdm["speedup"] / mean_times_mpdm["nt"]

# Grafico para B
plt.figure(figsize=(10, 6))
plt.plot(mean_times_mpdm["nt"], mean_times_mpdm["speedup"], label="Speedup", marker='o')
plt.xlabel("Número de hilos (nt)")
plt.ylabel("Speedup")
plt.title("Speedup vs Número de Hilos para MPDMSort [CPU]")
plt.grid(True)
plt.legend()
plt.savefig('graphs/Graph_B')

# Grafico para C
plt.figure(figsize=(10, 6))
plt.plot(mean_times_mpdm["nt"], mean_times_mpdm["efficiency"], label="Eficiencia Paralela", marker='o')
plt.xlabel("Número de hilos (nt)")
plt.ylabel("Eficiencia Paralela")
plt.title("Eficiencia Paralela vs Número de Hilos para MPDM Sort [CPU]")
plt.grid(True)
plt.legend()
plt.savefig('graphs/Graph_C')

### [Solo GPU] D Speedup(y) vs num-bloques(x) y E Eficiencia paralela(y) vs num-bloques(x)

# Cargar los datos
merge_data = pd.read_csv("CSV/for_D_and_E/ParallelMergeSort_Time.csv", header=None, names=["n", "blocks", "time"])

# Eliminar la columna 'n', ya que no es necesaria
del merge_data['n']

# Calcular el tiempo promedio por cada número de bloques
mean_times = merge_data.groupby("blocks")["time"].mean().reset_index()

# Calcular el tiempo promedio para 1 bloque
merge_1block_speed = mean_times[mean_times["blocks"] == 1]["time"].values[0]

# Unir el tiempo promedio de los bloques con el DataFrame original
merge_data = merge_data.merge(mean_times, on="blocks", suffixes=("", "_mean"))

# Calcular el speedup para cada número de bloques
merge_data["speedup"] = merge_1block_speed / merge_data["time_mean"]

# Calcular la eficiencia paralela
merge_data["efficiency"] = merge_data["speedup"] / merge_data["blocks"]

# Grafico D
plt.figure(figsize=(10, 6))
plt.plot(mean_times["blocks"], mean_times["time"] / merge_1block_speed, label="Speedup", marker='x')
plt.xlabel("Número de bloques")
plt.ylabel("Speedup")
plt.title("Speedup vs Número de Bloques para Parallel Merge Sort [GPU]")
plt.grid(True)
plt.legend()
plt.savefig('graphs/Graph_D')

# Grafico E
plt.figure(figsize=(10, 6))
plt.plot(mean_times["blocks"], (mean_times["time"] / merge_1block_speed) / mean_times["blocks"], label="Eficiencia Paralela", marker='x')
plt.xlabel("Número de bloques")
plt.ylabel("Eficiencia Paralela")
plt.title("Eficiencia Paralela vs Número de Bloques para Parallel Merge Sort [GPU]")
plt.grid(True)
plt.legend()
plt.savefig('graphs/Graph_E')