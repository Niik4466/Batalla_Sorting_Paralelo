#!/bin/bash

# Validar que se pase el argumento REP
if [ "$#" -ne 1 ]; then
    echo "Uso: $0 REP"
    exit 1
fi

# Cantidad de repeticiones
REP=$1

# Eliminamos los .csv anteriores
rm "MPDMSort_Time.csv" "ParallelMergeSort_Time.csv" "STDSort_Time.csv" 2> /dev/null
# Creamos las carpetas necesarias
mkdir -p "CSV/for_A_and_F"
mkdir -p "CSV/for_B_and_C"
mkdir -p "CSV/for_D_and_E"

#### Empezamos la experimentacion
### Punto a y f

# Definir el array `ns` con las cantidades de elementos
ns=($((2**12)) $((2**14)) $((2**16)) $((2**18)) $((2**20)) $((2**22)) $((2**24)))

echo "Experimentacion para los puntos a y f"

## Experimentacion en CPU
echo "Experimentando en CPU...."
for n in "${ns[@]}"; do
    for ((i = 0; i < REP; ++i)); do
        ./prog "$n" "0" "8" >> /dev/null
    done
done

## Experimentación en GPU
echo "Experimentando en GPU...."
for n in "${ns[@]}"; do
    for ((i = 0; i < REP; ++i)); do
        ./prog "$n" "1" "0" >> /dev/null
    done
done

## Experimentacion con la STL
echo "Experimentando con STL sort...."
for n in "${ns[@]}"; do
    for ((i = 0; i < REP; ++i)); do
        ./prog "$n" "2" "0" >> /dev/null
    done
done

echo "Done!. Copiando Resultados..."
## Copiamos los resultados para ser analizados posteriormente
cp "MPDMSort_Time.csv" "CSV/for_A_and_F/MPDMSort_Time.csv"
cp "ParallelMergeSort_Time.csv" "CSV/for_A_and_F/ParallelMergeSort_Time.csv"
cp "STDSort_Time.csv" "CSV/for_A_and_F/STDSort_Time.csv"
rm "MPDMSort_Time.csv" "ParallelMergeSort_Time.csv" "STDSort_Time.csv" 2> /dev/null
echo "Done!"

### Punto b y c

echo "Experimentacion para los puntos b y c"

# Definir el número de hilos en un array
nts=(1 2 4 8)
# Definir un n fijo para experimentar
n=$((2**22))

echo "Experimentando con CPU..."
for nt in "${nts[@]}"; do
    for ((i = 0; i < REP; ++i)); do
        ./prog "$n" "0" "$nt" >> /dev/null
    done
done

echo "Done!. Copiando Resultados..."
cp "MPDMSort_Time.csv" "CSV/for_B_and_C/MPDMSort_Time.csv"
rm "MPDMSort_Time.csv"
echo "Done!"

### Punto d y e

echo "Experimentacion para los puntos d y e"

# Obtener el número de SMs de la tarjeta gráfica
nvcc -o get_sm_count .get_sm_count.cu
SM_COUNT=$(./get_sm_count cudaDevAttrMultiProcessorCount)
rm get_sm_count

# Definir las configuraciones de bloques basadas en los SMs
CANTBLOCKS=(1)
for ((i = 1; i <= 5; ++i)); do
    CANTBLOCKS+=($((i * SM_COUNT))) # Rango de 1 a 5 veces el número de SMs
done

# Definimos un n Fijo
n=$((2**10))

echo "Experimentando en GPU...."
for cantblock in "${CANTBLOCKS[@]}"; do
    for ((i = 0; i < REP; ++i)); do
        blocksize=$(((n+cantblock-1)/cantblock))
        BLOCKSIZE="$blocksize" ./prog "$n" "1" "0" >> /dev/null
    done
done

echo "Done!. Copiando Resultados..."
cp "ParallelMergeSort_Time.csv" "CSV/for_D_and_E/ParallelMergeSort_Time.csv"
rm "ParallelMergeSort_Time.csv"
echo "Done!"

### Crear los graficos
mkdir -p "graphs"
echo "Creando graficos..."
python "makeGraphs.py"

echo "Experimentacion Terminada"