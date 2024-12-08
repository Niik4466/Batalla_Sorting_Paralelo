# Batalla de Sorting Paralelo
proyecto enfocado en realizar una comparacion entre los dos mejores algoritmos de ordenamiento en paralelo para CPU y GPU respectivamente

## Especificacion
Para este proyecto se implemento el algoritmo MPDMSort para CPU y un algoritmo de mergeSort en paralelo para GPU generico

## Compilacion
- El programa se compila con 
```bash
make
```

## Ejecucion
- El programa se ejecuta como:
```bash
./prog <n> <mode> <nt>
```
donde:
- **n**: Cantidad de elementos del arreglo
- **mode**: Modo a ejecutar
    - 0: CPU
    - 1: GPU
    - 2: STL
- **nt**: Cantidad de threads (disponible solo para mode=0)

## Experimentacion
Se puede experimentar con el programa utilizando
```bash
./experimentation.sh <REP>
```
donde **REP** es la cantidad de veces a ejecutar cada experimento

- El programa experimentation.sh ejecutara todos los experimentos necesarios para cumplir con los puntos descritos en la tarea.
- Tambien ejecutara el script python graphs.py que creara todos los graficos necesarios para visualizar los resultados de la experimentacion
