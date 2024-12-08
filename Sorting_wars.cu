#include <cuda.h>
#include <iostream>
#include <random>
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <utility> // Para swap y pair
#include <deque>
#include <vector>

using namespace std;
//----------------------------------------------------------------------------------------------------------------------------------------------------------
// Macros for MPDMSort
#define CUTOFF 32
#define blocksize 1024

// Macros for Parallel Radix Sort
#define K 8  // bits per pass
#define BINS (1 << K)
#define BLOCKSIZE 1024

// Macros for the general code
#define SEED 32

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

// Crea un array aleatorio de n elementos enteros positivos dado una semilla
void mkArray(int* Array, const int &n, const int seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
    for (int i = 0; i < n; ++i) Array[i] = dist(gen);
}

// Imprime un arreglo, permite dejar un mensaje
void printArray(int* Array, const int &n, const char* message){
    printf("%s\n", message); fflush(stdout);
    for (int i = 0; i < n; ++i) printf("%4d ", Array[i]);
    printf("\n"); fflush(stdout);
}

//------------------------------------------------------------------ MPDMSort ------------------------------------------------------------------------------------------------------

//Algoritmo 4
vector<deque<pair<int, int>>> InitBlocks(vector<int>& A, int left, int right, int pivot, int thread) {
    int blocks = (right - left) / blocksize;
    int blockremain = (right - left) % blocksize;
    vector<deque<pair<int, int>>> deq(thread); // Vector de deques
    int k = left + 1;

    for (int i = 0; i < blocks; i++) {
        deq[i % thread].push_back({k, k + blocksize - 1});
        k += blocksize;
    }

    if (blockremain > 0) {
        deq[blocks % thread].push_back({k, right});
    }

    return deq;
}

//Algoritmo 5
void DualDeqMerge(vector<int>& A, deque<pair<int, int>>& dl, deque<pair<int, int>>& dg) {
    while (!dl.empty() && !dg.empty()) {
        auto dltemp = dl.back();
        auto dgtemp = dg.back();
        dl.pop_back();
        dg.pop_back();

        int i = dltemp.first;
        int j = dgtemp.second;

        while (i <= dltemp.second && j >= dgtemp.first && i < j) {
            swap(A[i], A[j]);
            i++;
            j--;
        }

        if (j > dltemp.first) {
            dg.push_front({i, dgtemp.second});
        } else if (i < dgtemp.second) {
            dl.push_back({dltemp.first, j});
        }
    }
}

//Particionamiento secuencial
int SeqPartitioning(vector<int>& A, int left, int right, int pivot) {
    swap(A[pivot], A[left]);
    pivot = left;
    int i = left + 1;
    int j = right;

    while (i <= j) {
        while (i <= j && A[i] <= A[pivot]) {
            i++;
        }
        while (i <= j && A[j] > A[pivot]) {
            j--;
        }
        if (i < j) {
            swap(A[i], A[j]);
        }
    }

    swap(A[pivot], A[j]);
    return j;
}

//Implementación de MPDMPAr, algoritmo 2
int MPDMPAr(vector<int>& A, int left, int right, int pivot, int thread) {
    swap(A[pivot], A[left]);
    vector<deque<pair<int, int>>> deqs = InitBlocks(A, left, right, pivot, thread);
    
    deque<pair<int, int>> dl, dg;

    // Paralelismo con OpenMP
    #pragma omp parallel for shared(deqs)
    for (int t = 0; t < thread; t++) {
        while (!deqs[t].empty()) {
            pair<int, int> indices;
            
            #pragma omp critical
            {
                if (!deqs[t].empty()) {
                    indices = deqs[t].front();
                    deqs[t].pop_front();
                }
            }

            int i = indices.first;
            int j = indices.second;

            while (i <= j) {
                while (A[i] <= A[left] && i <= j) {
                    i++;
                }
                if (i <= j) {
                    swap(A[i], A[j]);
                    i++;
                    j--;
                }
            }

            #pragma omp critical
            {
                dl.push_back({i, j});
                dg.push_back({j + 1, indices.second});
            }
        }
    }

    // Fusiona las colas
    DualDeqMerge(A, dl, dg);

    // Particiona secuencialmente y calcula la nueva posición del pivote
    int new_pivot = SeqPartitioning(A, left, right, left);

    return new_pivot;
}

//Implementación del algoritmo de la mediana de 5
int medianOf5(vector<int>& A, int left, int right) {
    int mid = left + (right - left) / 2;
    int q1 = left + (mid - left) / 2;
    int q3 = mid + (right - mid) / 2;

    // Ordenamos los cinco elementos seleccionados
    std::vector<int> temp = {A[left], A[q1], A[mid], A[q3], A[right]};
    sort(temp.begin(), temp.end());
    
    // Colocamos el valor mediano en la posición de la izquierda
    A[left] = temp[2]; // Mediana es el tercer valor en el orden
    return left; // El índice del pivote es ahora en 'left'
}

//Algoritmo 1 - MPDMSort
void MPDMSort(vector<int>& A, int left, int right, int thread) {
    if (right - left < CUTOFF) {
        sort(A.begin() + left, A.begin() + right + 1);
        return;
    }

    int pivot = medianOf5(A, left, right);
    int new_pivot = MPDMPAr(A, left, right, pivot, thread);

    #pragma omp parallel sections
    {
        #pragma omp section
        MPDMSort(A, left, new_pivot - 1, thread);

        #pragma omp section
        MPDMSort(A, new_pivot + 1, right, thread);
    }
}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------- Merge SORT Paralelo -------------------------------------------------------------------------------------------------------

// Kernel para mezclar dos sub-rangos
__global__ void merge_kernel(int* d_data, int* d_temp, size_t n, size_t subrange_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t start1 = tid * 2 * subrange_size;
    size_t start2 = start1 + subrange_size;
    size_t end1 = min(start2, n);
    size_t end2 = min(start2 + subrange_size, n);

    size_t i = start1, j = start2, k = start1;

    // Realiza la mezcla
    while (i < end1 && j < end2) {
        if (d_data[i] <= d_data[j]) {
            d_temp[k++] = d_data[i++];
        } else {
            d_temp[k++] = d_data[j++];
        }
    }
    while (i < end1) d_temp[k++] = d_data[i++];
    while (j < end2) d_temp[k++] = d_data[j++];
}

// Algoritmo MergeSort en paralelo
double parallel_merge_sort(int* h_input, size_t n, size_t blockSize) {
    size_t subrangeSize = 1;
    int* d_data; 
    int* d_temp;

    cudaMalloc(&d_data, sizeof(int) * n);
    cudaMalloc(&d_temp, sizeof(int) * n);
    cudaMemcpy(d_data, h_input, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Comenzamos a calcular el tiempo
    double time = omp_get_wtime();
    // Se realizan múltiples pasos de mezcla
    while (subrangeSize < n) {
        // Número de subrangos
        size_t numSubranges = (n + 2 * subrangeSize - 1) / (2 * subrangeSize);

        // Calculamos dinámicamente el número de bloques
        size_t numBlocks = (numSubranges + blockSize - 1) / blockSize;
        // Ejecutamos el kernel de mezcla
        merge_kernel<<<numBlocks, blockSize>>>(d_data, d_temp, n, subrangeSize);
        cudaDeviceSynchronize();
        // Intercambiamos los punteros
        std::swap(d_data, d_temp);

        // Incrementamos el tamaño del subrango
        subrangeSize *= 2;
    }

    // Sincronizamos los threads y obtenemos el tiempo final
    cudaDeviceSynchronize();
    time = omp_get_wtime() - time;

    // Copiamos los datos de vuelta al host
    cudaMemcpy(h_input, d_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

    // Liberamos memoria en el dispositivo
    cudaFree(d_data);
    cudaFree(d_temp);

    return time;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc != 4){
        printf("Error: Se debe ejecutar como ./prog <n> <modo> <nt>\n"); fflush(stderr);
        exit(EXIT_FAILURE);
    }

    // Obtenemos los argumentos
    const int n =    atoi(argv[1]);
    const int mode = atoi(argv[2]);
    const int nt =   atoi(argv[3]);
    const int blockSize = (getenv("BLOCKSIZE") == NULL) ? BLOCKSIZE : atoi(getenv("BLOCKSIZE"));
    const int cantBlocks = (getenv("CANTBLOCKS") == NULL) ? (n+blockSize-1)/blockSize : atoi(getenv("CANTBLOCKS"));

    // Abrimos el archivo a escribir (EXPERIMENTACION)
    std::ofstream file;
    switch (mode)
    {
    case 0:
        file.open("MPDMSort_Time.csv", std::ios::app);
        break;
    case 1:
        file.open("ParallelMergeSort_Time.csv", std::ios::app);
        break;
    case 2:
        file.open("STDSort_Time.csv",std::ios::app);
        break;
    default:
        printf("Error: modo inválido. Debe ser 0 (CPU), 1 (GPU) o 2 (STL).\n"); fflush(stderr);
        exit(EXIT_FAILURE);
    }
    if (!file.is_open()) {
        printf("Error: no se pudo abrir el archivo CSV.\n"); fflush(stderr);
        exit(EXIT_FAILURE);
    }

    // Definimos variables
    const char* modes[3] = {"CPU", "GPU", "STL"};
    int* h_input = new int[n]; 
    mkArray(h_input, n, SEED); 
    double time = 0.0f;

    if (n <= 256) { printArray(h_input, n, "Input:"); }
    //Se define array para MPDMSort 
    omp_set_num_threads(nt);
    std::vector<int> A(h_input, h_input + n);

    // Realizamos el ordenamiento y guardamos el tiempo que toma
    printf("Ejecutando en modo [%3s]......", modes[mode]);
    switch (mode)
    {
    case 0:
        time = omp_get_wtime();
        MPDMSort(A, 0, n - 1, nt);
        time = omp_get_wtime() - time;
        std::copy(A.begin(), A.end(), h_input);
        break;
    case 1:
        time = parallel_merge_sort(h_input, n, blockSize); 
        break;
    default:
        time = omp_get_wtime();
        std::sort(h_input, h_input + n);
        time = omp_get_wtime() - time;
        break;
    }

    printf("done! elapsed: %.4f seconds\n", time);

    if (n <= 32) { printArray(h_input, n, "Result:"); }
    for (int i = 1; i < n; ++i){
        if (h_input[i-1] > h_input[i]){
            printf("Resultado Erroneo\n"); fflush(stderr);
            exit(EXIT_FAILURE);
        }
    }
    // Guardar resultados en el archivo CSV dependiendo del modo
    switch (mode)
    {
    case 0:
        file << n << "," << nt << "," << time << "\n";
        break;
    case 1:
        file << n << "," << cantBlocks << "," << time << "\n";
        break;
    default:
        file << n << "," << time << "\n";
        break;
    }
    file.close();

    // Eliminamos el array utilizado
    delete[] h_input;
    return 0;
}

//------------------------------
