all:
	nvcc -Xcompiler -fopenmp -O3 -o prog Sorting_wars.cu
clean:
	rm prog
