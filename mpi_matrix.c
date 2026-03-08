#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

double A[N*N];
double B[N*N];
double C[N*N];

int main(int argc, char *argv[])
{
    int rank, size;
    int rows_per_process;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    rows_per_process = N / size;
    double start, end;

    // Initialize matrices on rank 0
    if(rank == 0)
    {
        srand(1);
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                A[i*N + j] = rand() % 10;
                B[i*N + j] = rand() % 10;
            }
        }
    }

    // Broadcast matrix B
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate local matrices
    double *local_A = (double*)malloc(rows_per_process * N * sizeof(double));
    double *local_C = (double*)malloc(rows_per_process * N * sizeof(double));

    // Scatter rows of A
    MPI_Scatter(A, rows_per_process*N, MPI_DOUBLE,
                local_A, rows_per_process*N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    start = MPI_Wtime();

    // Local multiplication
    for(int i = 0; i < rows_per_process; i++)
    {
        for(int j = 0; j < N; j++)
        {
            local_C[i*N + j] = 0;
            for(int k = 0; k < N; k++)
            {
                local_C[i*N + j] += local_A[i*N + k] * B[k*N + j];
            }
        }
    }

    // Gather results
    MPI_Gather(local_C, rows_per_process*N, MPI_DOUBLE,
               C, rows_per_process*N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    end = MPI_Wtime();

    if(rank == 0)
    {
        printf("MPI Execution Time: %f seconds\n", end - start);

        FILE *fout = fopen("mpi_output.txt", "w");
        if(fout == NULL) { perror("File open failed"); return 1; }

        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                fprintf(fout, "%f ", C[i*N + j]);
            }
            fprintf(fout, "\n");
        }
        fclose(fout);
    }

    free(local_A);
    free(local_C);
    MPI_Finalize();
    return 0;
}