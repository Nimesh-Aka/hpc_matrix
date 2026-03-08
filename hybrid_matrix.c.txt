#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

int main(int argc, char *argv[])
{
    int rank, size;
    int rows_per_process;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    rows_per_process = N / size;

    double *A = NULL;
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = NULL;

    if(rank == 0)
    {
        A = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));

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

    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *local_A = (double*)malloc(rows_per_process * N * sizeof(double));
    double *local_C = (double*)malloc(rows_per_process * N * sizeof(double));

    MPI_Scatter(A, rows_per_process*N, MPI_DOUBLE,
                local_A, rows_per_process*N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double start = MPI_Wtime();

    #pragma omp parallel for
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

    MPI_Gather(local_C, rows_per_process*N, MPI_DOUBLE,
               C, rows_per_process*N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if(rank == 0)
    {
        printf("Hybrid MPI+OpenMP Execution Time: %f seconds\n", end - start);

        FILE *fout = fopen("hybrid_output.txt", "w");
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

    free(B);
    free(local_A);
    free(local_C);
    if(rank == 0)
    {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
}