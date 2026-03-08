#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1024

double A[N][N];
double B[N][N];
double C[N][N];

int main()
{
    int i, j, k;
    double start, end;

    srand(1); // fixed seed

    // Initialize matrices
    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
            C[i][j] = 0;
        }
    }

    start = omp_get_wtime();

    #pragma omp parallel for private(j,k)
    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            for(k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    end = omp_get_wtime();
    printf("OpenMP Execution Time: %f seconds\n", end - start);

    // Write output to file
    FILE *fout = fopen("openmp_output.txt", "w");
    if(fout == NULL) { perror("File open failed"); return 1; }

    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            fprintf(fout, "%f ", C[i][j]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);

    return 0;
}