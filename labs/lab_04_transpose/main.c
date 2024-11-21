#include <stdio.h>
#include <mpi.h>


int main(int argc, char **argv)
{
    int myrank, nprocs, nlen;
    char name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Get_processor_name(name, &nlen);

    printf("Hello from host %s[%d] %d of %d\n", name, nlen, myrank, nprocs);

    if (myrank % 2 == 0)
    {
        if (myrank + 1 < nprocs)
        {
            int a[10][10];
            int *b = a[0];

            for (int i = 0; i < 100; i++)
                *(b++) = i;

            MPI_Send(a, 100, MPI_INT, myrank + 1, 20, MPI_COMM_WORLD);
        }
    }
    else
    {
        int a[10][10];

        MPI_Status st;
        MPI_Datatype col, col1;
        MPI_Aint adr[2];
        MPI_Datatype tp[2];

        int len[2];

        MPI_Type_vector(10, 1, 10, MPI_INT, &col);

        adr[0] = 0;
        adr[1] = sizeof(int);

        tp[0] = col;
        tp[1] = MPI_UB;

        len[0] = 1;
        len[1] = 1;

        MPI_Type_struct(2, len, adr, tp, &col1);
        MPI_Type_commit(&col1);
        MPI_Type_free(&col);

        printf("Receive %s[%d] %d of %d\n", name, nlen, myrank, nprocs);
        
        MPI_Recv(a, 10, col1, myrank - 1, 20, MPI_COMM_WORLD, &st);
        MPI_Type_free(&col1);

        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
                printf("%6d", a[i][j]);
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
