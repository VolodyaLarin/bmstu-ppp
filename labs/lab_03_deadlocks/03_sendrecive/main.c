#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE 1024

int main(int argc, char **argv)
{
    int myrank, nprocs, len;
    char name[MPI_MAX_PROCESSOR_NAME];
    int *buf, *bufI;
    MPI_Status st;

    buf = (int *)malloc(sizeof(int) * (SIZE * 1024 + 100));
    bufI = (int *)malloc(sizeof(int) * (SIZE * 1024 + 100));

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Get_processor_name(name, &len);

    printf("Hello from processor %s[%d] %d of %d\n", name, len, myrank, nprocs);

    if (myrank % 2 == 0)
    {
        if (myrank < nprocs - 1)
        {
            int i, cl, sz = SIZE;
            double time;

            for (i = 0; i < SIZE * 1024; i++)
                buf[i] = i + 10;

            for (cl = 0; cl < 11; cl++)
            {
                time = MPI_Wtime();
                for (i = 0; i < 100; i++)
                {
                    MPI_Sendrecv( buf, sz, MPI_INT, myrank + 1, 10, 
                                  bufI, sz, MPI_INT, myrank + 1, 20, 
                                  MPI_COMM_WORLD, &st);
                    void *tmp = buf;
                    buf = bufI;
                    bufI = tmp;
                }
                time = MPI_Wtime() - time;
                printf("[%d] Time = %lf  Data=%9.0f KByte\n",
                       myrank,
                       time,
                       sz * sizeof(int) * 200.0 / 1024);
                printf("[%d]  Bandwith[%d] = %lf MByte/sek\n",
                       myrank,
                       cl,
                       sz * sizeof(int) * 200 / (time * 1024 * 1024));
                sz *= 2;
            }
        }
        else
            printf("[%d] Idle\n", myrank);
    }
    else
    {
        int i, cl, sz = SIZE;

        for(i = 0; i< SIZE * 1024; i++)
		    buf[i] = i + 100;

        for (cl = 0; cl < 11; cl++)
        {
            for (i = 0; i < 100; i++)
            {
                MPI_Send(buf, sz, MPI_INT, myrank - 1, 20, MPI_COMM_WORLD);
                MPI_Recv(bufI, sz, MPI_INT, myrank - 1, 10, MPI_COMM_WORLD, &st);

                void *tmp = buf;
                buf = bufI;
                bufI = tmp;

                if ((i < 10 || i >= 90) && myrank == 1) {
                    printf("BUF : %d %d %d %d ...\n", buf[0], buf[1], buf[2], buf[3]);
                    printf("BUFI: %d %d %d %d ...\n", bufI[0], bufI[1], bufI[2], bufI[3]);
                }
            }
            sz *= 2;

            
        }
    }
    MPI_Finalize();
    printf("--------------\n");

    return 0;
}
