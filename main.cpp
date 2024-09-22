#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int summ = rank;
    int recbuf;
    MPI_Request request;

    for (int i = 0; i < size; i++)
    {
        MPI_Irecv(&recbuf, 1, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, &request);
        MPI_Ssend(&summ, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        summ += recbuf;
    }
    cout << "Rank: " << rank << " Summ: " << summ << endl;
    MPI_Finalize();
    return 0;
}