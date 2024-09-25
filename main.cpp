// Assignment -2 COL730
// Name: Shreyansh Jain
// Entry No: 2021MT10230

#include <iostream>
#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <string>
#include <sstream>

using namespace std;

#define N 475
#define E 13289
#define FILENAME "twitter.dat"

// Directed Graph
class Graph
{
public:
    int n, e;                // n is number of nodes and e is number of edges
    vector<vector<int>> adj; // incoming adjacency list
    vector<int> degree;      // degree of each node
    vector<bool> visited;    // visited array

    Graph(int n, int e)
    {
        this->n = n;
        this->e = e;
        this->adj = vector<vector<int>>(n + 1);
        this->degree = vector<int>(n + 1, 0);
        this->visited = vector<bool>(n + 1, false);
    };

    void addEdge(int u, int v)
    {
        this->adj[v].push_back(u);
        this->degree[u]++;
        this->degree[v]++;
    };
};

// Read input from file according to the rank of the process using MPI
Graph *readFile(int rank, int size)
{
    MPI_File fh;
    Graph *graph;
    graph = new Graph(N, E);
    MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);
    MPI_Offset segment = file_size / size;
    MPI_Offset start = rank * segment;
    MPI_Offset end = rank == size - 1 ? file_size : (rank + 1) * segment;

    // Adjust start and end to read complete lines
    char c;
    if (rank > 0)
    {
        MPI_File_read_at(fh, start, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
        while (c != '\n')
        {
            start++;
            MPI_File_read_at(fh, start, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
        }
    }
    if (rank < size - 1)
    {
        MPI_File_read_at(fh, end, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
        while (c != '\n')
        {
            end++;
            MPI_File_read_at(fh, end, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
        }
    }

    char *buf = new char[end - start + 1];
    MPI_File_read_at(fh, start, buf, end - start, MPI_CHAR, MPI_STATUS_IGNORE);
    buf[end - start] = '\0';

    // Process the read data
    istringstream iss(buf);
    string line;
    vector<pair<string, string>> edges;

    while (getline(iss, line))
    {
        istringstream lineStream(line);
        string u, v;
        if (lineStream >> u >> v)
        {
            edges.emplace_back(u, v);
        }
    }

    delete[] buf;

    for (auto edge : edges)
    {
        int u = stoi(edge.first);
        int v = stoi(edge.second);
        graph->addEdge(u, v);
    }

    MPI_File_close(&fh);
    return graph;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    // start time
    double start_time = MPI_Wtime();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Graph *graph = readFile(rank, size);

    // end time
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    if (rank == 0)
    {
        cout << "Time taken: " << end_time - start_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}