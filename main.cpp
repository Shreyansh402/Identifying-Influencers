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

// #define N 6
// #define E 10
// #define FILENAME "sample.dat"

// Directed Graph
class Graph
{
public:
    unordered_map<string, vector<string>> adj; // adjacency list
    unordered_map<string, int> degree;         // indegree of each node
    unordered_map<string, bool> visited;       // nodes assigned to this process and is it visited or not

    Graph()
    {
        adj.reserve(N);
        degree.reserve(N);
        visited.reserve(N);
    };

    void addEdge(string u, string v)
    {
        adj[u].push_back(v);
        // if (degree.find(v) == degree.end())
        // {
        //     degree[v] = 0;
        // }
        // if (degree.find(u) == degree.end())
        // {
        //     degree[u] = 0;
        // }
        degree[v]++;
        degree[u]++;
    };
};

// Read input from file according to the rank of the process using MPI
Graph *readFile(int rank, int size, Graph *local_graph)
{
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);
    MPI_Offset segment = file_size / size;
    MPI_Offset start = rank == 0 ? 0 : rank * segment + file_size % size;
    MPI_Offset end = (rank + 1) * segment + file_size % size;

    // // Adjust start and end to read complete lines
    // char c;
    // if (rank > 0)
    // {
    //     MPI_File_read_at(fh, start, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
    //     while (c != '\n')
    //     {
    //         start++;
    //         MPI_File_read_at(fh, start, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
    //     }
    // }
    // if (rank < size - 1)
    // {
    //     MPI_File_read_at(fh, end, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
    //     while (c != '\n')
    //     {
    //         end++;
    //         MPI_File_read_at(fh, end, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
    //     }
    // }

    char *buf = new char[end - start];
    MPI_File_read_at(fh, start, buf, end - start, MPI_CHAR, MPI_STATUS_IGNORE);

    // Send raw data to root process
    if (rank == 0)
    {
        // Root process collects all data
        char *all_data = new char[file_size + 1];
        memcpy(all_data, buf, end - start);
        int offset = end - start;

        for (int i = 1; i < size; ++i)
        {
            MPI_Recv(all_data + offset, segment, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += segment;
        }

        delete[] buf;
        MPI_File_close(&fh);
        all_data[file_size] = '\0';

        // Process the collected data to create the graph
        Graph *graph = new Graph();
        vector<vector<char>> send_buf(size);
        vector<vector<char>> local_nodes(size);
        unordered_map<string, int> node_to_rank;
        node_to_rank.reserve(N);
        vector<vector<int>> send_buf_size(size);
        int next_rank = 0;
        istringstream iss(all_data);
        string line;
        while (getline(iss, line))
        {
            istringstream lineStream(line);
            string u, v;
            if (lineStream >> u >> v)
            {
                graph->addEdge(u, v);
                if (node_to_rank.find(v) == node_to_rank.end())
                {
                    node_to_rank[v] = next_rank;
                    if (next_rank == 0)
                    {
                        local_graph->visited[v] = false;
                    }
                    else
                    {
                        // add v to the local nodes of the process
                        local_nodes[next_rank].insert(local_nodes[next_rank].end(), v.begin(), v.end());
                        local_nodes[next_rank].push_back('\n');
                    }
                    next_rank = (next_rank + 1) % size;
                }
                if (node_to_rank.find(u) == node_to_rank.end())
                {
                    node_to_rank[u] = next_rank;
                    if (next_rank == 0)
                    {
                        local_graph->visited[u] = false;
                    }
                    else
                    {
                        // add u to the local nodes of the process
                        local_nodes[next_rank].insert(local_nodes[next_rank].end(), u.begin(), u.end());
                        local_nodes[next_rank].push_back('\n');
                    }
                    next_rank = (next_rank + 1) % size;
                }
                // add raw line data to the process responsible for the node
                if (node_to_rank[v] == 0)
                {
                    local_graph->addEdge(u, v);
                }
                else
                {
                    send_buf[node_to_rank[v]].insert(send_buf[node_to_rank[v]].end(), line.begin(), line.end());
                    send_buf[node_to_rank[v]].push_back('\n');
                }
                if (node_to_rank[u] != node_to_rank[v])
                {
                    if (node_to_rank[u] == 0)
                    {
                        local_graph->addEdge(u, v);
                    }
                    else
                    {
                        send_buf[node_to_rank[u]].insert(send_buf[node_to_rank[u]].end(), line.begin(), line.end());
                        send_buf[node_to_rank[u]].push_back('\n');
                    }
                }
            }
        }

        // Send data to the respective processes
        for (int i = 1; i < size; ++i)
        {
            pair<int, int> send_size = {local_nodes[i].size(), send_buf[i].size()};
            MPI_Send(&send_size, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(local_nodes[i].data(), local_nodes[i].size(), MPI_CHAR, i, 0, MPI_COMM_WORLD);

            MPI_Send(send_buf[i].data(), send_buf[i].size(), MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }

        delete[] all_data;

        return graph;
    }
    // Non-root processes send their data to the root process
    MPI_Send(buf, end - start, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

    delete[] buf;
    MPI_File_close(&fh); // gives error if not done correctly

    // Non-root processes receive their data from the root process
    pair<int, int> recv_size;

    MPI_Recv(&recv_size, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<char> recv_buffer_nodes(recv_size.first);
    MPI_Recv(recv_buffer_nodes.data(), recv_size.first, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // MPI_Status status;
    // MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    // MPI_Get_count(&status, MPI_CHAR, &recv_size);

    vector<char> recv_buffer(recv_size.second);
    MPI_Recv(recv_buffer.data(), recv_size.second, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // change visited to false for all nodes
    istringstream recv_nodes_iss(string(recv_buffer_nodes.begin(), recv_buffer_nodes.end()));
    string node;
    while (getline(recv_nodes_iss, node))
    {
        local_graph->visited[node] = false;
    }

    // Create local graph
    istringstream recv_iss(string(recv_buffer.begin(), recv_buffer.end()));
    string line;
    while (getline(recv_iss, line))
    {
        istringstream lineStream(line);
        string u, v;
        if (lineStream >> u >> v)
        {
            local_graph->addEdge(u, v);
        }
    }

    return nullptr;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    // start time
    double start_time = MPI_Wtime();
    int rank, size;
    Graph *local_graph = new Graph();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Graph *complete_graph = readFile(rank, size, local_graph);

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