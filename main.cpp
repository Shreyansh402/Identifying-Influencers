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

// #define N 475
// #define E 13289
// #define FILENAME "twitter.dat"

// #define N 107614
// #define E 13673453
// #define FILENAME "gplus_combined.dat"

#define N 6
#define E 10
#define FILENAME "sample.dat"

// Directed Graph
class Graph
{
public:
    unordered_map<string, vector<string>> adj;              // adjacency list
    unordered_map<string, int> degree;                      // degree of each node
    unordered_map<string, pair<double, double>> centrality; //  centrality score of each node
    unordered_map<string, bool> visited;                    // nodes assigned to this process and is it visited or not

    Graph()
    {
        adj.reserve(N);
        degree.reserve(N);
        visited.reserve(N);
        centrality.reserve(N);
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

// Write the Degree Centrality of each node to a file
void writeDegreeCentrality(Graph *local_graph, int rank, int size)
{
    // Collect data to write
    stringstream ss;
    for (auto &node : local_graph->visited)
    {
        ss << node.first << " " << local_graph->degree[node.first] << "\n";
    }
    string data_to_write = ss.str();
    int data_size = data_to_write.size();

    // Gather sizes of data to write from all processes
    vector<int> all_data_sizes(size);
    MPI_Gather(&data_size, 1, MPI_INT, all_data_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate offsets
    vector<int> offsets(size, 0);
    if (rank == 0)
    {
        for (int i = 1; i < size; ++i)
        {
            offsets[i] = offsets[i - 1] + all_data_sizes[i - 1];
        }
    }

    // Broadcast offsets to all processes
    MPI_Bcast(offsets.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    // Write data to file in parallel
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, "degree_centrality.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at(fh, offsets[rank], data_to_write.c_str(), data_size, MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

// Normalize the Degree Centrality of each node
void normalizeDegreeCentrality(Graph *local_graph)
{
    // Find the maximum degree
    int max_degree = 0;
    for (auto &node : local_graph->visited)
    {
        max_degree = max(max_degree, local_graph->degree[node.first]);
    }

    // Find the global max degree across all processes
    int global_max;
    MPI_Allreduce(&max_degree, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Find the minimum degree
    int min_degree = N;
    for (auto &node : local_graph->visited)
    {
        min_degree = min(min_degree, local_graph->degree[node.first]);
    }

    // Find the global min degree across all processes
    int global_min;
    MPI_Allreduce(&min_degree, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    global_max = global_max - global_min;

    for (auto &node : local_graph->visited)
    {
        local_graph->centrality[node.first].first = ((local_graph->degree[node.first] - global_min) / (double)global_max) * 0.6;
    }
}

// Read input from file parallely and create the graph
Graph *readFile(int rank, int size, Graph *local_graph)
{
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);
    MPI_Offset segment = file_size / size;
    MPI_Offset start = rank == 0 ? 0 : rank * segment + file_size % size;
    MPI_Offset end = (rank + 1) * segment + file_size % size;

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

// Dijkstra's Algorithm to find the shortest path from source to all other nodes
void dijkstra(Graph *local_graph, string source, int rank, int size)
{
    unordered_map<string, int> distance;
    unordered_map<string, vector<string>> parent;
    priority_queue<pair<int, string>, vector<pair<int, string>>, greater<pair<int, string>>> pq;

    // update all the nodes connected to min_dist_node
    if (local_graph->visited.find(source) != local_graph->visited.end())
    {
        distance[source] = 0;
        local_graph->visited[source] = true;
    }
    for (auto &node : local_graph->adj[source])
    {
        if (local_graph->visited.find(node) != local_graph->visited.end())
        {
            distance[node] = 1;
            pq.push({1, node});
        }
    }

    // Dijkstra's Algorithm
    while (1)
    {
        // Find the local node with the minimum distance
        pair<int, string> min_dist_node = {INT_MAX, ""};
        while (!pq.empty())
        {
            min_dist_node = pq.top();
            pq.pop();
            if (local_graph->visited[min_dist_node.second])
            {
                min_dist_node = {INT_MAX, ""};
            }
            else
            {
                break;
            }
        }

        // Create a pair to hold the distance and the rank
        pair<int, int> local_min_dist_node = {min_dist_node.first, rank};
        pair<int, int> global_min_dist_node;

        // Perform the reduction to find the global minimum distance node and the rank
        MPI_Allreduce(&local_min_dist_node, &global_min_dist_node, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        // If the global minimum distance node is infinity, then break
        if (global_min_dist_node.first == INT_MAX)
        {
            break;
        }

        // check if our node was the global minimum distance node
        if (global_min_dist_node.second == rank)
        {
            // Broadcast the node name to all processes as char *
            int send_size = min_dist_node.second.size();
            char min_dist_node_name[send_size + 1];
            strcpy(min_dist_node_name, min_dist_node.second.c_str());
            MPI_Bcast(min_dist_node_name, min_dist_node.second.size() + 1, MPI_CHAR, rank, MPI_COMM_WORLD);

            // mark the node as visited
            local_graph->visited[min_dist_node.second] = true;

            // update all the nodes connected to min_dist_node
            int new_dist = min_dist_node.first + 1;
            for (auto &node : local_graph->adj[min_dist_node.second])
            {
                if (local_graph->visited.find(node) != local_graph->visited.end())
                {
                    if (new_dist < distance[node])
                    {
                        distance[node] = new_dist;
                        pq.push({new_dist, node});
                        parent[node].clear();
                        parent[node].push_back(min_dist_node.second);
                    }
                    else if (new_dist == distance[node])
                    {
                        parent[node].push_back(min_dist_node.second);
                    }
                }
            }
        }
        else
        {
            // Recieve the node name from the process with the global minimum distance node
            char rec_buf[22];
            MPI_Bcast(rec_buf, 22, MPI_CHAR, global_min_dist_node.second, MPI_COMM_WORLD);

            // create string from char * with
            string vis_node(rec_buf);

            // add min_dist_node back into priority queue
            pq.push(min_dist_node);

            // update all the nodes connected to vis_node
            int new_dist = global_min_dist_node.first + 1;
            for (auto &node : local_graph->adj[vis_node])
            {
                if (local_graph->visited.find(node) != local_graph->visited.end())
                {
                    if (new_dist < distance[node])
                    {
                        distance[node] = new_dist;
                        pq.push({new_dist, node});
                        parent[node].clear();
                        parent[node].push_back(vis_node);
                    }
                    else if (new_dist == distance[node])
                    {
                        parent[node].push_back(vis_node);
                    }
                }
            }
        }
    }

    // Send the parent map to the root process
    if (rank == 0)
    {
        unordered_map<string, vector<string>> all_parent;
        for (int i = 1; i < size; ++i)
        {
            int parent_size;
            MPI_Recv(&parent_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            char parent_buf[parent_size];
            MPI_Recv(parent_buf, parent_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            istringstream parent_iss(string(parent_buf, parent_buf + parent_size));
            string line;
            while (getline(parent_iss, line))
            {
                istringstream lineStream(line);
                string node;
                vector<string> parents;
                if (lineStream >> node)
                {
                    while (lineStream >> node)
                    {
                        parents.push_back(node);
                    }
                    all_parent[node] = parents;
                }
            }
        }

        // Print the shortest path from source to all other nodes
        for (auto &node : all_parent)
        {
            cout << "Shortest path from " << source << " to " << node.first << " is: ";
            for (auto &parent : node.second)
            {
                cout << parent << " -> ";
            }
            cout << node.first << endl;
        }
    }
    else
    {
        // Send the parent map to the root process
        stringstream ss;
        for (auto &node : parent)
        {
            ss << node.first;
            for (auto &p : node.second)
            {
                ss << " " << p;
            }
            ss << "\n";
        }
        string data_to_send = ss.str();
        int data_size = data_to_send.size();
        MPI_Send(&data_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(data_to_send.c_str(), data_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
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

    // Read the file and create the graph
    Graph *complete_graph = readFile(rank, size, local_graph);

    // write the degree of each node to degree_centrality.txt for root without using function for complete graph
    writeDegreeCentrality(local_graph, rank, size);

    // Normalize Degree cenrality
    normalizeDegreeCentrality(local_graph);

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