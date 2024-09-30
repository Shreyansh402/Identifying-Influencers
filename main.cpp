// Assignment -2 COL730
// Name: Shreyansh Jain
// Entry No: 2021MT10230

#include <iostream>
#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <queue>
#include <stack>
#include <climits>
#include <cstring>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <sstream>
#include <fstream>

using namespace std;

#define L 23

#define N 475
#define E 13289
#define FILENAME "twitter.dat"

// #define N 107614
// #define E 13673453
// #define FILENAME "gplus_combined.dat"

// #define N 7198
// #define E 10000
// #define FILENAME "gplus_short.dat"

// #define N 6
// #define E 10
// #define FILENAME "sample.dat"

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

struct NodeScore
{
    char name[L];
    double score;
};

// Custom MPI datatype for NodeScore
MPI_Datatype MPI_NODESCORE;

// Custom MPI operator for finding top k nodes
void findTopK(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype)
{
    NodeScore *in = (NodeScore *)invec;
    NodeScore *inout = (NodeScore *)inoutvec;
    NodeScore *result = new NodeScore[*len];
    int i = 0, j = 0, k = 0;

    // Merge the two sorted arrays
    while (k < *len)
    {
        if (i == *len)
        {
            result[k++] = inout[j++];
        }
        else if (j == *len)
        {
            result[k++] = in[i++];
        }
        else if (in[i].score > inout[j].score ||
                 (in[i].score == inout[j].score && strcmp(in[i].name, inout[j].name) < 0))
        {
            result[k++] = in[i++];
        }
        else
        {
            result[k++] = inout[j++];
        }
    }

    // Copy the result back to inoutvec
    copy(result, result + *len, inout);
    delete[] result;
}

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
    MPI_File_write_at_all(fh, offsets[rank], data_to_write.c_str(), data_size, MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

// Write the Betweenness Centrality of each node
void writeBetweennessCentrality(Graph *graph, int rank, int size)
{
    // Collect data to write
    stringstream ss;
    for (auto &node : graph->visited)
    {
        ss << node.first << " " << graph->centrality[node.first].second << "\n";
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
    MPI_File_open(MPI_COMM_WORLD, "betweenness_centrality.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at_all(fh, offsets[rank], data_to_write.c_str(), data_size, MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

// Normalize the Centrality of each node
void normalizeCentrality(Graph *local_graph)
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

    // Find the maximum betweenness centrality
    double max_bc = 0;
    for (auto &node : local_graph->visited)
    {
        max_bc = max(max_bc, local_graph->centrality[node.first].second);
    }

    // Find the global max betweenness centrality across all processes
    double global_max_bc;
    MPI_Allreduce(&max_bc, &global_max_bc, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Find the minimum betweenness centrality
    double min_bc = 1;
    for (auto &node : local_graph->visited)
    {
        min_bc = min(min_bc, local_graph->centrality[node.first].second);
    }

    // Find the global min betweenness centrality across all processes
    double global_min_bc;
    MPI_Allreduce(&min_bc, &global_min_bc, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    global_max_bc = global_max_bc - global_min_bc;

    // Normalize the betweenness centrality and add it to the degree centrality to get the final centrality score
    for (auto &node : local_graph->visited)
    {
        local_graph->centrality[node.first].second = ((local_graph->centrality[node.first].second - global_min_bc) / global_max_bc) * 0.4 + local_graph->centrality[node.first].first;
    }
}

// Read input from file parallely and create the graph at root process
Graph *readFile(int rank, int size)
{
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);
    MPI_Offset segment = file_size / size;
    MPI_Offset start = rank == 0 ? 0 : rank * segment + file_size % size;
    MPI_Offset end = (rank + 1) * segment + file_size % size;

    char *buf = new char[end - start];
    MPI_File_read_at_all(fh, start, buf, end - start, MPI_CHAR, MPI_STATUS_IGNORE);

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

        // Broadcast all_data
        MPI_Bcast(all_data, file_size, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Process the collected data to create the graph
        Graph *local_graph = new Graph();
        vector<vector<char>> local_nodes(size);
        unordered_map<string, int> node_to_rank;
        node_to_rank.reserve(N);
        int next_rank = 0;
        istringstream iss(all_data);
        string line;
        while (getline(iss, line))
        {
            istringstream lineStream(line);
            string u, v;
            if (lineStream >> u >> v)
            {
                local_graph->addEdge(u, v);
                // graph->addEdge(u, v);
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
            }
        }

        // Send data to the respective processes
        for (int i = 1; i < size; ++i)
        {
            int send_size = local_nodes[i].size();
            MPI_Send(&send_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(local_nodes[i].data(), local_nodes[i].size(), MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }

        delete[] all_data;

        // return graph;
        return local_graph;
    }
    // Non-root processes send their data to the root process
    MPI_Send(buf, end - start, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

    delete[] buf;
    MPI_File_close(&fh); // gives error if not done correctly

    // Broadcast all_data
    char *all_data = new char[file_size + 1];
    MPI_Bcast(all_data, file_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Process the collected data to create the graph
    Graph *local_graph = new Graph();
    istringstream iss(all_data);
    string line;
    while (getline(iss, line))
    {
        istringstream lineStream(line);
        string u, v;
        if (lineStream >> u >> v)
        {
            local_graph->addEdge(u, v);
        }
    }

    delete[] all_data;

    int recv_size;
    MPI_Recv(&recv_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<char> recv_buffer_nodes(recv_size);
    MPI_Recv(recv_buffer_nodes.data(), recv_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // change visited to false for all nodes
    istringstream recv_nodes_iss(string(recv_buffer_nodes.begin(), recv_buffer_nodes.end()));
    string node;
    while (getline(recv_nodes_iss, node))
    {
        local_graph->visited[node] = false;
    }
    return local_graph;
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

// Breadth First Search to find the shortest path from source to all other nodes
void BFS(Graph *graph, int rank, int size)
{
    for (auto &start : graph->visited)
    {
        string source = start.first;
        unordered_map<string, vector<string>> pred;
        unordered_map<string, int> dist;
        unordered_map<string, int> sigma;
        queue<string> Q;
        stack<string> S;

        for (const auto &node : graph->degree)
        {
            dist[node.first] = INT_MAX;
            sigma[node.first] = 0;
        }

        dist[source] = 0;
        sigma[source] = 1;
        Q.push(source);

        while (!Q.empty())
        {
            string v = Q.front();
            Q.pop();
            S.push(v);

            for (const string &w : graph->adj[v])
            {
                if (dist[w] == INT_MAX)
                {
                    dist[w] = dist[v] + 1;
                    Q.push(w);
                }
                if (dist[w] == dist[v] + 1)
                {
                    sigma[w] += sigma[v];
                    pred[w].push_back(v);
                }
            }
        }

        unordered_map<string, double> delta;
        for (const auto &node : graph->adj)
        {
            delta[node.first] = 0;
        }

        while (!S.empty())
        {
            string w = S.top();
            S.pop();
            for (const string &v : pred[w])
            {
                delta[v] += (sigma[v] * 1.0 / sigma[w]) * (1 + delta[w]);
            }
            if (w != source)
            {
                graph->centrality[w].second += delta[w];
            }
        }
    }

    // Send the order of nodes from root process to all other processes
    if (rank == 0)
    {
        stringstream ss;
        vector<string> all_nodes;
        for (auto &node : graph->centrality)
        {
            ss << node.first << "\n";
            all_nodes.push_back(node.first);
        }
        string data_send = ss.str();
        int data_size = data_send.size();
        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        char *data_to_send_char = new char[data_size];
        strcpy(data_to_send_char, data_send.c_str());
        MPI_Bcast(data_to_send_char, data_size, MPI_CHAR, 0, MPI_COMM_WORLD);

        vector<double> all_centrality(all_nodes.size());
        for (size_t i = 0; i < all_nodes.size(); ++i)
        {
            all_centrality[i] = graph->centrality[all_nodes[i]].second;
        }

        // Reduce_all to find the sum of all centrality scores
        vector<double> global_centrality(all_nodes.size());
        MPI_Allreduce(all_centrality.data(), global_centrality.data(), all_nodes.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Update the centrality scores
        for (size_t i = 0; i < all_nodes.size(); ++i)
        {
            if (graph->visited.find(all_nodes[i]) != graph->visited.end())
            {
                graph->centrality[all_nodes[i]].second = global_centrality[i] / (N - 1);
                graph->centrality[all_nodes[i]].second = graph->centrality[all_nodes[i]].second / (N - 2);
            }
        }
    }
    else
    {
        int data_size;
        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        char *data_to_recv_char = new char[data_size];
        MPI_Bcast(data_to_recv_char, data_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        istringstream iss(string(data_to_recv_char, data_to_recv_char + data_size));
        string node;
        vector<string> all_nodes;
        while (getline(iss, node))
        {
            all_nodes.push_back(node);
        }
        vector<double> all_centrality(all_nodes.size());
        for (size_t i = 0; i < all_nodes.size(); ++i)
        {
            all_centrality[i] = graph->centrality[all_nodes[i]].second;
        }

        // Reduce_all to find the sum of all centrality scores
        vector<double> global_centrality(all_nodes.size());
        MPI_Allreduce(all_centrality.data(), global_centrality.data(), all_nodes.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Update the centrality scores
        for (size_t i = 0; i < all_nodes.size(); ++i)
        {
            if (graph->visited.find(all_nodes[i]) != graph->visited.end())
            {
                graph->centrality[all_nodes[i]].second = global_centrality[i] / (N - 1);
                graph->centrality[all_nodes[i]].second = graph->centrality[all_nodes[i]].second / (N - 2);
            }
        }
    }
}

// Find the top k nodes with the highest centrality score
void topKNodes(Graph *graph, int rank, int size, int k)
{
    // Create MPI datatype for NodeScore
    MPI_Datatype types[2] = {MPI_CHAR, MPI_DOUBLE};
    int blocklengths[2] = {L, 1};
    MPI_Aint offsets[2];
    offsets[0] = offsetof(NodeScore, name);
    offsets[1] = offsetof(NodeScore, score);
    MPI_Type_create_struct(2, blocklengths, offsets, types, &MPI_NODESCORE);
    MPI_Type_commit(&MPI_NODESCORE);

    // Prepare local top k nodes
    vector<NodeScore> localTopK;
    for (const auto &pair : graph->visited)
    {
        NodeScore ns;
        strncpy(ns.name, pair.first.c_str(), L - 1);
        ns.name[L - 1] = '\0';                           // Ensure null-termination
        ns.score = graph->centrality[pair.first].second; // Assuming .second is the normality score
        localTopK.push_back(ns);
    }

    // Sort local nodes and keep only top k
    sort(localTopK.begin(), localTopK.end(),
         [](const NodeScore &a, const NodeScore &b)
         {
             if (a.score == b.score)
             {
                 return strcmp(a.name, b.name) < 0; // if scores are equal, sort by name
             }
             return a.score > b.score;
         });
    if (localTopK.size() > (size_t)k)
    {
        localTopK.resize(k);
    }

    // Pad localTopK to exactly k elements
    localTopK.resize(k, NodeScore{"", -INFINITY});

    // Create custom MPI operator
    MPI_Op op;
    MPI_Op_create((MPI_User_function *)findTopK, 1, &op);

    // Reduce to find global top k
    vector<NodeScore> globalTopK(k);
    MPI_Reduce(localTopK.data(), globalTopK.data(), k, MPI_NODESCORE, op, 0, MPI_COMM_WORLD);

    // Clean up
    MPI_Type_free(&MPI_NODESCORE);
    MPI_Op_free(&op);

    if (rank == 0)
    {
        // write the top k nodes to top_k_nodes.txt after opening the file or creating it if it doesn't exist
        ofstream top_k_nodes("top_k_nodes.txt");
        for (const auto &node : globalTopK)
        {
            top_k_nodes << node.name << endl;
        }
        top_k_nodes.close();
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the correct number of arguments is provided
    if (argc != 2)
    {
        if (rank == 0)
        {
            cerr << "Usage: " << argv[0] << " <k>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Parse the command-line argument
    int k = atoi(argv[1]);

    // start time
    double start_time = MPI_Wtime();

    // Read the file and create the graph
    Graph *local_graph = readFile(rank, size);

    // write the degree of each node to degree_centrality.txt
    writeDegreeCentrality(local_graph, rank, size);

    // Find betweenness centrality of each node
    BFS(local_graph, rank, size);

    // Write the Betweenness Centrality of each node to betweenness_centrality.txt
    writeBetweennessCentrality(local_graph, rank, size);

    // Normalize cenrality and find the centrality score of each node
    normalizeCentrality(local_graph);

    // Find the top k nodes with the highest centrality score
    topKNodes(local_graph, rank, size, k);

    // end time
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    if (rank == 0)
    {
        cout << "Time taken with " << size << " processors :" << end_time - start_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}