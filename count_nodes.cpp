#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <sstream>

int countDistinctNodes(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return -1;
    }

    std::unordered_set<std::string> distinctNodes;
    std::string line;

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string source, target;

        if (!(iss >> source >> target))
        {
            std::cerr << "Error reading line: " << line << std::endl;
            continue;
        }

        distinctNodes.insert(source);
        distinctNodes.insert(target);
    }

    file.close();

    return distinctNodes.size();
}

int main()
{
    std::string filename = "gplus_short.dat";
    int nodeCount = countDistinctNodes(filename);

    if (nodeCount >= 0)
    {
        std::cout << "Number of distinct nodes: " << nodeCount << std::endl;
    }

    return 0;
}