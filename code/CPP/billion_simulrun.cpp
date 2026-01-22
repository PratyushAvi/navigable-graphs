#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <ctime>
#include <chrono>
#include <set>
#include <sstream>
#include <limits>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

class MemoryMappedFile {
private:
    void* data;
    size_t size;
    int fd;

public:
    MemoryMappedFile(const std::string& filename, size_t file_size) {
        fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        
        size = file_size;
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to mmap file: " + filename);
        }
    }

    ~MemoryMappedFile() {
        if (data != MAP_FAILED) {
            munmap(data, size);
        }
        if (fd != -1) {
            close(fd);
        }
    }

    template<typename T>
    T* get() {
        return static_cast<T*>(data);
    }
};

std::vector<uint32_t> load_shape(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open shape file");
    }
    
    // Simple .npy parser for shape (assumes specific format)
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Skip numpy header (typically 128 bytes for simple arrays)
    char header[256];
    file.read(header, 128);
    
    // Read shape values (this is simplified - full .npy parsing is more complex)
    std::vector<uint32_t> shape;
    // For this example, we'll assume you know the shape or parse it differently
    // You might want to use a proper .npy library or store shape in a simpler format
    
    return shape;
}

std::set<int> load_completed(const std::string& filename) {
    std::set<int> completed;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        if (!line.empty()) {
            completed.insert(std::stoi(line));
        }
    }
    
    return completed;
}

void compute_distances(const int8_t* X, int n_vectors, int dim, 
                       int source, float* dist_from_source) {
    // Compute: -2 * (X @ X[source]) + (X[source] @ X[source])
    const int8_t* source_vec = X + (size_t)source * dim;
    
    // Compute X[source] @ X[source]
    float source_norm = 0.0f;
    for (int d = 0; d < dim; d++) {
        source_norm += source_vec[d] * source_vec[d];
    }
    
    // Compute for all vectors
    #pragma omp parallel for
    for (int i = 0; i < n_vectors; i++) {
        const int8_t* vec = X + (size_t)i * dim;
        float dot_product = 0.0f;
        
        for (int d = 0; d < dim; d++) {
            dot_product += vec[d] * source_vec[d];
        }
        
        dist_from_source[i] = -2.0f * dot_product + source_norm;
    }
}

int main() {
    const std::string DATASET = "spacev1b";
    const std::string METRIC = "euclidean";
    const std::string SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/results";
    const std::string BINARY_FILE = "/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B/vectors_int.mmap";
    const std::string SQ_NORMS_FILE = "/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B/sq_vector_norms.mmap";
    const std::string SHAPE_FILE = "/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B/vectors_shape.npy";
    
    std::cout << "Building graph on " << DATASET << std::endl;
    
    // For simplicity, hardcode shape or read from a simpler format
    // You'll need to either parse .npy properly or save shape in text format
    uint32_t n_vectors = 1000000000; // 1B vectors - adjust as needed
    uint32_t dim = 100; // dimension - adjust as needed
    
    // Memory map the vectors file
    size_t vectors_size = (size_t)n_vectors * dim * sizeof(int8_t);
    MemoryMappedFile vectors_mmap(BINARY_FILE, vectors_size);
    int8_t* X = vectors_mmap.get<int8_t>();
    
    std::cout << "Loaded vectors: " << n_vectors << " x " << dim << std::endl;
    
    // Memory map squared norms
    size_t norms_size = n_vectors * sizeof(float);
    MemoryMappedFile norms_mmap(SQ_NORMS_FILE, norms_size);
    float* sq_norms = norms_mmap.get<float>();
    
    std::cout << "Loaded squared norms" << std::endl;
    
    // Load completed sources
    std::set<int> completed = load_completed(SAVEPATH + "/" + DATASET + "-" + METRIC + "-computed.txt");
    
    // Create shuffled array of sources
    std::vector<uint32_t> all_sources(n_vectors);
    for (uint32_t i = 0; i < n_vectors; i++) {
        all_sources[i] = i;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(all_sources.begin(), all_sources.end(), gen);
    
    // Process only first source (as in Python [:1])
    for (size_t idx = 0; idx < 1 && idx < all_sources.size(); idx++) {
        uint32_t source = all_sources[idx];
        
        if (completed.find(source) != completed.end()) {
            continue;
        }
        
        // Compute neighborhood of source
        std::vector<uint32_t> edges;
        edges.push_back(source);
        
        std::vector<uint8_t> active(n_vectors, 1);
        active[source] = 0;
        
        std::cout << "Computing pairwise distances from " << source << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<float> dist_from_source(n_vectors);
        compute_distances(X, n_vectors, dim, source, dist_from_source.data());
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Took " << elapsed.count() << " seconds to compute pairwise distances" << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        
        // Count active nodes
        size_t active_count = n_vectors - 1;
        
        while (active_count > 0) {
            std::cout << "Degree: " << edges.size() - 1 << ", Left: " << active_count << std::endl;
            
            // Find minimum distance among active nodes
            float min_dist = std::numeric_limits<float>::infinity();
            uint32_t waypoint = 0;
            
            for (uint32_t i = 0; i < n_vectors; i++) {
                if (active[i] && dist_from_source[i] < min_dist) {
                    min_dist = dist_from_source[i];
                    waypoint = i;
                }
            }
            
            active[waypoint] = 0;
            active_count--;
            edges.push_back(waypoint);
            
            // Compute distances from waypoint and prune
            std::vector<float> dist_from_waypoint(n_vectors);
            compute_distances(X, n_vectors, dim, waypoint, dist_from_waypoint.data());
            
            for (uint32_t i = 0; i < n_vectors; i++) {
                if (active[i] && dist_from_waypoint[i] < dist_from_source[i]) {
                    active[i] = 0;
                    active_count--;
                }
            }
        }
        
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Took " << elapsed.count() << " seconds to compute neighborhood" << std::endl;
        
        // Write adjacency list
        std::ofstream adj_file(SAVEPATH + "/adj-list-" + DATASET + "-" + METRIC + ".txt", 
                               std::ios::app);
        for (size_t i = 0; i < edges.size(); i++) {
            adj_file << edges[i];
            if (i < edges.size() - 1) adj_file << ",";
        }
        adj_file << "\n";
        adj_file.close();
        
        // Write completed
        std::ofstream comp_file(SAVEPATH + "/" + DATASET + "-" + METRIC + "-computed.txt",
                                std::ios::app);
        comp_file << source << "\n";
        comp_file.close();
    }
    
    std::cout << "Done with " << DATASET << std::endl;
    
    return 0;
}