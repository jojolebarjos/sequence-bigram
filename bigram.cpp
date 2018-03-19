
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <random>
#include <set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif


// Partially precomputed x*log(x)
template <int max>
struct CachedEntropy {
  
  float * cache;
  
  CachedEntropy() {
    cache = new float[max];
    cache[0] = 0.0f;
    for (int n = 1; n < max; ++n)
      cache[n] = eval(n);
  }
  
  ~CachedEntropy() {
    delete cache;
  }
  
  float eval(int n) const {
    return n * std::log(n);
  }
  
  float operator()(int n) const {
    return n < max ? cache[n] : eval(n);
  }
  
};


// Main entry point
int main(int argc, char* argv[]) {
  
  // Configuration
  char const * input_path = "input.bin";
  char const * output_path = "output.bin";
  int num_words = 0;
  // TODO maybe infer num_words from input
  int num_clusters = 128;
  int num_epochs = 100;
  // TODO early stop using swapping rate and number of epoch below swapping rate?
  
  // Parse arguments
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "-i") == 0)
      input_path = argv[++i];
    else if (std::strcmp(argv[i], "-o") == 0)
      output_path = argv[++i];
    else if (std::strcmp(argv[i], "-w") == 0)
      num_words = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "-c") == 0)
      num_clusters = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "-e") == 0)
      num_epochs = std::atoi(argv[++i]);
  
  // Instantiate entropy
  CachedEntropy<10000> entropy;
  
  // Allocate buffers for global statistics
  std::vector<int> word_count(num_words);
  std::vector<int> double_word_count(num_words);
  std::vector<std::unordered_map<int32_t, int>> successors(num_words);
  std::vector<std::vector<int32_t>> predecessors(num_words);
  
  // Cluster assignment
  std::vector<int> clusters(num_words);
  std::vector<int> unary_cluster_count(num_clusters);
  std::vector<std::vector<int>> binary_cluster_count(num_clusters, std::vector<int>(num_clusters));
  
  // Word candidate statistics
  std::vector<int> predecessor_cluster_count(num_clusters);
  std::vector<int> successor_cluster_count(num_clusters);
  std::vector<float> perplexities(num_clusters);
  
  // Stream file to compute statistics
  // TODO add option to ignore 0
  {
    FILE * file = std::fopen(input_path, "rb");
    
    // Iterate over bigrams
    int32_t word;
    int32_t next_word;
    if (std::fread(&next_word, sizeof(int32_t), 1, file))
      while (true) {
        
        // Acquire next word
        word = next_word;
        if (!std::fread(&next_word, sizeof(int32_t), 1, file)) {
          
          // Also count the last word, even if it is not part of a bigram
          ++word_count[next_word];
          break;
        }
        
        // Increase counters
        ++word_count[word];
        if (word == next_word)
          ++double_word_count[word];
        
        // Update successors and predecessors
        auto & map = successors[word];
        auto entry = map.find(next_word);
        if (entry == map.end()) {
          map[next_word] = 1;
          predecessors[next_word].push_back(word);
        } else
          ++entry->second;
      }
    
    // Close file
    std::fclose(file);
  }
  
  // Assign random uniform clusters
  {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    seed = 42;
    std::minstd_rand random(seed);
    std::uniform_int_distribution<> distribution(0, num_clusters - 1);
    for (int word = 0; word < num_words; ++word)
      clusters[word] = distribution(random);
  }
  
  // Initialize cluster count
  for (int word = 0; word < num_words; ++word) {
    unary_cluster_count[clusters[word]] += word_count[word];
    for (auto const & entry : successors[word]) {
      binary_cluster_count[clusters[word]][clusters[entry.first]] += entry.second;
    }
  }
  
  // Loop until convergence
  int epoch = 0;
  auto start = std::chrono::high_resolution_clock::now();
  while (true) {
    
    // Update epoch
    ++epoch;
    if (epoch > num_epochs) {
      printf("Maximal number of epochs reached\n");
      break;
    }
    
    // TODO early stopping
    
    // Consider each word
    int swap_count = 0;
    for (int word = 0; word < num_words; ++word) {
      
      // Generate cluster count
      for (int cluster = 0; cluster < num_clusters; ++cluster) {
        predecessor_cluster_count[cluster] = 0;
        successor_cluster_count[cluster] = 0;
      }
      for (int32_t predecessor : predecessors[word])
        predecessor_cluster_count[clusters[predecessor]] += successors[predecessor][word];
      for (auto const & entry : successors[word])
        successor_cluster_count[clusters[entry.first]] += entry.second;
      
      // Remove word from its cluster
      for (int cluster = 0; cluster < num_clusters; ++cluster)
        if (cluster != clusters[word]) {
          binary_cluster_count[cluster][clusters[word]] -= predecessor_cluster_count[cluster];
          binary_cluster_count[clusters[word]][cluster] -= successor_cluster_count[cluster];
        }
      unary_cluster_count[clusters[word]] -= word_count[word];
      binary_cluster_count[clusters[word]][clusters[word]] += double_word_count[word] - predecessor_cluster_count[clusters[word]] - successor_cluster_count[clusters[word]];
      predecessor_cluster_count[clusters[word]] -= double_word_count[word];
      successor_cluster_count[clusters[word]] -= double_word_count[word];
      
      // Compute expected perplexity for each assignment
      #pragma omp parallel for
      for (int candidate_cluster = 0; candidate_cluster < num_clusters; ++candidate_cluster) {
        float perplexity = 0;
        for (int cluster = 0; cluster < num_clusters; ++cluster)
          if (cluster != candidate_cluster) {
            int p = binary_cluster_count[cluster][candidate_cluster] + predecessor_cluster_count[cluster];
            perplexity += entropy(p);
            perplexity -= entropy(binary_cluster_count[cluster][candidate_cluster]);
            int s = binary_cluster_count[candidate_cluster][cluster] + successor_cluster_count[cluster];
            perplexity += entropy(s);
            perplexity -= entropy(binary_cluster_count[candidate_cluster][cluster]);
          }
        int u = unary_cluster_count[candidate_cluster] + word_count[word];
        perplexity -= 2 * entropy(u);
        perplexity += 2 * entropy(unary_cluster_count[candidate_cluster]);
        int d = binary_cluster_count[candidate_cluster][candidate_cluster] + successor_cluster_count[candidate_cluster] + predecessor_cluster_count[candidate_cluster] + double_word_count[word];
        perplexity += entropy(d);
        perplexity -= entropy(binary_cluster_count[candidate_cluster][candidate_cluster]);
        perplexities[candidate_cluster] = perplexity;
      }
      
      // Assign best cluster
      int assigned_cluster = std::max_element(perplexities.begin(), perplexities.end()) - perplexities.begin();
      for (int cluster = 0; cluster < num_clusters; ++cluster)
        if (cluster != assigned_cluster) {
          binary_cluster_count[cluster][assigned_cluster] += predecessor_cluster_count[cluster];
          binary_cluster_count[assigned_cluster][cluster] += successor_cluster_count[cluster];
        }
      unary_cluster_count[assigned_cluster] += word_count[word];
      binary_cluster_count[assigned_cluster][assigned_cluster] += successor_cluster_count[assigned_cluster] + predecessor_cluster_count[assigned_cluster] + double_word_count[word];
      
      // Check if the word has actually been moved
      if (assigned_cluster != clusters[word]) {
        ++swap_count;
        clusters[word] = assigned_cluster;
      }
    }
    
    // Compute total perplexity
    float perplexity = 0;
    for (int count : word_count)
      if (count > 0)
        perplexity += entropy(count);
    for (int count : unary_cluster_count)
      if (count > 0)
        perplexity -= 2 * entropy(count);
    for (auto const & entry : binary_cluster_count)
      for (int count : entry)
        if (count > 0)
          perplexity += entropy(count);
    
    // Report epoch
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start;
    start = now;
    printf("%d/%d: %f, %d swaps, %2.2f seconds\n", epoch, num_epochs, perplexity, swap_count, elapsed.count());
    
    // Abort if no swap were performed
    if (swap_count == 0) {
      printf("No more swap can be applied\n");
      break;
    }
  }
  
  // Export result
  {
    FILE * file = std::fopen(output_path, "wb");
    for (int32_t cluster : clusters)
      std::fwrite(&cluster, sizeof(int32_t), 1, file);
    std::fclose(file);
  }
  
  // Done.
  return 0;
}
