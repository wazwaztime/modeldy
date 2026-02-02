/*!
 * Copyright (c) 2026 Andrew Wang. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef MODELDY_INCLUDE_MEMORY_POOL_H_
#define MODELDY_INCLUDE_MEMORY_POOL_H_

#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <cstddef>
#include <functional>

#ifdef USE_CUDA
#include <include/cuda/cuda_check.h>
#endif

namespace modeldy {

/*! \brief Base memory pool interface */
template <typename T>
class MemoryPool {
 public:
  virtual ~MemoryPool() = default;
  
  /*! \brief Allocate memory of given size, returns raw pointer */
  virtual T* allocate(size_t size) = 0;
  
  /*! \brief Deallocate memory and return to pool */
  virtual void deallocate(T* ptr, size_t size) = 0;
  
  /*! \brief Clear all cached memory */
  virtual void clear() = 0;
};

namespace cpu {

/*! \brief CPU memory pool with caching */
template <typename T>
class cpuMemoryPool : public MemoryPool<T> {
 public:
  static cpuMemoryPool<T>& getInstance() {
    static cpuMemoryPool<T> instance;
    return instance;
  }

  T* allocate(size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Try to find available block in cache
    auto it = free_blocks_.find(size);
    if (it != free_blocks_.end() && !it->second.empty()) {
      T* ptr = it->second.back();
      it->second.pop_back();
      cached_memory_size_ -= size * sizeof(T);
      allocated_blocks_[ptr] = size;  // Track allocation
      return ptr;
    }
    
    // Allocate new memory
    T* raw_ptr = new T[size]();
    allocated_blocks_[raw_ptr] = size;  // Track allocation
    total_allocated_memory_ += size * sizeof(T);
    return raw_ptr;
  }

  void deallocate(T* ptr, size_t size) override {
    if (ptr == nullptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Verify this pointer was allocated by us
    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
      // This pointer wasn't allocated by this pool - potential error
      return;
    }
    
    // Verify size matches
    if (it->second != size) {
      // Size mismatch - use the original size to avoid memory leak
      size = it->second;
    }
    
    allocated_blocks_.erase(it);
    
    // Check if we should cache this block
    size_t block_memory = size * sizeof(T);
    if (free_blocks_[size].size() < max_cached_blocks_ && 
        cached_memory_size_ + block_memory < max_cached_memory_) {
      free_blocks_[size].push_back(ptr);
      cached_memory_size_ += block_memory;
    } else {
      // Too many cached blocks or too much memory, just delete
      delete[] ptr;
      total_allocated_memory_ -= block_memory;
    }
  }

  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Delete all cached blocks
    for (auto& pair : free_blocks_) {
      for (T* ptr : pair.second) {
        delete[] ptr;
        total_allocated_memory_ -= pair.first * sizeof(T);
      }
    }
    free_blocks_.clear();
    cached_memory_size_ = 0;
  }

  /*! \brief Get memory usage statistics */
  size_t getTotalAllocatedMemory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_memory_;
  }
  
  size_t getCachedMemorySize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cached_memory_size_;
  }
  
  size_t getActiveAllocations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_blocks_.size();
  }

  ~cpuMemoryPool() override {
    clear();
    // Warn if there are still active allocations (potential memory leak)
    if (!allocated_blocks_.empty()) {
      // Log warning: Memory leak detected
    }
  }

 private:
  cpuMemoryPool() = default;
  cpuMemoryPool(const cpuMemoryPool&) = delete;
  cpuMemoryPool& operator=(const cpuMemoryPool&) = delete;

  mutable std::mutex mutex_;
  std::unordered_map<size_t, std::vector<T*>> free_blocks_;  // size -> free blocks
  std::unordered_map<T*, size_t> allocated_blocks_;  // ptr -> size (track active allocations)
  size_t total_allocated_memory_ = 0;  // Total memory allocated from system
  size_t cached_memory_size_ = 0;  // Memory currently in cache
  
  static constexpr size_t max_cached_blocks_ = 100;  // Max blocks to cache per size
  static constexpr size_t max_cached_memory_ = 1024 * 1024 * 1024;  // Max 1GB in cache
};

} // namespace cpu

#ifdef USE_CUDA
namespace cuda {

/*! \brief CUDA memory pool with caching */
template <typename T>
class cudaMemoryPool : public MemoryPool<T> {
 public:
  static cudaMemoryPool<T>& getInstance() {
    static cudaMemoryPool<T> instance;
    return instance;
  }

  T* allocate(size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Try to find available block in cache
    auto it = free_blocks_.find(size);
    if (it != free_blocks_.end() && !it->second.empty()) {
      T* ptr = it->second.back();
      it->second.pop_back();
      cached_memory_size_ -= size * sizeof(T);
      allocated_blocks_[ptr] = size;  // Track allocation
      return ptr;
    }
    
    // Allocate new CUDA memory
    T* raw_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&raw_ptr, size * sizeof(T)));
    
    // Initialize memory to zero
    CUDA_CHECK(cudaMemset(raw_ptr, 0, size * sizeof(T)));
    
    allocated_blocks_[raw_ptr] = size;  // Track allocation
    total_allocated_memory_ += size * sizeof(T);
    return raw_ptr;
  }

  void deallocate(T* ptr, size_t size) override {
    if (ptr == nullptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Verify this pointer was allocated by us
    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
      // This pointer wasn't allocated by this pool - potential error
      return;
    }
    
    // Verify size matches
    if (it->second != size) {
      // Size mismatch - use the original size to avoid memory leak
      size = it->second;
    }
    
    allocated_blocks_.erase(it);
    
    // Check if we should cache this block
    size_t block_memory = size * sizeof(T);
    if (free_blocks_[size].size() < max_cached_blocks_ && 
        cached_memory_size_ + block_memory < max_cached_memory_) {
      free_blocks_[size].push_back(ptr);
      cached_memory_size_ += block_memory;
    } else {
      // Too many cached blocks or too much memory, just free
      CUDA_CHECK(cudaFree(ptr));
      total_allocated_memory_ -= block_memory;
    }
  }

  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Free all cached blocks
    for (auto& pair : free_blocks_) {
      for (T* ptr : pair.second) {
        CUDA_CHECK(cudaFree(ptr));
        total_allocated_memory_ -= pair.first * sizeof(T);
      }
    }
    free_blocks_.clear();
    cached_memory_size_ = 0;
  }

  /*! \brief Get memory usage statistics */
  size_t getTotalAllocatedMemory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_memory_;
  }
  
  size_t getCachedMemorySize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cached_memory_size_;
  }
  
  size_t getActiveAllocations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_blocks_.size();
  }

  ~cudaMemoryPool() override {
    clear();
    // Warn if there are still active allocations (potential memory leak)
    if (!allocated_blocks_.empty()) {
      // Log warning: Memory leak detected
    }
  }

 private:
  cudaMemoryPool() = default;
  cudaMemoryPool(const cudaMemoryPool&) = delete;
  cudaMemoryPool& operator=(const cudaMemoryPool&) = delete;

  mutable std::mutex mutex_;
  std::unordered_map<size_t, std::vector<T*>> free_blocks_;  // size -> free blocks
  std::unordered_map<T*, size_t> allocated_blocks_;  // ptr -> size (track active allocations)
  size_t total_allocated_memory_ = 0;  // Total memory allocated from system
  size_t cached_memory_size_ = 0;  // Memory currently in cache
  
  static constexpr size_t max_cached_blocks_ = 100;  // Max blocks to cache per size
  static constexpr size_t max_cached_memory_ = 1024 * 1024 * 1024;  // Max 1GB in cache
};

} // namespace cuda

#endif // USE_CUDA

} // namespace modeldy

#endif // MODELDY_INCLUDE_MEMORY_POOL_H_
