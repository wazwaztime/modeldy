# Modeldy 训练与优化器完整指南

本文档介绍 Modeldy 的训练接口、优化器系统以及 CPU/CUDA 双实现架构。

---

## 目录

1. [概述](#概述)
2. [优化器](#优化器)
3. [训练流程](#训练流程)
4. [API 说明](#api-说明)
5. [高级用法](#高级用法)
6. [CPU/CUDA 双实现架构](#cpucuda-双实现架构)
7. [CUDA Kernel 实现指南](#cuda-kernel-实现指南)
8. [优化建议](#优化建议)
9. [编译与测试](#编译与测试)

---

## 概述

Modeldy 提供了完整的训练接口，包括多种优化器和简单的训练循环 API。优化器系统支持 CPU 和 CUDA 两种实现，能够自动检测参数类型并调用相应的优化算法。

**核心特性：**
- 🚀 三种主流优化器：SGD、Adam、RMSprop
- 🔄 自动 CPU/CUDA 设备检测
- 📦 简单易用的训练 API
- ⚡ 完整的 CUDA 加速框架

---

## 优化器

### 可用的优化器

#### 1. SGD (随机梯度下降)

```cpp
modeldy::SGD<float> optimizer(
    0.01f,      // learning_rate
    0.0f,       // momentum (可选，默认0)
    0.0f        // weight_decay (L2正则化，可选，默认0)
);
```

**特点：**
- 简单高效
- 支持动量加速
- 支持权重衰减（L2正则化）
- CPU 和 CUDA 实现都已就绪

**推荐学习率：** 0.001 ~ 0.1

#### 2. Adam

```cpp
modeldy::Adam<float> optimizer(
    0.001f,     // learning_rate
    0.9f,       // beta1 (一阶矩估计的指数衰减率，可选)
    0.999f,     // beta2 (二阶矩估计的指数衰减率，可选)
    1e-8f,      // epsilon (数值稳定性，可选)
    0.0f        // weight_decay (可选)
);
```

**特点：**
- 自适应学习率
- 对超参数不敏感
- 适合大多数深度学习任务
- 收敛速度快

**推荐学习率：** 0.0001 ~ 0.01

#### 3. RMSprop

```cpp
modeldy::RMSprop<float> optimizer(
    0.01f,      // learning_rate
    0.99f,      // alpha (衰减率，可选)
    1e-8f,      // epsilon (可选)
    0.0f        // weight_decay (可选)
);
```

**特点：**
- 适合处理非平稳目标
- 适合 RNN 训练
- 自适应学习率

**推荐学习率：** 0.001 ~ 0.01

---

## 训练流程

### 完整训练示例

```cpp
#include <modeldy/include/model.h>
#include <modeldy/include/optimizer.h>
#include <modeldy/include/operator_registry.h>

int main() {
    // 1. 创建模型
    modeldy::Model<float> model;
    
    // 2. 定义网络结构
    model.newDataNode("input", {batch_size, input_dim}, false, "cpu");
    model.newDataNode("weights", {input_dim, output_dim}, true, "cpu");  // 可训练参数
    model.newDataNode("bias", {output_dim}, true, "cpu");                // 可训练参数
    model.newDataNode("output", {batch_size, output_dim}, true, "cpu");
    model.newDataNode("target", {batch_size, output_dim}, false, "cpu");
    model.newDataNode("loss", {1}, true, "cpu");
    
    // 3. 初始化参数
    model.setData("weights", initial_weights);
    model.setData("bias", initial_bias);
    
    // 4. 标记可训练参数
    model.add_parameter("weights");
    model.add_parameter("bias");
    
    // 5. 构建计算图
    model.newComputeNode("GemmOO", "linear", {"input", "weights"}, {"temp"}, "cpu");
    model.newComputeNode("Add", "add_bias", {"temp", "bias"}, {"output"}, "cpu");
    model.newComputeNode("MSELoss", "loss_fn", {"output", "target"}, {"loss"}, "cpu");
    
    // 6. 创建优化器并关联参数
    modeldy::Adam<float> optimizer(0.001f);
    model.setup_optimizer(optimizer);
    
    // 7. 训练循环
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // 设置输入数据
        model.setData("input", batch_input);
        model.setData("target", batch_target);
        
        // 单步训练
        float loss = model.train_step(optimizer, "loss", true);
        
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    }
    
    // 或者使用批量训练
    auto losses = model.train(optimizer, "loss", 100, 10);  // 100次迭代，每10次打印
    
    return 0;
}
```

---

## API 说明

### Model 类训练相关方法

#### add_parameter()
```cpp
void add_parameter(const std::string& name)
```

标记一个数据节点为可训练参数。

**参数：**
- `name`: 参数节点的名称

**要求：**
- 节点必须是 DataNode
- 节点必须设置 `requires_grad=true`

#### setup_optimizer()
```cpp
void setup_optimizer(Optimizer<T>& optimizer)
```

将模型的所有可训练参数注册到优化器。

**参数：**
- `optimizer`: 优化器实例的引用

#### train_step()
```cpp
T train_step(Optimizer<T>& optimizer, const std::string& loss_node_name, bool verbose = false)
```

执行一次训练迭代（前向传播、反向传播、参数更新）。

**参数：**
- `optimizer`: 优化器
- `loss_node_name`: 损失节点的名称
- `verbose`: 是否打印损失值

**返回：**
- 当前迭代的损失值

#### train()
```cpp
std::vector<T> train(Optimizer<T>& optimizer,
                     const std::string& loss_node_name,
                     size_t num_iterations,
                     size_t print_every = 0)
```

执行多次训练迭代。

**参数：**
- `optimizer`: 优化器
- `loss_node_name`: 损失节点的名称
- `num_iterations`: 训练迭代次数
- `print_every`: 每 N 次迭代打印一次（0 表示不打印）

**返回：**
- 每次迭代的损失值向量

### Optimizer 类方法

#### step()
```cpp
virtual void step() = 0
```

根据梯度更新所有参数。

#### zero_grad()
```cpp
virtual void zero_grad()
```

将所有参数的梯度清零。

#### learning_rate() / set_learning_rate()
```cpp
T learning_rate() const
void set_learning_rate(T lr)
```

获取或设置学习率。

---

## 高级用法

### 学习率调度

```cpp
modeldy::Adam<float> optimizer(0.1f);
model.setup_optimizer(optimizer);

// 训练前期
model.train(optimizer, "loss", 50);

// 降低学习率
optimizer.set_learning_rate(0.01f);

// 继续训练
model.train(optimizer, "loss", 50);
```

### 使用动量

```cpp
// SGD with momentum
modeldy::SGD<float> optimizer(
    0.01f,      // learning_rate
    0.9f,       // momentum
    0.0f        // weight_decay
);
```

### L2 正则化

```cpp
// 使用权重衰减进行L2正则化
modeldy::Adam<float> optimizer(
    0.001f,     // learning_rate
    0.9f,       // beta1
    0.999f,     // beta2
    1e-8f,      // epsilon
    0.01f       // weight_decay (L2 regularization)
);
```

### 自定义训练循环

```cpp
modeldy::SGD<float> optimizer(0.01f);
model.setup_optimizer(optimizer);

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    float total_loss = 0.0f;
    
    for (int batch = 0; batch < num_batches; ++batch) {
        // 加载批次数据
        model.setData("input", batch_input[batch]);
        model.setData("target", batch_target[batch]);
        
        // 前向传播
        model.predict();
        const float* loss_data = model.data("loss");
        total_loss += loss_data[0];
        
        // 反向传播
        optimizer.zero_grad();
        model.backward("loss");
        
        // 更新参数
        optimizer.step();
    }
    
    std::cout << "Epoch " << epoch 
              << ", Average Loss: " << total_loss / num_batches << std::endl;
}
```

---

## CPU/CUDA 双实现架构

### 架构设计

优化器系统支持 CPU 和 CUDA 两种实现，能够自动检测参数类型并调用相应的实现。

#### 1. 自动设备检测

优化器会自动检测参数是 `cpuDataNode` 还是 `cudaDataNode`，并调用相应的实现：

```cpp
void step() override {
  for (auto& param : this->parameters_) {
    if (param.is_cuda) {
      step_cuda(param);  // CUDA 实现
    } else {
      step_cpu(param);   // CPU 实现
    }
  }
}
```

#### 2. 分离的实现

每个优化器类都包含两个独立的实现方法：
- `step_cpu()` - CPU 实现（已完成）
- `step_cuda()` - CUDA 实现（框架已就绪，需要实现 kernel）

### 文件结构

```
include/
├── optimizer.h                    # 优化器主文件
└── cuda/
    └── optimizer_kernels.h        # CUDA kernel 声明和启动函数
```

### 使用示例

#### CPU 训练
```cpp
modeldy::Model<float> model;
// ... 设置 CPU 节点 ...

modeldy::Adam<float> optimizer(0.001f);
model.setup_optimizer(optimizer);

// 自动使用 CPU 实现
model.train(optimizer, "loss", 100);
```

#### CUDA 训练
```cpp
#ifdef USE_CUDA
modeldy::Model<float> model;
// ... 设置 CUDA 节点 ...

modeldy::Adam<float> optimizer(0.001f);
model.setup_optimizer(optimizer);

// 自动使用 CUDA 实现
model.train(optimizer, "loss", 100);
#endif
```

#### 混合模式
系统支持混合 CPU 和 CUDA 参数，每个参数会自动使用对应的实现。

---

## CUDA Kernel 实现指南

### 需要实现的 CUDA Kernels

在 `include/cuda/optimizer_kernels.h` 中，已提供以下 kernel 的框架：

#### 1. SGD Kernels

```cuda
// 无动量版本
template <typename T>
__global__ void sgd_kernel(T* data, const T* grad, size_t size, 
                          T lr, T weight_decay)

// 带动量版本
template <typename T>
__global__ void sgd_momentum_kernel(T* data, const T* grad, T* velocity,
                                   size_t size, T lr, T momentum, T weight_decay)
```

**公式：**
- 无动量: `data = data - lr * (grad + weight_decay * data)`
- 有动量: 
  ```
  velocity = momentum * velocity + (grad + weight_decay * data)
  data = data - lr * velocity
  ```

#### 2. Adam Kernel

```cuda
template <typename T>
__global__ void adam_kernel(T* data, const T* grad, T* m, T* v,
                           size_t size, T lr, T beta1, T beta2,
                           T epsilon, T weight_decay, size_t t)
```

**公式：**
```
g = grad + weight_decay * data
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
data = data - lr * m_hat / (sqrt(v_hat) + epsilon)
```

#### 3. RMSprop Kernel

```cuda
template <typename T>
__global__ void rmsprop_kernel(T* data, const T* grad, T* square_avg,
                              size_t size, T lr, T alpha,
                              T epsilon, T weight_decay)
```

**公式：**
```
g = grad + weight_decay * data
square_avg = alpha * square_avg + (1 - alpha) * g^2
data = data - lr * g / (sqrt(square_avg) + epsilon)
```

### 实现步骤

#### 1. 取消注释 kernel 代码

在 `include/cuda/optimizer_kernels.h` 中，每个 kernel 函数体都有注释的实现代码，取消注释即可：

```cuda
template <typename T>
__global__ void sgd_kernel(T* data, const T* grad, size_t size, T lr, T weight_decay) {
  // 取消下面的注释
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T g = grad[idx];
    if (weight_decay > 0) {
      g += weight_decay * data[idx];
    }
    data[idx] -= lr * g;
  }
}
```

#### 2. 取消注释 kernel 启动函数

```cuda
template <typename T>
void sgd_kernel_launch(T* data, const T* grad, size_t size, T lr, T weight_decay) {
  // 取消下面的注释
  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;
  sgd_kernel<<<grid_size, block_size>>>(data, grad, size, lr, weight_decay);
  CUDA_CHECK(cudaGetLastError());
}
```

#### 3. 创建 .cu 文件（可选）

如果需要将 kernel 实现分离到 .cu 文件：

```cuda
// src/cuda/optimizer_kernels.cu
#include <modeldy/include/cuda/optimizer_kernels.h>

namespace modeldy {
namespace cuda {

// 实现所有 kernel 和启动函数
// ...

} // namespace cuda
} // namespace modeldy
```

### 内存管理

#### CPU 实现
- 使用 `std::vector<T>` 存储辅助变量（velocity, momentum 等）
- 自动管理内存

#### CUDA 实现
- 使用 `T*` 设备指针存储辅助变量
- 在第一次使用时分配：
  ```cpp
  CUDA_CHECK(cudaMalloc(&velocity_ptr, total_size * sizeof(T)));
  CUDA_CHECK(cudaMemset(velocity_ptr, 0, total_size * sizeof(T)));
  ```
- 建议在析构函数中释放内存

---

## 优化建议

### CUDA Kernel 优化技巧

#### 1. 线程块大小
- 推荐 256 或 512
- 根据寄存器使用情况调整
- 使用 occupancy calculator 确定最优值

#### 2. 内存访问
- 确保合并访问（coalesced access）
- 使用 shared memory 优化（高级）
- 避免 bank conflicts

#### 3. 数值稳定性
- 注意除零检查
- 使用 `rsqrtf()` 代替 `1.0f / sqrtf()`
- 使用 FMA (fused multiply-add) 指令

#### 4. 优化示例

```cuda
__global__ void sgd_kernel_optimized(T* data, const T* grad, 
                                    size_t size, T lr, T weight_decay) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  // Grid-stride loop for better workload distribution
  for (int i = idx; i < size; i += stride) {
    T g = grad[i];
    if (weight_decay > 0) {
      g = fmaf(weight_decay, data[i], g);  // 使用 FMA
    }
    data[i] = fmaf(-lr, g, data[i]);  // 使用 FMA
  }
}
```

### 性能提示

1. **批量大小**：增大批量大小以提高 GPU 利用率
2. **异步操作**：使用 CUDA streams 进行异步计算
3. **数据传输**：最小化 CPU-GPU 数据传输
4. **混合精度**：考虑使用 FP16 加速训练（需要额外实现）

---

## 编译与测试

### CPU 模式编译

```powershell
# 从 Desktop/modeldy 目录
g++ -std=c++17 -I. modeldy/examples/training_example.cpp -o training_example.exe
.\training_example.exe
```

### CUDA 模式编译

```bash
# 编译 CUDA kernels
nvcc -c src/cuda/optimizer_kernels.cu -o optimizer_kernels.o -DUSE_CUDA

# 链接最终程序
nvcc -std=c++17 -I. modeldy/examples/training_example.cpp optimizer_kernels.o -o training_example -DUSE_CUDA

# 运行
.\training_example
```

### 测试用例

运行梯度测试以验证实现的正确性：

```powershell
cd Desktop/modeldy
g++ -std=c++17 -I. modeldy/examples/loss_gradient_test.cpp -o loss_gradient_test.exe
.\loss_gradient_test.exe
```

---

## 注意事项

### 通用注意事项

1. **梯度设置**：所有参与反向传播的中间节点都需要设置 `requires_grad=true`
2. **参数初始化**：训练前要合理初始化参数，避免梯度消失或爆炸
3. **内存管理**：每次迭代前确保数据已正确加载
4. **数值稳定性**：损失函数和激活函数内部已包含数值稳定性处理

### CUDA 特定注意事项

1. **设备同步**：在读取结果前确保 kernel 执行完成
2. **错误检查**：使用 `CUDA_CHECK` 宏检查所有 CUDA 调用
3. **内存泄漏**：确保正确释放分配的设备内存
4. **计算能力**：确认 GPU 支持所需的 CUDA 计算能力

---

## 状态总结

### ✅ 已完成

- CPU 实现（SGD, Adam, RMSprop）
- 训练接口和 API
- 自动设备检测机制
- CUDA 框架和接口
- 内存管理框架
- 完整的文档和示例

### ⏳ 待实现

- CUDA kernel 具体实现（框架和公式已提供）
- CUDA kernel 性能优化
- 混合精度支持（可选）
- 分布式训练支持（可选）

---

## 示例输出

```
=== Training Simple Network with Different Optimizers ===

--- Using SGD ---
Initial weight: 0.1, Target: 5.0
Iteration 5/20, Loss: 4.957568
Iteration 10/20, Loss: 4.346873
Iteration 15/20, Loss: 23.644493
Iteration 20/20, Loss: 16.507284
Final weight: 9.947211
Expected: ~5.0

--- Using Adam ---
Initial weight: 0.1, Target: 5.0
Iteration 5/20, Loss: 20.281775
Iteration 10/20, Loss: 15.998478
Iteration 15/20, Loss: 12.053538
Iteration 20/20, Loss: 8.512946
Final weight: 2.196091
Expected: ~5.0
```

---

## 参考资源

- **项目示例**：`examples/training_example.cpp`
- **梯度测试**：`examples/loss_gradient_test.cpp`
- **优化器头文件**：`include/optimizer.h`
- **CUDA kernel 框架**：`include/cuda/optimizer_kernels.h`

---

**最后更新：** 2026-02-02  
**版本：** 1.0
