# Modeldy 文档目录

本目录包含 Modeldy 深度学习框架的所有文档。

## 📚 文档列表

### [训练与优化器完整指南](TRAINING_AND_OPTIMIZER_GUIDE.md)
全面介绍 Modeldy 的训练接口、优化器系统以及 CPU/CUDA 双实现架构。

**包含内容：**
- 优化器使用指南（SGD、Adam、RMSprop）
- 训练流程和 API 说明
- 高级用法（学习率调度、动量、L2正则化）
- CPU/CUDA 双实现架构设计
- CUDA Kernel 实现指南
- 性能优化建议
- 编译与测试方法

**适用对象：**
- 想要使用 Modeldy 进行模型训练的开发者
- 需要实现自定义优化器的高级用户
- 想要为 CUDA 优化器实现 kernel 的开发者

---

## 📂 文档组织结构

```
docs/
├── README.md                              # 本文件
└── TRAINING_AND_OPTIMIZER_GUIDE.md        # 训练与优化器指南
```

---

## 🚀 快速开始

如果你是第一次使用 Modeldy 的训练功能，建议按以下顺序阅读：

1. **基础使用**：阅读"优化器"和"训练流程"章节
2. **API 参考**：查看"API 说明"了解详细接口
3. **进阶技巧**：学习"高级用法"中的各种技巧
4. **性能优化**：如需 CUDA 加速，参考"CUDA Kernel 实现指南"

---

## 💡 常见问题

### Q: 如何选择合适的优化器？
- **SGD**：简单任务，需要精确控制
- **Adam**：大多数场景的首选，收敛快
- **RMSprop**：处理 RNN 或非平稳问题

### Q: CPU 和 CUDA 版本如何切换？
优化器会自动检测参数类型（cpuDataNode 或 cudaDataNode），无需手动切换。

### Q: 如何实现自定义训练循环？
参考"高级用法"中的"自定义训练循环"示例。

### Q: CUDA kernel 在哪里实现？
所有 CUDA kernel 框架都在 `include/cuda/optimizer_kernels.h` 中，取消注释即可使用。

---

## 📝 贡献指南

如果你想为文档做出贡献：

1. 所有文档使用 Markdown 格式
2. 代码示例应该可以直接运行
3. 添加新文档后更新本 README
4. 保持文档结构清晰，使用合适的标题层级

---

## 🔗 相关链接

- **示例代码**：`../examples/`
- **头文件**：`../include/`
- **测试用例**：`../examples/loss_gradient_test.cpp`

---

**维护者：** Andrew Wang
**最后更新：** 2026-02-02
