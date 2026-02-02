#include <iostream>
#include <vector>
#include <include/cuda/node_cuda.h>
#include <include/cuda/operator/loss_op.h>

int main() {
    try {
        std::cout << "Creating CUDA data nodes..." << std::endl;
        auto pred = std::make_shared<modeldy::cudaDataNode<float>>(std::vector<size_t>{4}, true, "pred");
        auto target = std::make_shared<modeldy::cudaDataNode<float>>(std::vector<size_t>{4}, false, "target");
        auto loss = std::make_shared<modeldy::cudaDataNode<float>>(std::vector<size_t>{1}, false, "loss");
        
        std::cout << "Setting data..." << std::endl;
        pred->copy_from({0.7f, 0.2f, 0.8f, 0.3f});
        target->copy_from({1.0f, 0.0f, 1.0f, 0.0f});
        
        std::cout << "Creating MSELoss operator..." << std::endl;
        modeldy::cuda::cudaMSELoss<float> mse_op({pred, target}, {loss});
        
        std::cout << "Running forward..." << std::endl;
        mse_op.forward();
        
        std::cout << "Getting loss value..." << std::endl;
        std::vector<float> loss_val = loss->data_as_vector();
        std::cout << "Loss: " << loss_val[0] << std::endl;
        
        std::cout << " Test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << " Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
