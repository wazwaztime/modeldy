#include <iostream>
#include <vector>
#include <include/model.h>

int main() {
    try {
        std::cout << "Testing CUDA MSE Loss..." << std::endl;
        
        const std::vector<size_t> shape = {4};
        std::vector<float> predictions = {0.7f, 0.2f, 0.8f, 0.3f};
        std::vector<float> targets = {1.0f, 0.0f, 1.0f, 0.0f};
        
        // CUDA Model
        modeldy::Model<float> cuda_model;
        cuda_model.newDataNode("pred", shape, true, "cuda");
        cuda_model.newDataNode("target", shape, false, "cuda");
        cuda_model.newDataNode("loss", {1}, true, "cuda");
        
        std::cout << "Setting data..." << std::endl;
        cuda_model.setData("pred", predictions);
        cuda_model.setData("target", targets);
        
        std::cout << "Creating compute node..." << std::endl;
        cuda_model.newComputeNode("MSELoss", "loss_node", {"pred", "target"}, {"loss"}, "cuda");
        
        std::cout << "Compiling..." << std::endl;
        cuda_model.compile();
        
        std::cout << "Running predict..." << std::endl;
        cuda_model.predict();
        
        std::cout << "Getting loss..." << std::endl;
        auto cuda_loss = cuda_model.getData("loss");
        
        std::cout << "Loss: " << cuda_loss[0] << std::endl;
        
        std::cout << "Initializing gradient..." << std::endl;
        cuda_model.initGrad("loss", 1.0f);
        
        std::cout << "Running backward..." << std::endl;
        cuda_model.backward("loss");
        
        std::cout << "Getting gradients..." << std::endl;
        auto cuda_grad = cuda_model.getGrad("pred");
        
        std::cout << "Gradients: ";
        for (auto g : cuda_grad) {
            std::cout << g << " ";
        }
        std::cout << std::endl;
        
        std::cout << "✓ Test passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
