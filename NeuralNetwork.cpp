#include "MLPerceptrons.h"
#include <iostream>
#include <iomanip>

void trainXOR(MultilayerPerceptron& mlp, const std::string& activationName, int epochs = 3000) 
{
    std::cout << "Training the Neural Network as an XOR Gate using " << activationName << "..." << std::endl;

    double MSE;
    double minMSE = 1.0;
    
    for (int i = 0; i < epochs; i++) 
    {
        MSE = 0.0;
        MSE += mlp.backPropagation({0, 0}, {0});     // inputs and expected outputs
        MSE += mlp.backPropagation({0, 1}, {1});
        MSE += mlp.backPropagation({1, 0}, {1});
        MSE += mlp.backPropagation({1, 1}, {0});
        MSE /= 4.0;
        
        // Track the best MSE
        if (MSE < minMSE) 
        {
            minMSE = MSE;
        }
        
        if (i % 500 == 0 || i == epochs-1) 
        {
            std::cout << "Epoch " << i << " - MSE = " << MSE << std::endl;
        }
        
        // Early stopping if training is going well
        if (MSE < 0.01) 
        {
            std::cout << "Early stopping at epoch " << i << " with MSE = " << MSE << std::endl;
            break;
        }
    }
    
    std::cout << "\nFinal MSE: " << MSE << " (Best: " << minMSE << ")" << std::endl;
    std::cout << "Trained Weights:";
    mlp.printWeights();

    // Use a threshold to convert continuous outputs to binary for clearer results
    auto threshold = [](double x) -> int { return x >= 0.5 ? 1 : 0; };
    
    std::cout << "XOR Results with " << activationName << ":" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    double out00 = mlp.run({0, 0})[0];
    double out01 = mlp.run({0, 1})[0];
    double out10 = mlp.run({1, 0})[0];
    double out11 = mlp.run({1, 1})[0];
    
    std::cout << "Input: 0,0 | Output: " << out00 << " | Binary: " << threshold(out00) << " | Expected: 0" << std::endl;
    std::cout << "Input: 0,1 | Output: " << out01 << " | Binary: " << threshold(out01) << " | Expected: 1" << std::endl;
    std::cout << "Input: 1,0 | Output: " << out10 << " | Binary: " << threshold(out10) << " | Expected: 1" << std::endl;
    std::cout << "Input: 1,1 | Output: " << out11 << " | Binary: " << threshold(out11) << " | Expected: 0" << std::endl;
    
    // Calculate and display accuracy
    int correct = 0;
    correct += (threshold(out00) == 0) ? 1 : 0;
    correct += (threshold(out01) == 1) ? 1 : 0;
    correct += (threshold(out10) == 1) ? 1 : 0;
    correct += (threshold(out11) == 0) ? 1 : 0;
    
    std::cout << "Accuracy: " << (correct * 100.0 / 4.0) << "%" << std::endl;
}

int main() 
{
    srand(time(NULL));
    rand();

    // Test XOR with different activation functions
    std::cout << "\n\n ------------------------- Trained XOR Example with Sigmoid ------------------------- \n\n";
    MultilayerPerceptron mlpSigmoid({2, 4, 1}, ActivationType::SIGMOID, 1.0, 0.1);
    trainXOR(mlpSigmoid, "Sigmoid");
    
    std::cout << "\n\n ------------------------- Trained XOR Example with Tanh ------------------------- \n\n";
    MultilayerPerceptron mlpTanh({2, 4, 1}, ActivationType::TANH, 1.0, 0.1);
    trainXOR(mlpTanh, "Tanh");
    
    std::cout << "\n\n ------------------------- Trained XOR Example with ReLU ------------------------- \n\n";
    MultilayerPerceptron mlpRelu({2, 6, 1}, ActivationType::RELU, 1.0, 0.01); // ReLU needs more neurons & smaller learning rate
    trainXOR(mlpRelu, "ReLU", 10000); // Need more iterations for ReLU
    
    return 0;
}