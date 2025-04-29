#include "header/MLPerceptrons.h"
#include <iostream>
#include <sstream>

std::vector<double> vectorReader() 
{
    std::vector<double> inputVec = {0, 1};
    std::string input;

    while (inputVec.size() != 7) 
    {
        inputVec = {};
        std::cout << "Input pattern \"a b c d e f g\": ";
        getline(std::cin, input);
        std::stringstream ss(input);
        double val;
        while ( ss >> val) {
            inputVec.push_back(val);
            if (ss.peek() == ' ') 
            {
                ss.ignore();
            }
        }
        if (inputVec[0] < 0.0)
        {
            break;
        }
        if (inputVec.size() != 7) 
        {
            std::cout << "Error: Input must contain exactly 7 floating point values separated by spaces." << std::endl;
        }
    }

    return inputVec;
}

int main () 
{
    srand(time(NULL));
    rand();

    int epochs;
    double MSE;
    ActivationType aType;
    int type;
    // Segment Display Recognition:
    // Recognize number from a seven-segment display
    std::cout << "-------------------------------- Segment Display Recognition System --------------------------------" << std::endl;
    std::cout << "How many epochs?: ";
    std::cin >> epochs;
    std::cin.ignore();

    std::cout << "Activation type? (0: SIGMOID, 1: TANH, 2: RELU): ";
    std::cin  >> type;
    std::cin.ignore();

    switch(type) 
    {
        case 0:
            std::cout << "SIGMOID selected" << std::endl;
            aType = ActivationType::SIGMOID;
            break;
        case 1:
        std::cout << "TANH selected" << std::endl;
            aType = ActivationType::TANH;
            break;
        case 2:
        std::cout << "RELU selected" << std::endl;
            aType = ActivationType::RELU;
            break;
        default:
            std::cout << "Invalid selection; defaulting to SIGMOID\n";
            aType = ActivationType::SIGMOID;
            break;
    }
    
    // 7 to 1
    MultilayerPerceptron sdr({7, 7, 1}, aType);                // 7 inout, 7 neurons, 1 hidden layer, 1 output layer
    for (int i = 0; i < epochs; i++) 
    {
        MSE = 0.0;
        MSE += sdr.backPropagation({1, 1, 1, 1, 1, 1, 0}, {0.05});  // 0 pattern
        MSE += sdr.backPropagation({0, 1, 1, 0, 0, 0, 0}, {0.15});  // 1 pattern
        MSE += sdr.backPropagation({1, 1, 0, 1, 1, 0, 1}, {0.25});  // 2 pattern
        MSE += sdr.backPropagation({1, 1, 1, 1, 0, 0, 1}, {0.35});  // 3 pattern
        MSE += sdr.backPropagation({0, 1, 1, 0, 0, 1, 1}, {0.45});  // 4 pattern
        MSE += sdr.backPropagation({1, 0, 1, 1, 0, 1, 1}, {0.55});  // 5 pattern
        MSE += sdr.backPropagation({1, 0, 1, 1, 1, 1, 1}, {0.65});  // 6 pattern
        MSE += sdr.backPropagation({1, 1, 1, 0, 0, 0, 0}, {0.75});  // 7 pattern
        MSE += sdr.backPropagation({1, 1, 1, 1, 1, 1, 1}, {0.85});  // 8 pattern
        MSE += sdr.backPropagation({1, 1, 1, 1, 0, 1, 1}, {0.95});  // 9 pattern
    }
    MSE /= 10.0;            // number of different ouputs
    std::cout << "7 to 1 Network MSE: " << MSE << std::endl;

    // 7 to 10
    MultilayerPerceptron sdrTen({7, 7, 10}, aType);                // 7 inout, 7 neurons, 1 hidden layer, 10 output layer
    for (int i = 0; i < epochs; i++) 
    {
        MSE = 0.0;
        MSE += sdrTen.backPropagation({1, 1, 1, 1, 1, 1, 0}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0});  // 0 pattern
        MSE += sdrTen.backPropagation({0, 1, 1, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0});  // 1 pattern
        MSE += sdrTen.backPropagation({1, 1, 0, 1, 1, 0, 1}, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});  // 2 pattern
        MSE += sdrTen.backPropagation({1, 1, 1, 1, 0, 0, 1}, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0});  // 3 pattern
        MSE += sdrTen.backPropagation({0, 1, 1, 0, 0, 1, 1}, {0, 0, 0, 0, 1, 0, 0, 0, 0, 0});  // 4 pattern
        MSE += sdrTen.backPropagation({1, 0, 1, 1, 0, 1, 1}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0});  // 5 pattern
        MSE += sdrTen.backPropagation({1, 0, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0});  // 6 pattern
        MSE += sdrTen.backPropagation({1, 1, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0});  // 7 pattern
        MSE += sdrTen.backPropagation({1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0});  // 8 pattern
        MSE += sdrTen.backPropagation({1, 1, 1, 1, 0, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1});  // 9 pattern
    }
    MSE /= 10.0;            // number of different ouputs
    std::cout << "7 to 10 Network MSE: " << MSE << std::endl;

    // 7 to 7 NN
    MultilayerPerceptron sdrS({7, 7, 7}, aType);                // 7 inout, 7 neurons, 1 hidden layer, 7 output layer
    for (int i = 0; i < epochs; i++) 
    {
        MSE = 0.0;
        MSE += sdrS.backPropagation({1, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1, 0});  // 0 pattern
        MSE += sdrS.backPropagation({0, 1, 1, 0, 0, 0, 0}, {0, 1, 1, 0, 0, 0, 0});  // 1 pattern
        MSE += sdrS.backPropagation({1, 1, 0, 1, 1, 0, 1}, {1, 1, 0, 1, 1, 0, 1});  // 2 pattern
        MSE += sdrS.backPropagation({1, 1, 1, 1, 0, 0, 1}, {1, 1, 1, 1, 0, 0, 1});  // 3 pattern
        MSE += sdrS.backPropagation({0, 1, 1, 0, 0, 1, 1}, {0, 1, 1, 0, 0, 1, 1});  // 4 pattern
        MSE += sdrS.backPropagation({1, 0, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 1, 1});  // 5 pattern
        MSE += sdrS.backPropagation({1, 0, 1, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1, 1});  // 6 pattern
        MSE += sdrS.backPropagation({1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0});  // 7 pattern
        MSE += sdrS.backPropagation({1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1});  // 8 pattern
        MSE += sdrS.backPropagation({1, 1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 1, 1});  // 9 pattern
    }
    MSE /= 10.0;            // number of different ouputs
    std::cout << "7 to 7 Network MSE: " << MSE << std::endl;

    // Classifier tester
    std::vector<double> inputPattern = {1.2};
    while(inputPattern[0] >= 0.0) 
    { 
        inputPattern = vectorReader();
        if (inputPattern[0] < 0.0) 
        {
            break;
        }

        std::cout << "The output for above sample by 7 to 1 Network is " << (int) (sdr.run(inputPattern)[0] * 10) << std::endl;

        auto numList = sdrTen.run(inputPattern);
        auto maxItr = max_element(numList.begin(), numList.end());
        auto maxIdx = distance(numList.begin(), maxItr);
        std::cout << "The output for above sample by 7 to 10 Network is " << maxIdx << std::endl;

        numList =  sdrS.run(inputPattern);
        std::cout << "The output for above sample by 7 to 7 Network is ["; 
        for (auto i : numList)
        {
            std::cout << " " << int(i + 0.5);
        }
        std::cout << " ]" << std::endl << std::endl;

    }

    return 0;
}