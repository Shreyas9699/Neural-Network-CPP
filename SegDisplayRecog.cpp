#include "header/MLPerceptrons.h"
#include <iostream>
#include <sstream>

using namespace std;

vector<double> vectorReader() {
    vector<double> inputVec = {0, 1};
    string input;

    while (inputVec.size() != 7) {
        inputVec = {};
        cout << "Input pattern \"a b c d e f g\": ";
        getline(cin, input);
        stringstream ss(input);
        double val;
        while ( ss >> val) {
            inputVec.push_back(val);
            if (ss.peek() == ' ') {
                ss.ignore();
            }
        }
        if (inputVec[0] < 0.0){
            break;
        }
        if (inputVec.size() != 7) {
            cout << "Error: Input must contain exactly 7 floating point values separated by spaces." << endl;
        }
    }

    return inputVec;
}

int main () {
    srand(time(NULL));
    rand();

    int epochs;
    double MSE;
    // Segment Display Recognition:
    // Recognize number from a seven-segment display
    cout << "-------------------------------- Segment Display Recognition System --------------------------------" << endl;
    cout << "How many epochs?: ";
    cin >> epochs;
    cin.ignore();
    
    // 7 to 1 MM
    MultilayerPerceptron sdr({7, 7, 1});                // 7 inout, 7 neurons, 1 hidden layer, 1 output layer
    for (int i = 0; i < epochs; i++) {
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
    cout << "7 to 1 Network MSE: " << MSE << endl;

    // 7 to 10 NN
    MultilayerPerceptron sdrTen({7, 7, 10});                // 7 inout, 7 neurons, 1 hidden layer, 10 output layer
    for (int i = 0; i < epochs; i++) {
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
    cout << "7 to 10 Network MSE: " << MSE << endl;

    // 7 to 7 NN
    MultilayerPerceptron sdrS({7, 7, 7});                // 7 inout, 7 neurons, 1 hidden layer, 7 output layer
    for (int i = 0; i < epochs; i++) {
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
    cout << "7 to 10 Network MSE: " << MSE << endl;

    // Classifier tester
    vector<double> inputPattern = {1.2};
    while(inputPattern[0] >= 0.0) { 
        inputPattern = vectorReader();
        if (inputPattern[0] < 0.0) {
            break;
        }

        cout << "The output for above sample by 7 to 1 Network is " << (int) (sdr.run(inputPattern)[0] * 10) << endl;

        auto numList = sdrTen.run(inputPattern);
        auto maxItr = max_element(numList.begin(), numList.end());
        auto maxIdx = distance(numList.begin(), maxItr);
        cout << "The output for above sample by 7 to 10 Network is " << maxIdx << endl;

        numList =  sdrS.run(inputPattern);
        cout << "The output for above sample by 7 to 10 Network is ["; 
        for (auto i : numList){
            cout << " " << int(i + 0.5);
        }
        cout << " ]" << endl << endl;

    }

    return 0;
}