#include "header/MLPerceptrons.h"
#include <iostream>

using namespace std;

int main () {
    srand(time(NULL));
    rand();

    // Test code - Trained XOR
    cout << "\n\n ------------------------- Trained XOR Example using the backPropogation ------------------------- \n\n";
    MultilayerPerceptron mlpObj({2, 2, 1});
    cout << "Training the Neural Network as an XOR Gate..." << endl;

    double MSE;
    for (int i = 0; i < 2000; i++) {                    // number of training runs
        MSE = 0.0;
        MSE += mlpObj.backPropagation({0, 0}, {0});     // inputs and expected outputs
        MSE += mlpObj.backPropagation({0, 1}, {1});
        MSE += mlpObj.backPropagation({1, 0}, {1});
        MSE += mlpObj.backPropagation({1, 1}, {0});
        MSE /= 4.0;
        if (i % 100 == 0) {
            cout << "MSE = " << MSE << endl;            // MSE for evey 100th cycle
        }
    }
    cout << "\n\nTrained Weights (compared to hardcoded weights):\n";
    mlpObj.printWeights();

    cout << "XOR: " << endl;
    cout << "0 0 = " << mlpObj.run({0, 0})[0] << endl;
    cout << "0 1 = " << mlpObj.run({0, 1})[0] << endl;
    cout << "1 0 = " << mlpObj.run({1, 0})[0] << endl;
    cout << "1 1 = " << mlpObj.run({1, 1})[0] << endl;
    return 0;
}