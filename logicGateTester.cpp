#include "header/MLPerceptrons.h"
#include <iostream>

using namespace std;

int main () {
    srand(time(NULL));
    rand();
    
    cout << "\n\n ------------ logic Gate Tester ------------ \n\n";
    // remember that the weights decide how the sigmoid function acts as, chaning it will change its behaviour
    // here the bias also act as crucial since its the only value that is not effect by the input value.
    Perceptron P(2);
    P.set_weights({10, 10, -15}); // AND 2 values and 1 bias
    cout << "AND Gate: " << endl;
    cout << P.run({0, 0}) << endl;
    cout << P.run({0, 1}) << endl;
    cout << P.run({1, 0}) << endl;
    cout << P.run({1, 1}) << endl;
    cout << "\n --------------------------- \n";

    //P.set_weights({10, 10, -5}); // OR 2 values and 1 bias
    //P.set_weights({-10, -10, 5}); // NOR 2 values and 1 bias
    //P.set_weights({-10, -10, 15}); // NAND 2 values and 1 bias

    // Multilayer Merceptrons
    cout << "\n\n ------------ logic Gate XOR using Multilayer Merceptrons ------------ \n\n";
    MultilayerPerceptron mLP({2, 2, 1});
    mLP.set_weights({{{-10, -10, 15}, {10, 10, -5}}, {{10, 10, -15}}}); // ({ {NAND, OR}, {AND} })
    cout << "Hardcoded weights are :" << endl;
    mLP.printWeights();

    cout << "XOR:" << endl;
    cout << "0 0 = " << mLP.run({0, 0})[0] << " Rounded Value: " << round(mLP.run({0, 0})[0]) << endl;
    cout << "0 1 = " << mLP.run({0, 1})[0] << " Rounded Value: " << round(mLP.run({0, 1})[0]) << endl;
    cout << "1 0 = " << mLP.run({1, 0})[0] << " Rounded Value: " << round(mLP.run({1, 0})[0]) << endl;
    cout << "1 1 = " << mLP.run({1, 1})[0] << " Rounded Value: " << round(mLP.run({1, 1})[0]) << endl;
    
    return 0;
}