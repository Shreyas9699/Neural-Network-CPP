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

    P.set_weights({20, 20, -15}); // OR 2 values and 1 bias
    cout << "OR Gate: " << endl;
    cout << P.run({0, 0}) << endl;
    cout << P.run({0, 1}) << endl;
    cout << P.run({1, 0}) << endl;
    cout << P.run({1, 1}) << endl;
    cout << "\n --------------------------- \n";

    P.set_weights({-10, -10, 5}); // NOR 2 values and 1 bias
    cout << "NOR Gate: " << endl;
    cout << P.run({0, 0}) << endl;
    cout << P.run({0, 1}) << endl;
    cout << P.run({1, 0}) << endl;
    cout << P.run({1, 1}) << endl;
    cout << "\n --------------------------- \n";

    P.set_weights({-10, -10, 10}); // NAND 2 values and 1 bias
    cout << "NAND Gate: " << endl;
    cout << P.run({0, 0}) << endl;
    cout << P.run({0, 1}) << endl;
    cout << P.run({1, 0}) << endl;
    cout << P.run({1, 1}) << endl;
    cout << "\n --------------------------- \n";


    return 0;
}