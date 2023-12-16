#include "MLPerceptrons.h"

double frand () {
    return (2.0 * (double)rand() / RAND_MAX) - 1.0;
}

//constructor for Perceptron class
Perceptron::Perceptron (size_t inputs, double bias) {
    this->bias = bias;
    weights.resize(inputs + 1); // +1 since we have bias as well
    generate(weights.begin(), weights.end(), frand());
}

double Perceptron::run (std::vector<double> x) {
    x.push_back(bias); // push bias into the vector
    return sigmoid(inner_product(x.begin(), x.end(), weights.begin(), (double)0.0));
}

void Perceptron::set_weights (std::vector<double> w_init) {
    weights = w_init;
}

double Perceptron::sigmoid (double x) {
    return 1.0 / ( 1.0 + std::exp(-x));
}
