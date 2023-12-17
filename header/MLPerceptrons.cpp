#include "MLPerceptrons.h"

double frand () {
    return (2.0 * (double)rand() / RAND_MAX) - 1.0;
}

//constructor for Perceptron class
Perceptron::Perceptron (size_t inputs, double bias) {
    this->bias = bias;
    weights.resize(inputs + 1); // +1 since we have bias as well
    //generate(weights.begin(), weights.end(), frand());
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


MultilayerPerceptron::MultilayerPerceptron (std::vector<size_t> layers, double bias, double eta) {
    this->layers = layers;
    this->bias = bias;
    this->eta = eta;

    // create neurons layer by layers
    for (size_t i = 0; i < layers.size(); i++) {
        // to store output values for each neuron in layer i, initilised to 0
        values.push_back(std::vector<double> (layers[i], 0.0) );
        network.push_back(std::vector<Perceptron> () ); // initially empty
        if (i > 0) { // network[0] is the input layers, so it has no neurons
            for (size_t j = 0; j < layers[i]; j++) {
                network[i].push_back(Perceptron(layers[i - 1], bias)); // create same number of neurons as the previous layer
            }
        }
    }
}

void MultilayerPerceptron::set_weights (std::vector< std::vector< std::vector<double> > > w_init) {
    for (size_t i = 0; i < w_init.size(); i++) { // to itr thorugh layers in the network
        for (size_t j = 0; j < w_init[i].size(); j++) { // to itr thorugh neurons in the layer
            network[i + 1][j].set_weights(w_init[i][j]); // i + 1 -> to skip input layer
        }
    }
}

void MultilayerPerceptron::printWeights () {
    std::cout << std::endl;
    for (size_t i = 0; i < network.size(); i++) {
        for (size_t j = 0; j < layers[i]; j++) {
            std::cout << "Layer " << i + 1 << " Neurons " << j << ": ";
            for (auto &itr: network[i][j].weights) {
                std::cout << itr << "   ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}


std::vector<double> MultilayerPerceptron::run (std::vector<double> x) {
    values[0] = x; // input
    for (size_t i = 1; i < network.size(); i++) { // excluding the input layer
        for (size_t j = 0; j < layers[i]; j++) {
            values[i][j] = network[i][j].run(values[i - 1]);
        }
    }
    return values.back();
}