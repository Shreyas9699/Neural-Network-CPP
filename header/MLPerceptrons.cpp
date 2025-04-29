#include "MLPerceptrons.h"

double frand() 
{
    return (2.0 * (double)rand() / RAND_MAX) - 1.0;
}

// Activation and derivative functions
double sigmoid(double x)    { return 1.0 / (1.0 + std::exp(-x)); }
double dsigmoid(double x)   { return sigmoid(x) * (1 - sigmoid(x)); }

double tanh_act(double x)    { return std::tanh(x); }
double dtanh_act(double x)   { return 1 - std::tanh(x) * std::tanh(x); }

double relu(double x)        { return x > 0 ? x : 0; }
double drelu(double x)       { return x > 0 ? 1 : 0; }

double step_fn(double x)     { return x >= 0 ? 1 : 0; }
double dstep_fn(double x)    { return 0; /* not differentiable at 0, rarely used in backprop */ }

Activation getActivation(ActivationType t) 
{
    switch(t) 
    {
      case ActivationType::TANH:    return {tanh_act,   dtanh_act};
      case ActivationType::RELU:    return {relu,       drelu};
      case ActivationType::STEP:    return {step_fn,    dstep_fn};
      default:                      return {sigmoid,    dsigmoid};
    }
}

//constructor for Perceptron class
Perceptron::Perceptron(size_t inputs, ActivationType type, double bias)
    : bias(bias), aType(type)
{
    weights.resize(inputs + 1); // +1 since we have bias as well
    generate(weights.begin(), weights.end(), frand);
    auto act = getActivation(type);
    activate = act.fn;
    derivative = act.dfn;
}

double Perceptron::run(std::vector<double> x) 
{
    x.push_back(bias); // push bias into the vector
    return activate(inner_product(x.begin(), x.end(), weights.begin(), (double)0.0));
}

void Perceptron::set_weights(std::vector<double> w_init) 
{
    weights = w_init;
}

MultilayerPerceptron::MultilayerPerceptron(std::vector<size_t> layers, ActivationType type, double bias, double eta) 
{
    this->layers = layers;
    this->bias = bias;
    this->eta = eta;
    this->actType = type;

    // create neurons layer by layers
    for (size_t i = 0; i < layers.size(); i++) {
        // to store output values for each neuron in layer i, initilised to 0
        values.push_back(std::vector<double>(layers[i], 0.0));
        d.push_back(std::vector<double>(layers[i], 0.0));
        // Store the raw inputs to each neuron (before activation)
        raw_values.push_back(std::vector<double>(layers[i], 0.0));
        network.push_back(std::vector<Perceptron>());
        if (i > 0) { // network[0] is the input layer, so it has no neurons
            for (size_t j = 0; j < layers[i]; j++) {
                network[i].push_back(Perceptron(layers[i - 1], type, bias)); // Pass activation type to perceptron
            }
        }
    }
}

void MultilayerPerceptron::set_weights(std::vector<std::vector<std::vector<double>>> w_init) 
{
    for (size_t i = 0; i < w_init.size(); i++) { 
        for (size_t j = 0; j < w_init[i].size(); j++) { 
            network[i + 1][j].set_weights(w_init[i][j]); 
        }
    }
}

void MultilayerPerceptron::printWeights() 
{
    std::cout << std::endl;
    for (size_t i = 1; i < network.size(); i++) {
        for (size_t j = 0; j < layers[i]; j++) {
            std::cout << "Layer " << i << " Neuron " << j << ": ";
            for (auto &itr: network[i][j].weights) {
                std::cout << itr << "   ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

std::vector<double> MultilayerPerceptron::run(std::vector<double> x) 
{
    values[0] = x; // input
    
    for (size_t i = 1; i < network.size(); i++) { 
        for (size_t j = 0; j < layers[i]; j++) {
            // Calculate the raw input to the neuron
            std::vector<double> inputs = values[i-1];
            inputs.push_back(bias);
            double raw_input = inner_product(inputs.begin(), inputs.end(), 
                                            network[i][j].weights.begin(), 0.0);
            
            // Store the raw input for backpropagation
            raw_values[i][j] = raw_input;
            
            // Calculate the activation output
            values[i][j] = network[i][j].activate(raw_input);
        }
    }
    return values.back();
}

// Improved Backpropagation
double MultilayerPerceptron::backPropagation(std::vector<double> x, std::vector<double> y) 
{
    // STEP 1: Feed a sample to the network
    std::vector<double> output = run(x);
    
    // STEP 2: Calculate the MSE
    double MSE = 0.0;
    std::vector<double> error;
    for (size_t i = 0; i < y.size(); i++) {
        error.push_back(y[i] - output[i]);
        MSE += error[i] * error[i];
    }
    MSE /= y.size();
    
    // STEP 3: Calculate the output error terms
    for (size_t i = 0; i < output.size(); i++) {
        // Use the derivative of the actual activation function
        // Pass the RAW input to the derivative function
        d.back()[i] = network.back()[i].derivative(raw_values.back()[i]) * error[i];
    }
    
    // STEP 4: Calculate the error term of each unit on each layer (backpropagate)
    for (int i = network.size() - 2; i > 0; i--) {
        for (size_t j = 0; j < network[i].size(); j++) {
            double fwdErr = 0.0;
            for (size_t k = 0; k < layers[i + 1]; k++) {
                fwdErr += network[i + 1][k].weights[j] * d[i + 1][k];
            }
            // Use the neuron's actual derivative function with the raw input
            d[i][j] = network[i][j].derivative(raw_values[i][j]) * fwdErr;
        }
    }

    // STEPS 5 & 6: Calculate the deltas and update the weights
    for (size_t i = 1; i < network.size(); i++) {
        for (size_t j = 0; j < layers[i]; j++) {
            for (size_t k = 0; k < layers[i - 1] + 1; k++) {
                double delta;
                if (k == layers[i - 1]) {
                    delta = eta * d[i][j] * bias;
                } else {
                    delta = eta * d[i][j] * values[i - 1][k];
                }
                network[i][j].weights[k] += delta;
            }
        }
    }

    return MSE;
}