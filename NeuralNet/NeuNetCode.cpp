#include "NeuNetCode.h"
#include <vector>
#include <cstdlib>
#include <numeric>
#include <iostream>
#include <fstream>

Node::Node(int numInputs, ActivationType type) {

    for (int i = 0; i < numInputs; i++) {
        float randomWeight = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        weights.push_back(randomWeight);
    }

    //bias = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    bias = 0.1f;

    output_cache = 0.0f;
    delta = 0.0f;
    this->actType = type;
}

float Node::feedForward(std::vector<float> inputs) {
    float sum = 0.0f;
    if (inputs.size() != weights.size()) {
        std::cout << "error in inputs/weights size" << std::endl;
        return 0.0f;
    }
    for (int i = 0; i < weights.size(); i++) {
        sum += inputs[i] * weights[i];
    }

    sum += bias;

    if (this->actType == ActivationType::TANH) {
        this->output_cache = tanh(sum);
    }
    else if (this->actType == ActivationType::RELU) {
        this->output_cache = (sum > 0) ? sum : 0.0f;
    }
    else if (this->actType == ActivationType::SIGMOID) {
        this->output_cache = 1.0f / (1.0f + exp(-sum));
    }
    else if (this->actType == ActivationType::SOFTMAX) {
        // Pass the raw sum back to the Layer. 
        // The Layer will handle the exp() and division.
        this->output_cache = sum;
    }

    return this->output_cache;
}

float Node::getActivationDerivative() {
    if (this->actType == ActivationType::TANH) {
        return 1.0f - (output_cache * output_cache);
    }
    else if (this->actType == ActivationType::RELU) {
        return (output_cache > 0) ? 1.0f : 0.0f;
    }
    else if (this->actType == ActivationType::SIGMOID) {
        return output_cache * (1.0f - output_cache);
    }
    return 0.0f;
}

void Node::updateWeights(std::vector<float> inputs, float learningRate) {
    for (int i = 0; i < weights.size(); i++) {
        weights[i] += learningRate * (delta * inputs[i]); // multiply learning rate by gradient
    }
    bias += learningRate * delta;
}

Layer::Layer(int numNeurons, int numInputs, ActivationType type) {
    for (int i = 0; i < numNeurons; i++) {
        neurons.push_back(Node(numInputs, type));
    }
}

std::vector<float >Layer::feedForward(std::vector<float> inputs) {
    std::vector<float> outputs;

    for (int i = 0; i < neurons.size(); i++) {
        outputs.push_back(neurons[i].feedForward(inputs));
    }

    if (neurons.size() > 0 && neurons[0].actType == ActivationType::SOFTMAX) {
        float sumExp = 0.0f;

        for (int i = 0; i < outputs.size(); i++) {
            outputs[i] = exp(outputs[i]); // e^x
            sumExp += outputs[i];
        }

        for (int i = 0; i < outputs.size(); i++) {
            outputs[i] /= sumExp;
            neurons[i].output_cache = outputs[i];
        }
    }

    return outputs;
}

Network::Network(std::vector<int> layerNeurons, int outputs, int inputs) {
    if (layerNeurons.empty() || outputs <= 0 || inputs <= 0) return;
    layers.push_back(Layer(layerNeurons[0], inputs, ActivationType::RELU));

    for (int i = 1; i < layerNeurons.size(); i++) {
        layers.push_back(Layer(layerNeurons[i], layers.back().neurons.size(), ActivationType::RELU));
    }

    layers.push_back(Layer(outputs, layers.back().neurons.size(), ActivationType::SOFTMAX));

}

std::vector<float> Network::feedForward(std::vector<float> inputs) {
    std::vector<float> currentInputs = inputs;
    for (int i = 0; i < layers.size(); i++) {
        currentInputs = layers[i].feedForward(currentInputs);
    }
    return currentInputs;
}

void Network::backPropagate(std::vector<float> inputs, std::vector<float> targets, float learningRate) {
    std::vector<std::vector<float>> layerInputs;
    std::vector<float> currentInputs = inputs;
    layerInputs.push_back(currentInputs);

    for (int i = 0; i < layers.size(); i++) {
        currentInputs = layers[i].feedForward(currentInputs);
        layerInputs.push_back(currentInputs);
    }

    Layer& outputLayer = layers.back();

    for (int i = 0; i < outputLayer.neurons.size(); i++) {
        float output = outputLayer.neurons[i].output_cache;
        float target = targets[i];

        if (outputLayer.neurons[i].actType == ActivationType::SOFTMAX) {
            outputLayer.neurons[i].delta = target - output;
        } else {
            float error = target - output;

            outputLayer.neurons[i].delta = error * outputLayer.neurons[i].getActivationDerivative();
        }
    }

    for (int i = layers.size() - 2; i >= 0; i--) {
        Layer& curLayer = layers[i];
        Layer& nextLayer = layers[i + 1];

        for (int j = 0; j < curLayer.neurons.size(); j++) {
            float errorSum = 0.0f;

            for (int k = 0; k < nextLayer.neurons.size(); k++) {
                float weight = nextLayer.neurons[k].weights[j];
                float delta = nextLayer.neurons[k].delta;

                errorSum += weight * delta;
            }
            curLayer.neurons[j].delta = errorSum * curLayer.neurons[j].getActivationDerivative();
        }
    }

    for (int i = 0; i < layers.size(); i++) {
        std::vector<float> inputsForThisLayer = layerInputs[i];

        for (int j = 0; j < layers[i].neurons.size(); j++) {
            layers[i].neurons[j].updateWeights(inputsForThisLayer, learningRate);
        }
    }

}

void Network::saveNetwork(std::string filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: Could not save to file " << filename << std::endl;
        return;
    }

    // Loop through every layer, every neuron
    for (Layer& layer : layers) {
        for (Node& neuron : layer.neurons) {
            // Write Bias
            file << neuron.bias << "\n";

            // Write all Weights
            for (float w : neuron.weights) {
                file << w << "\n";
            }
        }
    }

    file.close();
    std::cout << "Network saved to " << filename << std::endl;
}

bool Network::loadNetwork(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false; // File doesn't exist yet

    // IMPORTANT: This assumes the Network structure (784->30->10) 
    // is EXACTLY the same as when you saved it.
    for (Layer& layer : layers) {
        for (Node& neuron : layer.neurons) {
            // Read Bias
            file >> neuron.bias;

            // Read Weights
            for (int i = 0; i < neuron.weights.size(); i++) {
                file >> neuron.weights[i];
            }
        }
    }

    file.close();
    std::cout << "Network loaded from " << filename << "!" << std::endl;
    return true;
}