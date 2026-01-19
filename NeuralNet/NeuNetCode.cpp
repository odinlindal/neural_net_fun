#include "NeuNetCode.h"
#include <vector>
#include <cstdlib>
#include <numeric>
#include <iostream>

Node::Node(int numInputs) {

    for (int i = 0; i < numInputs; i++) {
        float randomWeight = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        weights.push_back(randomWeight);
    }

    bias = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    output_cache = 0.0f;
    delta = 0.0f;
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

    // Tanh function activation method
    this->output_cache = tanh(sum);

    return this->output_cache;
}

float Node::getActivationDerivative() {
    // Derivative of sigmoid: f(x) * (1 - f(x))
    //return output_cache * (1.0f - output_cache);

    // Derivative of tanh = 1 - output^2
    return 1 - (output_cache * output_cache);
}

void Node::updateWeights(std::vector<float> inputs, float learningRate) {
    for (int i = 0; i < weights.size(); i++) {
        weights[i] += learningRate * (delta * inputs[i]); // multiply learning rate by gradient
    }
    bias += learningRate * delta;
}

Layer::Layer(int numNeurons, int numInputs) {
    for (int i = 0; i < numNeurons; i++) {
        neurons.push_back(Node(numInputs));
    }
}

std::vector<float >Layer::feedForward(std::vector<float> inputs) {
    std::vector<float> outputs;

    for (int i = 0; i < neurons.size(); i++) {
        outputs.push_back(neurons[i].feedForward(inputs));
    }
    return outputs;
}

Network::Network(std::vector<int> layerNeurons, int outputs, int inputs) {
    if (layerNeurons.empty() || outputs <= 0 || inputs <= 0) return;
    layers.push_back(Layer(layerNeurons[0], inputs));

    for (int i = 1; i < layerNeurons.size(); i++) {
        layers.push_back(Layer(layerNeurons[i], layers[i - 1].neurons.size()));
    }

    layers.push_back(Layer(outputs, layers.back().neurons.size()));

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

        float error = target - output;

        outputLayer.neurons[i].delta = error * outputLayer.neurons[i].getActivationDerivative();
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