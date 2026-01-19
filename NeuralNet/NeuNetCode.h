#pragma once
#include <vector>

class Node {

public:
	Node(int numInputs);

	float feedForward(std::vector<float> inputs);

	float getActivationDerivative();

	void updateWeights(std::vector<float> inputs, float learningRate);

	std::vector<float> weights;
	float bias;
	float output_cache;
	float delta;

};

class Layer {

public:
	Layer(int numNeurons, int numInputs);
	std::vector<float> feedForward(std::vector<float> inputs);
	std::vector<Node> neurons;

};

class Network {

public:
	Network(std::vector<int> layerNeurons, int outputs, int inputs);
	void backPropagate(std::vector<float> inputs, std::vector<float> targets, float learningRate);
	std::vector<Layer> layers;
	std::vector<float> feedForward(std::vector<float> inputs);
};