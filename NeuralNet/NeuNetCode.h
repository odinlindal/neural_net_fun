#pragma once
#include <vector>
#include <string>

enum class ActivationType {
	TANH,
	RELU,
	SIGMOID,
	SOFTMAX
};

class Node {

public:
	Node(int numInputs, ActivationType type);

	float feedForward(std::vector<float> inputs);

	float getActivationDerivative();

	void updateWeights(std::vector<float> inputs, float learningRate);

	std::vector<float> weights;
	float bias;
	float output_cache;
	float delta;
	ActivationType actType;

};

class Layer {

public:
	Layer(int numNeurons, int numInputs, ActivationType type);
	std::vector<float> feedForward(std::vector<float> inputs);
	std::vector<Node> neurons;

};

class Network {

public:
	Network(std::vector<int> layerNeurons, int outputs, int inputs);
	void backPropagate(std::vector<float> inputs, std::vector<float> targets, float learningRate);
	void saveNetwork(std::string filename);
	bool loadNetwork(std::string filename);

	std::vector<Layer> layers;
	std::vector<float> feedForward(std::vector<float> inputs);
};