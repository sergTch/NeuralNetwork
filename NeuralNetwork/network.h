#pragma once
#include "matrix.h"
#include "layer.h"

class network
{
private:
	std::vector<matrix> activations;
	std::vector<matrix> sums;

	double eta = 0.3;
	size_t curBatch;

public:
	std::vector<layer> layers;
	std::vector<layer> lderiv;
	int size;

	network();
	network(std::vector<int>& layers);

	void removeNeuron(int i, int j);
	void load(std::string dir);
	void save(std::string dir);

	matrix& feed(matrix& input);
	double test(std::vector<std::pair<matrix, matrix>*>& testData);
	void SGD(std::vector<std::pair<matrix,matrix>*>& data, int batchSize=10);
	void backprop(matrix& input, matrix& output);
	void applyGrad();

	int evaluateClasification(std::pair<matrix, matrix>* test);
	int evaluateEstimation(std::pair<matrix, matrix>* test);
	
	void print();
};

