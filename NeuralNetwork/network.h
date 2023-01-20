#pragma once
#include "matrix.h"
#include "layer.h"

class network
{
private:
	std::vector<matrix> activations;
	std::vector<matrix> sums;

	size_t curBatch;
	
	void connectLayerOuts();

public:
	std::vector<layer> layers;
	std::vector<layer> lderiv;
	int size;
	double eta = 0.3;

	network();
	network(std::vector<int>& layers);

	void addLayer(size_t input, size_t output, actType atype);
	void addLayer(size_t output, actType atype);

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

