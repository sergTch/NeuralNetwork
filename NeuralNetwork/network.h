#pragma once
#include "matrix.h"
#include "layer.h"

class network
{
private:
public:
	size_t curBatch;
	size_t outn;

	std::vector<layer> layers;
	std::vector<layer> lderiv;
	int size;
	double eta = 0.3;

	network();
	network(const std::vector<int>& layersn);

	void addLayer(size_t input, size_t output, actType atype);
	void addLayer(size_t output, actType atype);

	void removeNeuron(int i, int j);
	void load(const std::string& dir);
	void save(const std::string& dir);

	matrix feed(const matrix& input);
	matrix feed(const matrix& input, std::vector<matrix>& sums, std::vector<matrix>& activations);
	double test(std::vector<std::pair<matrix, matrix>*>& testData);
	void SGD(std::vector<std::pair<matrix,matrix>*>& data, int batchSize=10);
	void backprop(const matrix& input, const matrix& output);
	void applyGrad();

	int evaluateClasification(std::pair<matrix, matrix>* test);
	int evaluateEstimation(std::pair<matrix, matrix>* test);
	
	void print();
};

