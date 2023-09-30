#pragma once
#include "matrix.h"
#include "layer.h"
#include <mutex>
#include <shared_mutex>

class network
{
private:
	mutable std::shared_mutex gmutex;

public:
	size_t curBatch = 1;
	size_t outn;

	std::vector<layer> layers;
	int size;

	network();
	network(const std::vector<int>& layersn);
	network(const network& net);

	void addLayer(size_t input, size_t output, actType atype);
	void addLayer(size_t output, actType atype);

	void load(const std::string& dir);
	void save(const std::string& dir) const;
	void zero();

	matrix feed(const matrix& input) const;
	matrix feed(const matrix& input, std::vector<matrix>& sums, std::vector<matrix>& activations) const;
	double test(std::vector<std::pair<matrix, matrix>>& testData) const;
	void SGD(const std::vector<std::pair<matrix,matrix>>& data, double eta = 0.1, int batchSize=10);
	void backprop(const matrix& input, const matrix& output, const network& net);
	void applyGrad(network& deriv, double eta = 0.1);

	int evaluateClasification(std::pair<matrix, matrix>& test) const;
	
	void print() const;
};

