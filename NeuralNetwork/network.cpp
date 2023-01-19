#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include "time.h"
#include "network.h"


int network::evaluateClasification(std::pair<matrix, matrix>* test) {
	matrix& output = feed(test->first);

	double max = 0;
	int t = 0;
	for (int i = 0; i < test->second.n; i++) {
		if (output.get(0, i) > max) {
			max = output.get(0,i);
			t = i;
		}
	}
	if (test->second.get(0, t) == 1)
		return 1;
	return 0;
}

int network::evaluateEstimation(std::pair<matrix, matrix>* test) {
	matrix& output = feed(test->first);
	return (test->second.get(0,0) - 0.5) * (output.get(0, 0) - 0.5) > 0;
}

network::network() {
	size = 0;
	activations = std::vector<matrix>();
	sums = std::vector<matrix>();
	layers = std::vector<layer>();
	lderiv = std::vector<layer>();
}

network::network(std::vector<int>& layersn) {
	size = layersn.size() - 1;
	activations = std::vector<matrix>(size);
	sums = std::vector<matrix>(size);
	layers = std::vector<layer>(size);

	for (int i = 0; i < size; i++){
		layers[i] = layer(layersn[i], layersn[i + 1], sigmType);
		layers[i].out = &activations[i];
		layers[i].sums = &sums[i];
		activations[i] = matrix(1, layersn[i + 1]);
		sums[i] = matrix(1, layersn[i + 1]);
	}

	lderiv = layers;
}

matrix& network::feed(matrix& input) {
	layers[0].feed(input);

	for (int i = 1; i < size; i++)
		layers[i].feed(*layers[i - 1].out);

	return activations.back();
}

double network::test(std::vector<std::pair<matrix, matrix>*>& testData)
{
	double rez = 0;
	for (int i = 0; i < testData.size(); i++)
		rez += evaluateClasification(testData[i]);
	return rez / testData.size();
}

void network::SGD(std::vector<std::pair<matrix, matrix>*>& data, int batchSize) {
	std::random_shuffle(std::begin(data), std::end(data));
	size_t n = data.size();
	
	for (int i = 0; i < size; i++)
		lderiv[i] = 0;

	for (int i = 0; i < n; i += batchSize) {
		for (int j = 0; j < batchSize && i + j < n; j++)
			backprop(data[i + j]->first, data[i + j]->second);
		applyGrad();
	}
}

void network::backprop(matrix& input, matrix& output) {
	curBatch++;
	feed(input);
	int t = size - 1;

	matrix dB, dW;
	
	dB = activations[t] - output;
	sums[t].apply(layers[t].activateD);
	dB.elemMult(sums[t]);
	lderiv[t].B += dB;
	t--;
	
	for (; t >= 0; t--) {
		dW = activations[t].T() * dB;
		lderiv[t + 1].W += dW;
		dB = dB * layers[t + 1].W.T();
		
		sums[t].apply(layers[t].activateD);
		dB.elemMult(sums[t]);
		lderiv[t].B += dB;
	}
	dW = input.T() * dB;
	lderiv[t + 1].W += dW;
}

void network::applyGrad()
{
	if (curBatch) {
		for (int t = 0; t < size; t++) {
			lderiv[t] *= eta / curBatch;
			layers[t] -= lderiv[t];
		}
	}
	curBatch = 0;
}

void network::print() {
	for (int i = 0; i < size; i++) {
		layers[i].B.print();
		layers[i].W.print();
	}
}

void network::removeNeuron(int i, int j) {
	layers[i].B.removeCol(j);
	sums[i].removeCol(j);
	activations[i].removeCol(j);
	layers[i].W.removeCol(j);
	layers[i].W.removeRow(j);
}

void network::load(std::string dir) {
	std::ifstream f(dir + "/data.txt");
	f >> size;
	activations = std::vector<matrix>(size);
	sums = std::vector<matrix>(size);
	layers = std::vector<layer>(size);
	for (int i = 0; i < size; i++) {
		size_t atype;
		f >> atype;
		layers[i].setAct(actType(atype));
	}
	f.close();

	for (int i = 0; i < size; i++) {
		layers[i].B.load(dir + "\\" + std::to_string(i) + "biases.txt");
		layers[i].W.load(dir + "\\" + std::to_string(i) + "weights.txt");
		activations[i] = matrix(1, layers[i].B.n);
		sums[i] = matrix(1, layers[i].B.n);
	}
}

void network::save(std::string dir) {
	system(("mkdir " + dir).c_str());
	std::ofstream f(dir + "\\data.txt");
	f << size << "\n";
	for (int i = 0; i < size; i++)
		f << size_t(layers[i].atype) << " ";
	f.close();

	for (int i = 0; i < size; i++) {
		layers[i].B.save(dir + "\\" + std::to_string(i) + "biases.txt");
		layers[i].W.save(dir + "\\" + std::to_string(i) + "weights.txt");
	}
}