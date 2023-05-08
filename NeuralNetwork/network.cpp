#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include "time.h"
#include "network.h"

int network::evaluateClasification(std::pair<matrix, matrix>& test) const {
	matrix output = feed(test.first);

	double max = 0;
	int t = 0;
	for (int i = 0; i < test.second.n; i++) {
		if (output.get(0, i) > max) {
			max = output.get(0,i);
			t = i;
		}
	}
	if (test.second.get(0, t) == 1)
		return 1;
	return 0;
}

network::network() {
	size = 0;
	layers = std::vector<layer>();
	outn = 0;
}

network::network(const std::vector<int>& layersn) {
	size = layersn.size() - 1;
	layers = std::vector<layer>(size);

	for (int i = 0; i < size; i++){
		layers[i] = layer(layersn[i], layersn[i + 1], sigmType);
	}

	outn = layersn.back();
}

network::network(const network& net) : layers(net.layers), size(net.size), outn(net.outn) {}

void network::addLayer(size_t input, size_t output, actType atype)
{
	size++;
	if (layers.size())
		addLayer(output, atype);
	else
		layers.push_back(layer(input, output, atype));

	outn = output;
}

void network::addLayer(size_t output, actType atype)
{
	size++;
	layers.push_back(layer(outn, output, atype));
	outn = output;
}

matrix network::feed(const matrix& input) const {
	matrix rez = layers[0].feed(input);

	for (int i = 1; i < size; i++)
		rez = layers[i].feed(rez);

	return rez;
}

matrix network::feed(const matrix& input, std::vector<matrix>& sums, std::vector<matrix>& activations) const
{
	layers[0].feed(input, sums[0], activations[0]);

	for (int i = 1; i < size; i++)
		layers[i].feed(activations[i - 1], sums[i], activations[i]);

	return activations.back();
}

double network::test(std::vector<std::pair<matrix, matrix>>& testData) const
{
	double rez = 0;
	for (int i = 0; i < testData.size(); i++)
		rez += evaluateClasification(testData[i]);
	return rez / testData.size();
}

void network::SGD(const std::vector<std::pair<matrix, matrix>>& data, double eta, int batchSize) {
	std::vector<const std::pair<matrix, matrix>*> pointers(data.size());
	for (int i = 0; i < data.size(); i++)
		pointers[i] = &data[i];
	std::random_shuffle(std::begin(pointers), std::end(pointers));
	
	network deriv = *this;
	deriv.zero();
	size_t n = data.size();

	for (int i = 0; i < n; i += batchSize) {
		for (int j = 0; j < batchSize && i + j < n; j++)
			deriv.backprop(pointers[i + j]->first, pointers[i + j]->second, *this);

		applyGrad(deriv, eta);
	}
}

void network::backprop(const matrix& input, const matrix& output, const network& net) {
	curBatch++;

	std::vector<matrix> sums(size), activations(size);
	net.feed(input, sums, activations);
	int t = size - 1;

	matrix dB, dW;

	dB = activations[t] - output;
	sums[t].apply(net.layers[t].activateD);
	dB.elemMult(sums[t]);
	layers[t].B += dB;
	t--;

	for (; t >= 0; t--) {
		dW = activations[t].T() * dB;
		layers[t + 1].W += dW;
		dB = dB * net.layers[t + 1].W.T();

		sums[t].apply(net.layers[t].activateD);
		dB.elemMult(sums[t]);
		layers[t].B += dB;
	}
	dW = input.T() * dB;
	layers[t + 1].W += dW;
}

void network::applyGrad(network& deriv, double eta)
{
	if (deriv.curBatch) {
		gmutex.lock();
		for (int t = 0; t < size; t++) {
			deriv.layers[t] *= eta / deriv.curBatch;
			layers[t] -= deriv.layers[t];
		}
		gmutex.unlock();
	}
	deriv.curBatch = 1;
}

void network::print() const {
	for (int i = 0; i < size; i++) {
		layers[i].B.print();
		layers[i].W.print();
	}
}

void network::load(const std::string& dir) {
	std::ifstream f(dir + "/data.txt");
	f >> size;
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
	}
}

struct stat info;
void network::save(const std::string& dir) const {
	if (stat(dir.c_str(), &info) != 0)
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

void network::zero()
{
	for (int i = 0; i < size; i++)
		layers[i] = 0;
}
