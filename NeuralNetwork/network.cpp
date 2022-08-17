#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include "time.h"
#include "network.h"

double network::activate(double x) {
	double e = exp(-x);
	return 1 / (e + 1);
}

double network::activDeriv(double x) {
	double e = exp(-x);
	return e / (e + 1) / (e + 1);
}

int network::evaluateClasification(std::pair<matrix, matrix>* test) {
	matrix& output = feed(test->first);
	//output.print();
	//test->second.print();
	double max = 0;
	int t = 0;
	for (int i = 0; i < test->second[0].size(); i++) {
		if (output[0][i] > max) {
			max = output[0][i];
			t = i;
		}
	}
	if (test->second[0][t] == 1)
		return 1;
	return 0;
}

int network::evaluateEstimation(std::pair<matrix, matrix>* test) {
	matrix& output = feed(test->first);
	return (test->second[0][0] - 0.5) * (output[0][0] - 0.5) > 0;
}

network::network() {
	size = 0;
	activations = std::vector<matrix>();
	sums = std::vector<matrix>();
	biases = std::vector<matrix>();
	weights = std::vector<matrix>();
}

network::network(std::vector<int>& layersn) {
	size = layersn.size() - 1;
	activations = std::vector<matrix>(size);
	sums = std::vector<matrix>(size);
	biases = std::vector<matrix>(size);
	weights = std::vector<matrix>(size);

	for (int i = 0; i < size; i++){
		activations[i] = matrix(1, layersn[i + 1]);
		sums[i] = matrix(1, layersn[i + 1]);
		biases[i] = matrix(1, layersn[i + 1]);
		biases[i].rand();
		weights[i] = matrix(layersn[i], layersn[i + 1]);
		weights[i].rand();
	}
}

matrix& network::feed(matrix& input) {
	sums[0].multiply(input, weights[0]);
	sums[0].add(biases[0]);
	activations[0].foreach(sums[0], activate);

	for (int i = 1; i < size; i++) {
		sums[i].multiply(activations[i - 1], weights[i]);
		sums[i].add(biases[i]);
		activations[i].foreach(sums[i], activate);
	}

	return activations[size-1];
}

double network::SGD(std::vector<std::pair<matrix, matrix>*>& data, std::vector<std::pair<matrix, matrix>*>& testData) {
	//matrix buf;
	//std::string v;
	
	int rez = 0;
	for (int i = 0; i < testData.size(); i++) {
		rez += evaluateClasification(testData[i]);
	}
	std::cout << rez << "/" << testData.size() << " -> ";
	std::random_shuffle(std::begin(data), std::end(data));
	int n = data.size();
	std::vector<matrix> bDeriv(size);
	std::vector<matrix> dbDeriv(size);
	std::vector<matrix> wDeriv(size);
	std::vector<matrix> dwDeriv(size);

	for (int i = 0; i < size; i++) {
		bDeriv[i] = matrix(biases[i].m, biases[i].n);
		dbDeriv[i] = matrix(biases[i].m, biases[i].n);
		wDeriv[i] = matrix(weights[i].m, weights[i].n);
		dwDeriv[i] = matrix(weights[i].m, weights[i].n);
	}

	for (int i = 0; i < n; i += batchSize) {
		for (int t = 0; t < size; t++) {
			bDeriv[t].zero();
			wDeriv[t].zero();
		}
		for (int j = 0; j < batchSize && i + j < n; j++) {
			backprop(data[i + j]->first, data[i + j]->second, dbDeriv, dwDeriv);
			for (int t = 0; t < size; t++) {
				//v = std::to_string(i) + "_" + std::to_string(j + 1) + "_" + std::to_string(t) + "part.txt";
				//buf.load("grad\\biases\\" + v);
				//dbDeriv[t].compare(buf);
				//buf.load("grad\\weights\\" + v);
				//dwDeriv[t].compare(buf);
				bDeriv[t].add(dbDeriv[t]);
				wDeriv[t].add(dwDeriv[t]);
			}
		}
		for (int t = 0; t < size; t++) {
			//v = std::to_string(i) + "_" + std::to_string(t) + ".txt";
			//buf.load("grad\\biases\\" + v);
			//bDeriv[t].compare(buf);
			//buf.load("grad\\weights\\" + v);
			//wDeriv[t].compare(buf);
			bDeriv[t].multiply(eta / batchSize);
			wDeriv[t].multiply(eta / batchSize);
			biases[t].sub(bDeriv[t]);
			weights[t].sub(wDeriv[t]);
		}

		//std::vector<int> vec{ 784, 30, 10 };
		//network ANNrez = network(vec);
		//ANNrez.load("ANNrez" + std::to_string(i / banchSize + 1));
		//compare(ANNrez);
	}

	rez = 0;
	for (int i = 0; i < testData.size(); i++) {
		rez += evaluateClasification(testData[i]);
	}
	std::cout << rez << "/" << testData.size() << std::endl;
	return 1.0 * rez / testData.size();
}

void network::backprop(matrix& input, matrix& output, std::vector<matrix>& dbDeriv, std::vector<matrix>& dwDeriv) {
	feed(input);
	int t = size - 1;

	dbDeriv[t].sub(activations[t], output);
	sums[t].foreach(activDeriv);
	dbDeriv[t].foreachMult(sums[t]);
	t--;

	for (; t >= 0; t--) {
		for (int i = 0; i < weights[t + 1].m; i++)
			for (int j = 0; j < weights[t + 1].n; j++)
				dwDeriv[t + 1][i][j] = activations[t][0][i] * dbDeriv[t + 1][0][j];

		for (int i = 0; i < dbDeriv[t].n; i++) {
			dbDeriv[t][0][i] = 0;
			for (int j = 0; j < dbDeriv[t + 1].n; j++)
				dbDeriv[t][0][i] += weights[t + 1][i][j] * dbDeriv[t + 1][0][j];
		}
		sums[t].foreach(activDeriv);
		dbDeriv[t].foreachMult(sums[t]);
	}
	for (int i = 0; i < weights[t + 1].m; i++)
		for (int j = 0; j < weights[t + 1].n; j++)
			dwDeriv[0][i][j] = input[0][i] * dbDeriv[0][0][j];
}

void network::compare(network& ann) {
	for (int i = 0; i < size; i++) {
		weights[i].compare(ann.weights[i]);
		biases[i].compare(ann.biases[i]);
	}
}

void network::print() {
	for (int i = 0; i < size; i++) {
		biases[i].print();
		weights[i].print();
	}
}

void network::removeNeuron(int i, int j) {
	biases[i].removeCol(j);
	sums[i].removeCol(j);
	activations[i].removeCol(j);
	weights[i].removeCol(j);
	weights[i + 1].removeRow(j);
}

void network::load(std::string dir) {
	std::ifstream f(dir + "/data.txt");
	f >> size;
	f.close();
	activations = std::vector<matrix>(size);
	sums = std::vector<matrix>(size);
	biases = std::vector<matrix>(size);
	weights = std::vector<matrix>(size);

	for (int i = 0; i < size; i++) {
		biases[i].load(dir + "\\biases\\" + std::to_string(i) + ".txt");
		weights[i].load(dir + "\\weights\\" + std::to_string(i) + ".txt");
		activations[i] = matrix(1, biases[i].n);
		sums[i] = matrix(1, biases[i].n);
	}
}

void network::save(std::string dir) {
	system(("mkdir " + dir).c_str());
	system(("mkdir " + dir + "\\biases").c_str());
	system(("mkdir " + dir + "\\weights").c_str());
	std::ofstream f(dir + "\\data.txt");
	f << size;
	f.close();

	for (int i = 0; i < size; i++) {
		biases[i].save(dir + "/biases/" + std::to_string(i) + ".txt");
		weights[i].save(dir + "/weights/" + std::to_string(i) + ".txt");
	}
}