#pragma once
#include "matrix.h"

class network
{
private:
	std::vector<matrix> activations;
	std::vector<matrix> sums;

	int batchSize = 10;
	double eta = 0.3;

	static double activate(double x);
	static double activDeriv(double x);
public:
	std::vector<matrix> biases;
	std::vector<matrix> weights;
	int size;

	network();
	network(std::vector<int>& layers);

	void removeNeuron(int i, int j);
	void load(std::string dir);
	void save(std::string dir);

	matrix& feed(matrix& input);
	double SGD(std::vector<std::pair<matrix,matrix>*>& data, std::vector<std::pair<matrix, matrix>*>& testData);
	void backprop(matrix& input, matrix& output, std::vector<matrix>& dbDeriv, std::vector<matrix>& dwDeriv);
	
	int evaluateClasification(std::pair<matrix, matrix>* test);
	int evaluateEstimation(std::pair<matrix, matrix>* test);
	
	void compare(network& ann);
	void print();
};

