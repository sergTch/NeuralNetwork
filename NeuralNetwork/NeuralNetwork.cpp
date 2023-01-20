// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include "time.h"
#include "matrix.h"
#include "network.h"
#include "mnist_loader.h"
#include "picture.h"
#include "checkers.h"
#include "timer.h"

double f(double x) {
	return x - 1;
}

std::shared_ptr<int> f(std::shared_ptr<int> p) {
	return p;
}

int main(int argc, char** argv)
{
	//std::srand(time(NULL));

	//if (argc == 1 || argv[1] == "train") {
	if (true) {
		double rate, rateTrain;
		std::vector<std::pair<matrix, matrix>> trainData(0), testData(0);
		std::vector<std::pair<matrix, matrix>*> trainPointers(0), testPointers(0);

		pictures::loadData(trainData, testData, trainPointers, testPointers, "dataset");
		//checkers::loadData(trainData, testData, trainPointers, testPointers, 20,"positions.txt");

		std::cout << trainData[0].first.m << " " << trainData[0].first.n << std::endl
			<< trainData[0].second.m << " " << trainData[0].second.n << std::endl;
		std::cout << "learning starts\n";

		std::vector<int> vec{ 196,40,40,10 };
		//network ann(vec);
		network ann;
		ann.addLayer(196, 40, reluType);
		ann.addLayer(40, sigmType);
		ann.addLayer(10, sigmType);
		ann.load("ANN");
		rate = ann.test(testPointers);
		std::cout << rate << std::endl;
		
		int k = 0;
		for (int i = 0; i < 1000; i++) {
			timer_restart();
			std::cout << i << std::endl;
			ann.SGD(trainPointers);
			ann.SGD(trainPointers);

			rateTrain = ann.test(testPointers);
			rate = ann.test(testPointers);

			std::cout << rate << std::endl;
			if (rate > 0.98 && rateTrain > 0.99 && ann.layers[0].B.n > 1) {
				for (int k = 0; k < ann.size; k++)
					std::cout << ann.layers[k].W.m << " ";
				std::cout << "\nRemoving neurons\n";
				ann.removeNeuron(0, 0);
				ann.removeNeuron(1, 0);
			}
			ann.save("ANN");
		}
	}
	else {
		matrix input;
		network ann;
		ann.load("ANN");
		input.load(std::string(argv[1]));
	}
}



//for (int i = 0; i < 28; i++) {
//	for (int j = 0; j < 28; j++) {
//		if (vec[i * 28 + j] > 0.2)
//			std::cout << "x ";
//		else std::cout << "  ";
//	}
//	std::cout << "\n";
//}