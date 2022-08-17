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

int main(int argc, char** argv)
{
	std::srand(time(NULL));
	//if (argc == 1 || argv[1] == "train") {
	if (true) {
		std::vector<std::pair<matrix, matrix>> trainData(0), testData(0);
		std::vector<std::pair<matrix, matrix>*> trainPointers(0), testPointers(0);

		pictures::loadData(trainData, testData, trainPointers, testPointers, "dataset");
		//checkers::loadData(trainData, testData, trainPointers, testPointers, 20,"positions.txt");

		std::cout << trainData[0].first.m << " " << trainData[0].first.n << std::endl
			<< trainData[0].second.m << " " << trainData[0].second.n << std::endl;
		std::cout << "learning starts\n";

		std::vector<int> vec{ 196, 40, 40, 10 };
		network ann(vec);
		//ann.load("ANN98%");

		/*for (int i = 0; i < testData.size(); i++)
			if (!ann.evaluateClasification(testPointers[i])) {
				pictures::showImageTest(*testPointers[i]);
				matrix rez = ann.feed(testPointers[i]->first);
				for (int i = 0; i < rez.n; i++)
					std::cout << rez[0][i];
				std::cout << std::endl;
			}*/

		int k = 0;
		for (int i = 0; i < 1000; i++) {
			std::cout << i << std::endl;
			double rateTrain = ann.SGD(trainPointers, trainPointers);
			double rate = ann.SGD(trainPointers, testPointers);
			std::cout << rate << std::endl;
			if (rate > 0.98 && rateTrain > 0.99 && ann.biases[0].n > 1) {
				for (int k = 0; k < ann.size; k++)
					std::cout << ann.weights[k].m << " ";
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
		//input.load("input.txt");
		pictures::scaleImage(input.vals[0], 14);
		input.n = 196;
		matrix& rez = ann.feed(input);
		float max = 0;
		int t = 0;
		for (int i = 0; i < 10; i++) {
			if (rez[0][i] > max) {
				max = rez[0][i];
				t = i;
			}
			//std::cout << (int)(rez[0][i] * 1000) * 0.001 << " ";
		}
		std::cout << t << std::endl;
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