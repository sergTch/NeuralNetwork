// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <thread>
#include <functional>
#include "time.h"
#include "matrix.h"
#include "network.h"
#include "mnist_loader.h"
#include "picture.h"
#include "timer.h"


int main(int argc, char** argv)
{
	//std::srand(time(NULL));

	//if (argc == 1 || argv[1] == "train") {
	if (true) {
		double rate, rateTrain;
		std::vector<std::pair<matrix, matrix>> trainData(0), testData(0);
		pictures::loadData(trainData, testData, "dataset");

		std::cout << trainData[0].first.m << " " << trainData[0].first.n << std::endl
			<< trainData[0].second.m << " " << trainData[0].second.n << std::endl;
		std::cout << "learning starts\n";

		network ann;
		ann.addLayer(196, 40, sigmType);
		ann.addLayer(40, sigmType);
		ann.addLayer(10, sigmType);
		//ann.load("ANN");
		rate = ann.test(testData);
		std::cout << rate << std::endl;
		
		auto trainF = [&ann, &trainData]() {
			ann.SGD(trainData, 0.3, 20);
		};

		int k = 0;
		for (int i = 0; i < 1000; i++) {
			timer_restart();
			std::cout << i << std::endl;

			size_t tn = 1;
			std::vector<std::thread> threads(tn);
			for (int t = 0; t < tn; t++)
				threads[t] = std::thread(trainF);
			for (int t = 0; t < tn; t++)
				threads[t].join();

			rate = ann.test(testData);
			std::cout << rate << std::endl;
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