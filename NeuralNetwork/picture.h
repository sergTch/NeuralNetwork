#pragma once

#include "matrix.h"
#include "mnist_loader.h"

namespace pictures {
	void showImageTest(std::pair<matrix, matrix>& image);
	double intersection(double a1, double a2, double b1, double b2);
	void scaleImage(std::vector<double>& image, int newSize);
	void cutCorners(std::vector<double>& image, int newSize);
	void prepareData(std::vector<std::pair<matrix, matrix>>& dataSet, mnist_loader& mnistData);
	void loadData(std::vector<std::pair<matrix, matrix>>& trainData, 
		std::vector<std::pair<matrix, matrix>>& testData, std::string path);
}