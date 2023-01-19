#include "picture.h"
#include <iostream>
#include <string>
#include <algorithm>

namespace pictures {
	void showImageTest(std::pair<matrix, matrix>& image) {
		int size = sqrt(image.first.n);
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (image.first.get(0, i * size + j) > 0.3)
					std::cout << "o ";
				else std::cout << "  ";
			}
			std::cout << std::endl;
		}
		for (int i = 0; i < 10; i++) {
			if (image.second.get(0, i) == 1)
				std::cout << i << std::endl;
		}
	}

	double intersection(double a1, double a2, double b1, double b2) {
		double x = std::max(a1, b1), y = std::min(a2, b2);

		if (x < y)
			return y - x;
		return 0;
	}

	void scaleImage(std::vector<double>& image, int newSize) {
		int size = (int)(sqrt(image.size()) + 0.1);
		int left = size, right = 0, up = size, down = 0;

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (image[i * size + j] > 0) {
					if (left > j)
						left = j;
					if (right < j)
						right = j;
					if (up > i)
						up = i;
					if (down < i)
						down = i;
				}
			}
		}

		std::vector<double> newImage = std::vector<double>(newSize * newSize);

		for (int i = 0; i < newSize; i++) {
			for (int j = 0; j < newSize; j++) {
				newImage[i * newSize + j] = 0;
				for (int a = up + (double)i / newSize * (down - up); a < up + (double)(i + 1) / newSize * (down - up); a++) {
					for (int b = left + (double)j / newSize * (right - left); b < left + (double)(j + 1) / newSize * (right - left); b++) {
						double area = intersection((double)(a - up) / (down - up) * newSize, (double)(a - up + 1) / (down - up) * newSize, i, i + 1) *
							intersection((double)(b - left) / (right - left) * newSize, (double)(b - left + 1) / (right - left) * newSize, j, j + 1);
						newImage[i * newSize + j] += image[a * size + b] * area;
					}
				}
			}
		}

		image = newImage;
	}

	void cutCorners(std::vector<double>& image, int newSize) {
		int size = (int)(sqrt(image.size()) + 0.1);
		int left = size, right = 0, up = size, down = 0;

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (image[i * size + j] > 0) {
					if (left > j)
						left = j;
					if (right < j)
						right = j;
					if (up > i)
						up = i;
					if (down < i)
						down = i;
				}
			}
		}

		double x = 0; double y = 0;
		double mass = 0;
		int box = std::max(right - left, down - up) + 1;

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (image[i * size + j] > 0) {
					x = i + (x - i) * mass / (mass + image[i * size + j]);
					y = j + (y - j) * mass / (mass + image[i * size + j]);
					mass += image[i * size + j];
				}
			}
		}

		int newLeft = y + 0.5 - box * 0.5;
		int newRight = newLeft + box - 1;
		int newUp = x + 0.5 - box * 0.5;
		int newDown = newUp + box - 1;

		if (newLeft > left) {
			newLeft = left;
			newRight = newLeft + box - 1;
		}
		if (newRight < right) {
			newRight = right;
			newLeft = newRight - box + 1;
		}
		if (newUp > up) {
			newUp = up;
			newDown = newUp + box - 1;
		}
		if (newDown < down) {
			newDown = down;
			newUp = newDown - box + 1;
		}

		left = newLeft;
		right = newRight;
		up = newUp;
		down = newDown;

		std::vector<double> newImage = std::vector<double>(newSize * newSize);

		for (int i = 0; i < newSize; i++) {
			for (int j = 0; j < newSize; j++) {
				newImage[i * newSize + j] = 0;
				for (int a = up + (double)i / newSize * (down - up); a < up + (double)(i + 1) / newSize * (down - up); a++) {
					for (int b = left + (double)j / newSize * (right - left); b < left + (double)(j + 1) / newSize * (right - left); b++) {
						double area = intersection((double)(a - up) / (down - up) * newSize, (double)(a - up + 1) / (down - up) * newSize, i, i + 1) *
							intersection((double)(b - left) / (right - left) * newSize, (double)(b - left + 1) / (right - left) * newSize, j, j + 1);
						if (a >= 0 && a < size && b >= 0 && b < size)
							newImage[i * newSize + j] += image[a * size + b] * area;
					}
				}
			}
		}

		image = newImage;
	}

	void prepareData(std::vector<std::pair<matrix, matrix>>& dataSet,
		std::vector<std::pair<matrix, matrix>*>& pointersSet, mnist_loader& mnistData)
	{
		dataSet = std::vector<std::pair<matrix, matrix>>(mnistData.size());
		pointersSet = std::vector<std::pair<matrix, matrix>*>(mnistData.size());
		for (int t = 0; t < mnistData.size(); t++) {
			//trainData[t].first = matrix(train.images(t));
			dataSet[t].first = matrix(1, 196);
			std::vector<double> image = mnistData.images(t);
			scaleImage(image, 14);
			dataSet[t].first = matrix(image);
			/*for (int i = 0; i < 14; i++)
				std::cout << "__";
			std::cout << "|\n|";
			for (int i = 0; i < 14; i++) {
				for (int j = 0; j < 14; j++) {
					if (image[i * 14 + j] > 0.3)
						std::cout << "o ";
					else std::cout << "  ";
				}
				std::cout << "|\n|";
			}
			for (int i = 0; i < 14; i++)
				std::cout << "__";
			std::cout << "|\n";*/
			int l = mnistData.labels(t);
			dataSet[t].second = matrix(1, 10);
			//std::cout << l << std::endl;
			dataSet[t].second.get(0, l) = 1;
			pointersSet[t] = &dataSet[t];
		}
	}

	void loadData(std::vector<std::pair<matrix, matrix>>& trainData, std::vector<std::pair<matrix, matrix>>& testData,
		std::vector<std::pair<matrix, matrix>*>& trainPointers, std::vector<std::pair<matrix, matrix>*>& testPointers,
		std::string path) {
		mnist_loader train(path + "/train-images-idx3-ubyte",
			"dataset/train-labels-idx1-ubyte");
		mnist_loader test(path + "/t10k-images-idx3-ubyte",
			"dataset/t10k-labels-idx1-ubyte");
		pictures::prepareData(trainData, trainPointers, train);
		pictures::prepareData(testData, testPointers, test);
	}
}