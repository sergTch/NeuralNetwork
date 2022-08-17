#pragma once
#include <iostream>
#include <fstream>
#include "matrix.h"

namespace checkers {
	class position{
	public:
		uint32_t allies;
		uint32_t enemies;
		uint32_t kings;
		int turn;
		int whiteWins;
		int blackWins;
		int draws;

		void read(std::ifstream& fin);
		void vectorize(std::pair<matrix, matrix>& data);
	};
	void loadData(std::vector<std::pair<matrix, matrix>>& trainData, std::vector<std::pair<matrix, matrix>>& testData,
		std::vector<std::pair<matrix, matrix>*>& trainPointers, std::vector<std::pair<matrix, matrix>*>& testPointers,
		int amount, std::string path);
}
