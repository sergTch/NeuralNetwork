#include "checkers.h"

namespace checkers {
	uint32_t reverse(uint32_t k) {
		uint32_t a = 0;
		for (int i = 0; i < 32; i++) {
			if (k & (1 << i))
				a |= (1 << (31 - i));
		}
		return a;
	}

	void position::read(std::ifstream& fin) {
		fin >> allies;
		fin >> enemies;
		fin >> kings;
		fin >> turn;
		fin >> whiteWins >> blackWins >> draws;
		if (turn < 0) {
			turn = 1;
			allies = reverse(allies);
			enemies = reverse(enemies);
			kings = reverse(kings);
			int t = whiteWins;
			whiteWins = blackWins;
			blackWins = t;
		}
	}

	void position::vectorize(std::pair<matrix, matrix>& data) {
		data.first = matrix(1, 128);
		for (int i = 0; i < 32; i++) {
			if (allies & (1 << i)) {
				if (kings & (1 << i))
					data.first[0][i + 32] = 1;
				data.first[0][i] = 1;
			}
			if (enemies & (1 << i)) {
				if (kings & (1 << i))
					data.first[0][i + 96] = 1;
				data.first[0][i + 64] = 1;
			}
		}
		data.second = matrix(1, 1);
		if (whiteWins >= 2.5 * blackWins && whiteWins >= 3)
			data.second[0][0] = 1;
		else data.second[0][0] = 0;
		//std::cout << white << " " << black << std::endl;
		//for (int i = 0; i < 8; i++)
		//	std::cout << data.second[0][i];
		//std::cout << std::endl;
	}

	void loadData(std::vector<std::pair<matrix, matrix>>& trainData, std::vector<std::pair<matrix, matrix>>& testData,
		std::vector<std::pair<matrix, matrix>*>& trainPointers, std::vector<std::pair<matrix, matrix>*>& testPointers,
		int amount, std::string path) {
		position pos;
		std::pair<matrix, matrix> pair;
		std::ifstream fin(path);
		while (fin) {
			pos.read(fin);
			if (!fin)
				break;
			int k = 0;
			for (int i = 0; i < 32; i++) {
				if (pos.allies & (1 << i))
					k++;
				if (pos.enemies & (1 << i))
					k++;
			}
			if (k <= amount || k > amount + 4)
				continue;
			if ((!(pos.whiteWins >= 4 * pos.blackWins) || !(pos.whiteWins >= 3)) &&
				(!(pos.blackWins >= 4 * pos.whiteWins) || !(pos.blackWins >= 3)))
				continue;
			pos.vectorize(pair);
			int r = rand() % 10;
			if (r != 0) {
				trainData.push_back(pair);
				trainPointers.push_back(&trainData[trainData.size() - 1]);
			}
			else {
				testData.push_back(pair);
				testPointers.push_back(&testData[testData.size() - 1]);
			}
		}
		trainPointers = std::vector<std::pair<matrix, matrix>*>(trainData.size());
		testPointers = std::vector<std::pair<matrix, matrix>*>(testData.size());
		for (int i = 0; i < trainData.size(); i++)
			trainPointers[i] = &trainData[i];
		for (int i = 0; i < testData.size(); i++)
			testPointers[i] = &testData[i];
	}
}