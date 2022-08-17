#include "matrix.h"
#include <iostream>
#include <fstream>
#include "time.h"

matrix::matrix() {
	m = 0;
	n = 0;
	vals = std::vector<std::vector<double>>(0);
}

matrix::matrix(std::vector<double> vec) {
	m = 1;
	n = vec.size();
	vals = std::vector<std::vector<double>>(1);
	vals[0] = vec;
}

matrix::matrix(int m, int n) {
	this->m = m;
	this->n = n;

	vals = std::vector<std::vector<double>>(m);
	for (int i = 0; i < m; i++)
		vals[i] = std::vector<double>(n);
}

void matrix::multiply(matrix& m1, matrix& m2) {
	if (m1.m == m && m2.n == n && m1.n == m2.m) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vals[i][j] = 0;
				for (int k = 0; k < m1.n; k++) {
					vals[i][j] += m1[i][k] * m2[k][j];
				}
			}
		}
	}
}

void matrix::multiply(double t) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			vals[i][j] *=  t;
		}
	}
}

void matrix::add(matrix& m1) {
	if (m == m1.m && n == m1.n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vals[i][j] += m1[i][j];
			}
		}
	}
}

void matrix::sum(matrix& m1, matrix& m2) {
	if (m == m1.m && n == m1.n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vals[i][j] = m1[i][j] + m2[i][j];
			}
		}
	}
}

void matrix::sub(matrix& m1) {
	if (m == m1.m && n == m1.n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vals[i][j] -= m1[i][j];
			}
		}
	}
}

void matrix::sub(matrix& m1, matrix& m2) {
	if (m == m1.m && n == m1.n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vals[i][j] = m1[i][j] - m2[i][j];
			}
		}
	}
}

void matrix::foreach(matrix& m1, double f(double)) {
	if (m == m1.m && n == m1.n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vals[i][j] = f(m1[i][j]);
			}
		}
	}
}

void matrix::foreach(double f(double)) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			vals[i][j] = f(vals[i][j]);
		}
	}
}

void matrix::foreachMult(matrix& m1) {
	if (m == m1.m && n == m1.n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vals[i][j] *= m1[i][j];
			}
		}
	}
}

void matrix::foreachDiv(matrix& m1) {
	if (m == m1.m && n == m1.n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vals[i][j] /= m1[i][j];
			}
		}
	}
}

void matrix::compare(matrix& m1) {
	if (m1.m == m && m1.n == n) {
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				if (abs(m1[i][j] - vals[i][j]) > 0.0000001)
					std::cout << "panic\n";
	}
	else {
		std::cout << "panic\n";
	}
}

void matrix::zero() {
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			vals[i][j] = 0;
}

void matrix::rand() {
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			vals[i][j] = std::rand() % 2001 * 0.001 - 1;
}

void matrix::save(std::string file) {
	std::ofstream f(file);
	f << m << " " << n << std::endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			f << vals[i][j] << " ";
		f << std::endl;
	}
}

void matrix::load(std::string file) {
	std::ifstream f(file);
	f >> m >> n;
	vals = std::vector<std::vector<double>>(m);
	for (int i = 0; i < m; i++) {
		vals[i] = std::vector<double>(n);
		for (int j = 0; j < n; j++)
			f >> vals[i][j];
	}
	f.close();
}

void matrix::removeCol(int k) {
	std::vector<std::vector<double>> newVals(m);
	for (int i = 0; i < m; i++) {
		newVals[i] = std::vector<double>(n - 1);
		int t = 0;
		for (int j = 0; j < n; j++) {
			if (j == k)
				j++;
			newVals[i][t] = vals[i][j];
			t++;
		}
	}
	vals = newVals;
	n--;
}

void matrix::removeRow(int k) {
	std::vector<std::vector<double>> newVals(m - 1);
	int t = 0;
	for (int i = 0; i < m; i++) {
		if (i == k)
			i++;
		newVals[t] = std::vector<double>(n);
		for (int j = 0; j < n; j++) {
			newVals[t][j] = vals[i][j];
		}
		t++;
	}
	vals = newVals;
	m--;
}

std::vector<double>& matrix::operator [](int index) {
	return vals[index];
}

void matrix::print() {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			std::cout << vals[i][j] << " ";
		std::cout << "\n";
	}
}