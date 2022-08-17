#pragma once
#include <vector>
#include <string>

class matrix
{
public:
	std::vector<std::vector<double>> vals;
	int m;
	int n;
	matrix();
	matrix(std::vector<double> vec);
	matrix(int m, int n);

	void removeCol(int k);
	void removeRow(int k);
	void zero();
	void rand();
	void load(std::string file);
	void save(std::string file);

	void multiply(matrix& m1, matrix& m2);
	void multiply(double t);
	void add(matrix& m1);
	void sum(matrix& m1, matrix& m2);
	void sub(matrix& m1);
	void sub(matrix& m1, matrix& m2);
	void foreach(matrix& m1, double f(double));
	void foreach(double f(double));
	void foreachMult(matrix& m1);
	void foreachDiv(matrix& m1);
	void compare(matrix& m1);

	std::vector<double>& operator [](int index);
	void print();
};

