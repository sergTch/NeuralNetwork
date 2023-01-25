#pragma once
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

class matrix
{
private:
	std::shared_ptr<double[]> data;

	size_t istep;
	size_t jstep;

public:
	size_t m;
	size_t n;

	matrix();
	matrix(std::vector<double> vec);
	matrix(double t);
	matrix(size_t m, size_t n);
	matrix(matrix&& A);
	matrix(const matrix& A);

	void removeCol(size_t k);
	void removeRow(size_t k);
	void zero();
	void rand();
	void load(const std::string& file);
	void save(const std::string& file) const;
	void print() const;

	matrix& apply(double (*f)(double));
	matrix& elemMult(const matrix& A);
	matrix& elemDiv(const matrix& A);

	double& get(size_t i, size_t j);
	const double& get(size_t i, size_t j) const;

	matrix T();
	const matrix T() const;

	matrix sum(int axis) const;
	double sum() const;

	matrix& operator= (matrix&& A);
	matrix& operator= (const matrix& A);
	matrix& operator= (double t);

	matrix& operator += (const matrix& A);
	matrix& operator -= (const matrix& A);
	matrix& operator += (double t);
	matrix& operator -= (double t);
	matrix& operator *= (double t);
	matrix& operator /= (double t);
};

matrix operator + (const matrix& A, const matrix& B);
matrix operator - (const matrix& A, const matrix& B);
matrix operator + (double t, const matrix& A);
matrix operator - (double t, const matrix& A);
matrix operator + (const matrix& A, double t);
matrix operator - (const matrix& A, double t);


std::vector<double> operator * (const matrix& A, const std::vector<double>& v);
matrix operator	* (const matrix& A, const matrix& B);
matrix operator	* (const matrix& A, double t);
matrix operator	* (double t, const matrix& A);

matrix operator / (const matrix& A, double t);

