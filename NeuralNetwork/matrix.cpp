#include "matrix.h"
#include <iostream>
#include <fstream>
#include "time.h"

matrix::matrix(): m(0), n(0), data(0), istep(0), jstep(0) {}

matrix::matrix(std::vector<double> vec): m(1), n(vec.size()), data(new double[vec.size()]), 
										 istep(vec.size()), jstep(1) {
	for (int i = 0; i < n; i++)
		data[i] = vec[i];
}

matrix::matrix(size_t m, size_t n) : m(m), n(n), data(new double[m * n]), 
									 istep(n), jstep(1) { 
	for (int i = 0; i < m * n; i++)
		data[i] = 0;
}

matrix::matrix(matrix&& A) = default;

matrix::matrix(const matrix& A): m(A.m), n(A.n), istep(A.istep), jstep(A.jstep) {
	data = std::shared_ptr<double[]>(new double[m * n]);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			get(i, j) = A.get(i, j);
}

void matrix::zero() {
	for (size_t i = 0; i < m; i++)
		for (size_t j = 0; j < n; j++)
			get(i, j) = 0;
}

void matrix::rand() {
	for (size_t i = 0; i < m; i++)
		for (size_t j = 0; j < n; j++)
			get(i, j) = std::rand() % 2001 * 0.001 - 1;
}

void matrix::save(const std::string& file) const {
	std::ofstream f(file);
	f << m << " " << n << std::endl;
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++)
			f << get(i, j) << " ";
		f << std::endl;
	}
}

void matrix::load(const std::string& file) {
	std::ifstream f(file);
	f >> m >> n;

	data = std::shared_ptr<double[]>(new double[m * n]);
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++)
			f >> get(i, j);
	}
	f.close();
}

void matrix::removeCol(size_t k) {
	//std::vector<std::vector<double>> newVals(m);
	//for (size_t i = 0; i < m; i++) {
	//	newVals[i] = std::vector<double>(n - 1);
	//	size_t t = 0;
	//	for (size_t j = 0; j < n; j++) {
	//		if (j == k)
	//			j++;
	//		newVals[i][t] = vals[i][j];
	//		t++;
	//	}
	//}
	//vals = newVals;
	//n--;
}

void matrix::removeRow(size_t k) {
	//std::vector<std::vector<double>> newVals(m - 1);
	//size_t t = 0;
	//for (size_t i = 0; i < m; i++) {
	//	if (i == k)
	//		i++;
	//	newVals[t] = std::vector<double>(n);
	//	for (size_t j = 0; j < n; j++) {
	//		newVals[t][j] = vals[i][j];
	//	}
	//	t++;
	//}
	//vals = newVals;
	//m--;
}

matrix matrix::T()
{
	matrix rez;
	rez.data = data;
	rez.m = n;
	rez.n = m;
	rez.istep = jstep;
	rez.jstep = istep;

	return rez;
}

matrix matrix::sum(int axis) const
{
	matrix rez;
	if (axis == 0) {
		rez = matrix(m, 1);
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				rez.get(i, 0) += get(i, j);
	}
	else {
		rez = matrix(1, n);
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				rez.get(0, j) += get(i, j);
	}
	return rez;
}

const double& matrix::get(size_t i, size_t j) const
{
	return data[i * istep + j * jstep];
}

double& matrix::get(size_t i, size_t j)
{
	return data[i * istep + j * jstep];
}

void matrix::print() const {
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++)
			std::cout << get(i, j) << " ";
		std::cout << "\n";
	}
}

matrix& matrix::apply(double(*f)(double))
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			get(i, j) = f(get(i, j));
	return *this;
}

matrix& matrix::elemMult(const matrix& A)
{
	if (m != A.m || n != A.n)
		throw "sizes doesn't match";
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			get(i, j) *= A.get(i, j);
		}
	}
	return *this;
}

matrix& matrix::elemDiv(const matrix& A)
{
	if (m != A.m || n != A.n)
		throw "sizes doesn't match";
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			get(i, j) /= A.get(i, j);
		}
	}
	return *this;
}

double matrix::sum() const
{
	double rez = 0;
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			rez += get(i, j);
	return rez;
}

matrix& matrix::operator = (matrix&& A) = default;

matrix& matrix::operator = (const matrix& A) {
	m = A.m;
	n = A.n;
	istep = A.istep;
	jstep = A.jstep;
	data = std::shared_ptr<double[]>(new double[m * n]);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			get(i, j) = A.get(i, j);
	return *this;
}

matrix& matrix::operator=(double t)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			get(i, j) = t;
	return *this;
}

matrix& matrix::operator += (const matrix& A)
{
	if (m != A.m || n != A.n)
		throw "sizes doesn't match";
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			get(i, j) += A.get(i, j);
		}
	}
	return *this;
}

matrix& matrix::operator -= (const matrix& A)
{
	if (m != A.m || n != A.n)
		throw "sizes doesn't match";
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			get(i, j) -= A.get(i, j);
		}
	}
	return *this;
}

matrix& matrix::operator+=(double t)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			get(i, j) += t;
	return *this;
}

matrix& matrix::operator-=(double t)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			get(i, j) -= t;
	return *this;
}

matrix& matrix::operator *= (double t)
{
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			//vals[i][j] *= t;
			get(i, j) *= t;
		}
	}
	return *this;
}

matrix& matrix::operator /= (double t)
{
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			//vals[i][j] /= t;
			get(i, j) /= t;
		}
	}
	return *this;
}

matrix operator+ (const matrix & A, const matrix & B) {
	matrix rez = A;
	rez += B;
	return rez;
}

matrix operator- (const matrix& A, const matrix& B) {
	matrix rez = A;
	rez -= B;
	return rez;
}

matrix operator+(double t, const matrix& A)
{
	matrix rez = A;
	rez += t;
	return rez;
}

matrix operator-(double t, const matrix& A)
{
	matrix rez = A;
	rez -= t;
	return rez;
}

matrix operator+(const matrix& A, double t)
{
	matrix rez = A;
	rez += t;
	return rez;
}

matrix operator-(const matrix& A, double t)
{
	matrix rez = A;
	rez -= t;
	return rez;
}

std::vector<double> operator * (const matrix& A, const std::vector<double>& v) {

	if (A.n != v.size())
		throw "sizes doesn't match";

	std::vector<double> rez(A.m);

	for (int i = 0; i < rez.size(); i++) {
		for (int j = 0; j < A.n; j++) {
			//rez[i] += A[i][j] * v[j];
			rez[i] += A.get(i, j) * v[j];
		}
	}

	return rez;
}

matrix operator* (const matrix& A, const matrix& B) {

	if (A.n != B.m)
		throw "sizes doesn't match";

	matrix rez(A.m, B.n);

	for (int i = 0; i < rez.m; i++) {
		for (int j = 0; j < rez.n; j++) {
			for (int k = 0; k < A.n; k++) {
				//rez[i][j] += A[i][k] * B[k][j];
				rez.get(i, j) += A.get(i, k) * B.get(k, j);
			}
		}
	}

	return rez;
}

matrix operator* (const matrix& A, double t) {
	matrix rez = A;
	rez *= t;
	return rez;
}

matrix operator* (double t, const matrix& A) {
	matrix rez = A;
	rez *= t;
	return rez;
}

matrix operator/ (const matrix& A, double t) {
	matrix rez = A;
	rez /= t;
	return rez;
}
