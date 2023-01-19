#pragma once
#include "matrix.h"
#include <vector>
#include <string>

double sigm(double);
double sigmDeriv(double);
double relu(double);
double reluDeriv(double);

enum actType {
	sigmType = 0,
	reluType
};

class layer
{
public:
	actType atype;
	matrix W;
	matrix B;
	matrix* sums = nullptr;
	matrix* out = nullptr;
	
	double (*activate)(double);
	double (*activateD)(double);

	layer();
	layer(size_t input, size_t output, actType t);

	void setAct(actType t);

	void feed(const matrix& input) const;
	
	size_t inpSize() const;
	size_t outSize() const;

	layer& operator += (const layer& l);
	layer& operator -= (const layer& l);
	layer& operator *= (double t);
	layer& operator /= (double t);
	layer& operator = (double t);

	//layer& operator = (const layer& l);
	//layer& operator = (layer&& l);
};
