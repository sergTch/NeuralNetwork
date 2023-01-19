#include "layer.h"
#include <algorithm>

std::vector<double(*)(double)> activations = { sigm, relu };
std::vector<double(*)(double)> activationDerivs = { sigmDeriv, reluDeriv };

double sigm(double x)
{
	double e = exp(-x);
	return 1 / (e + 1);
}

double sigmDeriv(double x)
{
	double e = exp(-x);
	return e / (e + 1) / (e + 1);
}

double relu(double x)
{
	return std::max(x, 0.0);
}

double reluDeriv(double x)
{
	return x > 0 ? 1 : 0;
}

layer::layer(): activate(nullptr), activateD(nullptr), atype(sigmType) {}

layer::layer(size_t input, size_t output, actType t) : W(input, output), B(1, output), atype(t)
{
	activate = activations[t];
	activateD = activationDerivs[t];
	W.rand();
	B.rand();
}

void layer::setAct(actType t)
{
	atype = t;
	activate = activations[t];
	activateD = activationDerivs[t];
}

void layer::feed(const matrix& input) const
{
	*sums = input * W;
	*out = *sums;
	out->apply(activate);
}

size_t layer::inpSize() const
{
	return W.m;
}

size_t layer::outSize() const
{
	return B.n;
}

layer& layer::operator += (const layer& l)
{
	B += l.B;
	W += l.W;
	return *this;
}

layer& layer::operator -= (const layer& l)
{
	B -= l.B;
	W -= l.W;
	return *this;
}

layer& layer::operator *= (double t)
{
	B *= t;
	W *= t;
	return *this;
}

layer& layer::operator /= (double t)
{
	B /= t;
	W /= t;
	return *this;
}

layer& layer::operator=(double t)
{
	B = t;
	W = t;
	return *this;
}

//layer& layer::operator=(const layer& l) = default;
//layer& layer::operator=(layer&& l) = default;
