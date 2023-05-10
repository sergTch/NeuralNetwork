#include "layer.h"
#include <algorithm>

std::vector<double(*)(double)> activations = { sigm, relu, linear, tanh };
std::vector<double(*)(double)> activationDerivs = { sigmDeriv, reluDeriv, linearDeriv, tanhfDeriv };

double sigm(double x)
{
	double e = exp(-x);
	return 1 / (e + 1);
}

double sigmDeriv(double x)
{
	double e = exp(-x);
	return e / ((e + 1) * (e + 1));
}

double relu(double x)
{
	return std::max(x, 0.0);
}

double reluDeriv(double x)
{
	return x > 0 ? 1 : 0;
}

double linear(double x)
{
	return x;
}

double linearDeriv(double x)
{
	return 1;
}

double tanhf(double x)
{
	double e = exp(-2 * x);
	return (1 - e) / (1 + e);
}

double tanhfDeriv(double x)
{
	double e = exp(2 * x);
	return 4 * e / ((1 + e) * (1 + e));
}

layer::layer(): activate(nullptr), activateD(nullptr), atype(sigmType) {}

layer::layer(size_t input, size_t output, actType t) : W(input, output), B(1, output), atype(t)
{
	activate = activations[t];
	activateD = activationDerivs[t];
	W.rand();
	B.rand();
}

layer::layer(const layer& l) = default;
layer::layer(layer&& l) = default;

void layer::setAct(actType t)
{
	atype = t;
	activate = activations[t];
	activateD = activationDerivs[t];
}

matrix layer::feed(const matrix& input) const
{	
	return (input * W).apply(activate);
}

matrix& layer::feed(const matrix& input, matrix& sums, matrix& out) const
{
	sums = input * W;
	out = sums;
	out.apply(activate);
	return out;
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

layer& layer::operator=(const layer& l) = default;
layer& layer::operator=(layer&& l) = default;
