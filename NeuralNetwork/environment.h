#pragma once

#include "matrix.h"
#include "network.h"

class environment
{
public:
	bool alive = false;

	virtual void reset();
	virtual double step() = 0;
	virtual matrix state() = 0;
};

class qlearn
{
public:
	qlearn();

	void train();

	network ann;
	environment* env;
};
