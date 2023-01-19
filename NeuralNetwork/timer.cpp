#include "timer.h"
#include <time.h>
#include <iostream>

using namespace std;

static clock_t tStart;
static bool working = false;

void timer_restart() {
	if (working)
		timer_log();
	tStart = clock();
	working = true;
}

void timer_log() {
	cout << "Time from restart: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;
}
