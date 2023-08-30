#include "PID.h"
#include <iostream>
#include <algorithm>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	this->Kp = Kp; // Proportionnal term
	this->Ki = Ki; // Integral term 
	this->Kd = Kd; // Differential term

	// Init the Proportionnal error to 0
	p_error = 0.0;

	// Init the Integral term to 0
	// Because, at the very beginning, the total area (Integral definition) between the position of the car 
	// and the CTE is null (because the car has not started yet)
	i_error = 0.0;

	// Init the Differential error to 0
	// Because, at the very beginning, the position of the car is assumed to be far away from CTE
	d_error = 0.0;
}

void PID::UpdateError(double cte) {
	
	double prev_cte = p_error;
	
	// Proportional error.
	p_error = cte;

	// Integral error.
	i_error += cte;
	
	// Diferential error.
	d_error = cte - prev_cte;
	
	prev_cte = cte;

	
	// update total error only if we're past number of settle steps
	/*
	if ((step % (n_settle_steps + n_eval_steps) > n_settle_steps) && step/(n_settle_steps + n_eval_steps) < 1))  {
		total_error += pow(cte, 2);
	}
	*/
}

double PID::TotalError() {
	return  p_error * Kp + i_error * Ki + d_error * Kd;
}
