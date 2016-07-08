/*******************************************************************************
 * Summary: Neural networking is the shit. It's got networking AND neurons!
 * 
 * Copyright (c) 2013 -- Zulker Nayeen Nahiyan <nahiyan02@gmail.com>
 * Licensed with the 3-clause BSD license, which means that there's no warranty!
 ******************************************************************************/

#ifndef NEURALNET_H
#define NEURALNET_H

/////////////////////////// --- HEADERS/INCLUDES --- ///////////////////////////
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>
#include <time.h>

///////////////////////// --- DEFINES AND DECLARES --- /////////////////////////
typedef struct
{
	float output, error, bias, *weights, *errors, *deltas;
} neuron_t;

struct nnet_t
{
	neuron_t** brain;      // -- actual array that holds the neurons. It's 2D.
	uint layers, *neurons; // -- the counts/sizes of the arrays.
	float delta_coeff, epsilon_coeff, mu_coeff;
	float total_error;
};

///////////////////////// --- FUNCTION PROTOTYPING --- /////////////////////////
void NNetInit(struct nnet_t *s, uint layers, uint *neurons);
void NNetFree(struct nnet_t *s);

void NNetUpdateOutputs(struct nnet_t *s);
void NNetUpdateWeights(struct nnet_t *s, float *expected_values);

void NNetInput (struct nnet_t *s, float *input_values);
void NNetOutput(struct nnet_t *s, float *output_values);

float NNetActivationFunction  (float x);
float NNetActivationDerivative(float x);
float NNetErrorFunction       (float x);

#endif // NEURALNET_H
