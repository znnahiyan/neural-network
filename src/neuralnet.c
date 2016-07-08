/*******************************************************************************
 * Summary: Neural networking is the shit. It's got networking AND neurons!
 * 
 * Copyright (c) 2013 -- Zulker Nayeen Nahiyan <nahiyan02@gmail.com>
 * Licensed with the 3-clause BSD license, which means that there's no warranty!
 ******************************************************************************/

#include "neuralnet.h"

////////////////////////////////////////////////////////////////////////////////
/////////////////////////// --- SETUP & STOPPING --- ///////////////////////////
////////////////////////////////////////////////////////////////////////////////
void NNetInit(struct nnet_t* s, uint layers, uint *neurons)
{
	// Save the counts of neurons.
	s->brain         = calloc(layers, sizeof(neuron_t*));
	s->layers        = layers;
	s->neurons       = calloc(layers, sizeof(uint));
	s->delta_coeff   = 0.8f;
	s->epsilon_coeff = 0.8f;
	s->mu_coeff      = 1.75f;
	
	// i is the layer-slot, j is the neuron-slot.
	for (uint i = 0; i < s->layers; i++)
	{
		// Allocate the neuron structs for the [i] layer.
		s->neurons[i] = neurons[i];
		s->brain[i]   = calloc(s->neurons[i], sizeof(neuron_t));
		
		// Skip the input layer, it doesn't need any weight arrays.
		if (i != 0)
			for (uint j = 0; j < s->neurons[i]; j++)
			{
				s->brain[i][j].weights = calloc(s->neurons[i - 1], sizeof(float));
				s->brain[i][j].errors  = calloc(s->neurons[i - 1], sizeof(float));
				s->brain[i][j].deltas  = calloc(s->neurons[i - 1], sizeof(float));
				
				// Initialise the weights with small values in the range [-1, 1).
				for (uint k = 0; k < s->neurons[i - 1]; k++)
				{
					s->brain[i][j].weights[k] = (drand48() - 0.5f) * 2.0f / s->neurons[i - 1];
					s->brain[i][j].errors[k]  = 0.0f;
					s->brain[i][j].deltas[k]  = 1.0f;
				}
				
				s->brain[i][j].bias = (drand48() - 0.5f) * 2.0f;
			}
	}
}

////////////////////////////////////////////////////////////////////////////////

void NNetFree(struct nnet_t *s)
{
	for (uint i = 0; i < s->layers; i++)
	{
		// Deallocate the weights from the neurons, but skip the inputs.
		if (i != 0)
			for (uint j = 0; j < s->neurons[i]; j++)
			{
				free(s->brain[i][j].weights);
				free(s->brain[i][j].errors);
				free(s->brain[i][j].deltas);
			}
		
		free(s->brain[i]);
	}
	
	free(s->brain);
	s->layers = 0;
	free(s->neurons);
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// --- UPDATERS --- ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void NNetUpdateOutputs(struct nnet_t *s)
{
	uint     i,     j;
	uint add_i, add_j;
	
	// Okay, this loops through the layers (1st for),
	// and through their neurons (2nd for),
	// then adds the weighted output of the neurons from the previous layer (3rd
	// for) then finally applies a sigmoid & bias on the sums.
	//
	// For starters, i is the current layer, and j is the current neuron.
	// add_i is (i - 1); it's the layer of the neurons we're adding from.
	// add_j is the slot of the neuron we're adding from.
	for (i = 1, add_i = 0; i < s->layers; add_i = i++)
		for (j = 0; j < s->neurons[i]; j++)
		{
			for (add_j = 0; add_j < s->neurons[add_i]; add_j++)
			{
				s->brain[i][j].output += s->brain[add_i][add_j].output *
				                         s->brain[i][j].weights[add_j];
			}
			
			// Apply bias and sigmoid.
			s->brain[i][j].output += s->brain[i][j].bias;
			s->brain[i][j].output  = NNetActivationFunction(s->brain[i][j].output);
		}
}

////////////////////////////////////////////////////////////////////////////////

// Reference: https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
void NNetUpdateWeights(struct nnet_t *s, float *expected_values)
{
	float output, error, sum, delta;
	s->total_error = 0.0f;
	
	for (uint i = (s->layers - 1); i; i--)
	{
		for (uint j = 0; j < s->neurons[i]; j++)
		{
			// Calculate the error.
			output = s->brain[i][j].output;
			delta  = NNetActivationDerivative(output);
			
			// If the neuron is an output neuron, the factor = (Expected-Output)
			// However, if the neuron is a hidden neuron, then the factor is the
			// summation of the formula:
			// (the next layer neuron's error * the weight to that neuron).
			if (i == (s->layers - 1))
			{
				error  = (expected_values[j] - output);
				delta *= error;
				s->total_error += powf(error, 2.0f);
			}
			else
			{
				sum = 0.0f;
				
				for (uint k = 0; k < s->neurons[i+1]; k++)
					sum += s->brain[i+1][k].error * s->brain[i+1][k].weights[j];
				
				delta *= sum;
			}
			
			delta = NNetErrorFunction(delta);
			
			// Update the weights.
			for (uint k = 0; k < s->neurons[i-1]; k++)
			{
				error = s->delta_coeff * delta * s->brain[i-1][k].output;
				//error = fminf(s->delta_coeff * delta, s->mu_coeff * s->brain[i][j].deltas[k]) * s->brain[i-1][k].output;
				//error = (delta/(s->brain[i][j].errors[k] - delta)) * s->brain[i][j].errors[k] * s->brain[i-1][k].output;
				
				// @debug
				//printf("UPDATE: i=%u j=%u k=%u bias=%+.3f error=%+.3f,%+.3f delta=%+.3f,%+.3f\n",
				//       i, j, k, s->brain[i][j].bias, error, s->brain[i][j].errors[k], delta, s->brain[i][j].deltas[k]);
				
				s->brain[i][j].weights[k] += error;
				s->brain[i][j].errors[k]   = error;
				s->brain[i][j].deltas[k]   = delta;
			}
			
			s->brain[i][j].bias += s->delta_coeff * delta;
			
			// Save the error in the neuron for back-propagation of the error.
			s->brain[i][j].error  = error;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// --- INPUT/OUTPUT --- /////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void NNetInput(struct nnet_t *s, float *input_values)
{
	for (uint j = 0; j < s->neurons[0]; j++)
		s->brain[0][j].output = input_values[j];
}

////////////////////////////////////////////////////////////////////////////////

void NNetOutput(struct nnet_t *s, float *output_values)
{
	for (uint j = 0; j < s->neurons[s->layers - 1]; j++)
		output_values[j] = s->brain[s->layers - 1][j].output;
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// --- HELPER --- ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
inline float NNetActivationFunction(float x)
{
	// Sigmoidal function for transfer. Range is (-1,1).
	return (1.0 / (1.0 + exp(-1.0f * x)));
	
	// Softplus function for transfer. Range is (0, +inf).
	//return log(1 + exp(x));
}

inline float NNetActivationDerivative(float x)
{
	// Derivative of the sigmoidal function.
	return (x * (1 - x));
	
	// Derivative of the softplus function for transfer. Range is (-1, 1).
	//return (1.0 / (1.0 + exp(-1.0f * x)));
}

inline float NNetErrorFunction(float x)
{
	// Hyperbolic arctangent function for error remapping. Range is (-7.5,7.5).
	//return atanhf(x >= 0.999999f ? 0.999999f : (x <= -0.999999f ? -0.999999f : x));
	return x;
}
