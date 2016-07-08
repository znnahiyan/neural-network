/*******************************************************************************
 * Summary: Neural networking is the shit. It's got networking AND neurons!
 * 
 * Copyright (c) 2013 -- Zulker Nayeen Nahiyan <nahiyan02@gmail.com>
 * Licensed with the 3-clause BSD license, which means that there's no warranty!
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>

#include "neuralnet.h"

void output_data(int signal);

struct nnet_t s;

int main()
{
	uint  neurons[] = {2, 3, 1}, raw[2];
	float inputs[2], expected[1], outputs[1];
	
	srand48(time(0));
	signal(SIGINT, &output_data);
	
	NNetInit(&s, sizeof(neurons)/sizeof(uint), neurons);
	s.delta_coeff   = 2.0f;
	s.epsilon_coeff = 0.6f;
	s.mu_coeff      = 1.1f;
	
	for (uint count = 1; count <= 3000; count++)
	{
		raw[0] = lrand48() & 0x1; inputs[0] = raw[0] ? 1.0f : 0.0f;
		raw[1] = lrand48() & 0x1; inputs[1] = raw[1] ? 1.0f : 0.0f;
		expected[0] = (raw[0] ^ raw[1]) ? 1.0f : 0.0f;
		
		NNetInput(&s, inputs);
		NNetUpdateOutputs(&s);
		NNetOutput(&s, outputs);
		NNetUpdateWeights(&s, expected);
		
		printf("Propagation %4u: input=%.3f,%.3f output=%.3f/%.3f error=%.6f\n",
		       count, inputs[0], inputs[1], outputs[0], expected[0], s.total_error);
		//printf("%f\n", s.total_error);
	}
	
	output_data(0);
	
	return 0;
}

void output_data(int signal)
{
	fputs(stdout, "\nNEURAL NETWORK DATA\n");
	
	for (uint i = 0; i < (s.layers - 1); i++)
	{
		printf("LAYER %u-%u:\n", i, i+1);
		
		for (uint j = 0; j < s.neurons[i]; j++)
		{
			printf("NEURON %2u:", j);
			
			for (uint k = 0; k < s.neurons[i+1]; k++)
				printf(" %+f", s.brain[i+1][k].weights[j]);
			
			printf(" BIAS=%+f\n", s.brain[i][j].bias);
		}
	}
	
	exit(signal);
}
