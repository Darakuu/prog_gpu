#include <stdio.h>
#include <stdlib.h>

void vec_init(int* restrict v1, int* restrict v2, int nels) 
{
	for(int i = 0; i < nels; ++i)
	{
		v1[i] = i;
		v2[i] = nels-i;
	}
}

void vec_sum_k(int * restrict vsum, const int * restrict v1, const int * restrict v2, int nels, int i)
{
		vsum[i] = v1[i] + v2[i];
}

void vec_sum(int * restrict vsum, const int * restrict v1, const int * restrict v2, int nels)
{
	for(int i = 0; i < nels; ++i)
	{
		vec_sum_k(vsum, v1, v2, nels, i);   // Funzione scorporata, consigliabile per i kernel
	}
}

void verify(const int * restrict vsum, int nels)	// Verify per codice più complicato: controllare parallelo con seriale (che è sicuro è giusto)
{
	for (int i=0; i < nels; ++i)
	{
		if(vsum[i] != nels)
		{
			fprintf(stderr,"mismatch @%d : %d\n", i, vsum[i]);
			exit(3);
		}
	}
}

int main(int argc, char*argv[])
{	
	if(argc<=1)
	{
		fprintf(stderr, "Specify number of elements\n");
		exit(1);
	}	

	int nels = atoi(argv[1]);
	
	// In openCL cl_int è a 32bit sicuro
	int *v1 = NULL, *v2 = NULL, *vsum = NULL;
	size_t memsize = nels*sizeof(*v1);
	
	v1 = malloc(memsize);		//*Missing
	v2 = malloc(memsize);		//* if(v1==NULL) exit(-1)
	vsum = malloc(memsize);		//*Errorcheck (è sotto)

	if (!v1 || !v2 || !vsum)
	{
		fprintf(stderr,"Failed to malloc arrays\n");
		exit(2);
	}

	vec_init(v1,v2,nels);
	vec_sum(vsum, v1,v2,nels);
	verify(vsum,nels);
	
	free(vsum);	vsum=NULL;
	free(v2);	v2=NULL;
	free(v1);	v1=NULL;
}
