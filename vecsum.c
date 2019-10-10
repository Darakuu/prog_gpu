#include <stdio.h>
#include <stdlib.h>

/*	Puntatori ristretti
	Il compilatore non può evitare che il puntatore di v1 e v2 non si riferiscano ad aree di memoria sovrapposte
	Così non può fare loop unrolling o altre ottimizzazioni e ho casini vari, nel risultato sopratutto.

	Voglio informare il compilatore che non capiterà mai di sovrapporre le aree di memoria dei parametri v1 e v2, saranno sempre distinte.
	uso allora i puntatori ristretti
*/

void vec_init(int* restrict v1, int* restrict v2, int nels) 
{
	for(int i = 0; i < nels; ++i)
	{
		v1[i] = i;
		v2[i] = nels-i;
	}
}

void vec_sum(int * restrict vsum, const int * restrict v1, const int * restrict v2, int nels)
{
	for(int i = 0; i < nels; ++i)
	{
		vsum[i] = v1[i] + v2[i];
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
