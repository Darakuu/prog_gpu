#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "ocl_boiler.h"

/*	const int i = get_global_id(0);
 *	Funzione built-in OpenCL, per ciascuna istanza, indica qual è l'indice globale per qualsiasi istanza.
 *  (int) = dimensione della griglia di lancio. 0 = monodim, 1 = bidim, 2 = tridim.
*/

// I kernel restituiscono sempre void
// Si deve specificare il tipo di memoria, i puntatori di solito risiedono in global memory
cl_event vecinit(cl_kernel vecinit_k, 
					cl_command_queue que,
					global cl_mem d_v1, 
					global cl_mem d_v2, 
					cl_int nels)
{
	const size_t gws[] = {nels};
	cl_event vecinit_evt;
	cl_int err;

	cl_uint i = 0;	//0, 1, 2 v
	err = clSetKernelArg(vecinit_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set vecinit arg", i-1);	// non è il massimo...
	clSetKernelArg(vecinit_k, i++, sizeof(d_v2), &d_v2);
	ocl_check(err, "set vecinit arg", i-1);
	clSetKernelArg(vecinit_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set vecinit arg", i-1);

	err = clEnqueueNDRangeKernel(que, vecinit_k, 1, NULL, gws, NULL, 0, NULL, &vecinit_evt);
	ocl_check(err, "enqueue vecinit");
	return vecinit_evt;
}

/* void vecinit(int * restrict v1, int * restrict v2, int nels)
{
	for (int i = 0; i < nels; ++i)
		vecinit_k(v1, v2, nels, i);
} */

cl_event vecsum(
					cl_kernel vecsum_k,
					cl_command_queue que,
					cl_mem d_vsum, 
					cl_mem d_v1, 
					cl_mem d_v2, 
					cl_int nels)
{
	const size_t gws[] = {nels};
	cl_event vecsum_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(vecsum_k, i++, sizeof(d_vsum), &d_vsum);
	ocl_check(err, "set vecinit arg", i-1);	// non è il massimo...
	clSetKernelArg(vecsum_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set vecinit arg", i-1);
	clSetKernelArg(vecsum_k, i++, sizeof(d_v2), &d_v2);
	ocl_check(err, "set vecinit arg", i-1);
	clSetKernelArg(vecsum_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set vecinit arg", i-1);

	err = clEnqueueNDRangeKernel(que, vecsum_k, 1, NULL, gws, NULL, 0, NULL, &vecsum_evt);
	ocl_check(err, "enqueue vecinit");
	return vecsum_evt;
}

void verify(const int *vsum, int nels)
{
	for (int i = 0; i < nels; ++i) 
	{
		if (vsum[i] != nels) 
		{
			fprintf(stderr, "mismatch @ %d : %d != %d\n", i, vsum[i], nels);
			exit(3);
		}
	}
}


int main(int argc, char *argv[])
{
	if (argc <= 1) 
	{
		fprintf(stderr, "specify number of elements\n");
		exit(1);
	}

	const int nels = atoi(argv[1]);
	const size_t memsize = nels*sizeof(cl_int);	// su device è signed int (con complemento a due)
												// su host è definito il cl_int, garantito che sia identico a quello del device (entro certi limiti)
												// cl_float, cl_double, ...
	cl_platform_id plat_id = select_platform();
	cl_device_id dev_id = select_device(plat_id);
	cl_context ctx = create_context(plat_id, dev_id);
	cl_command_queue que = create_queue(ctx, dev_id);
	cl_program prog = create_program("kernels.ocl", ctx, dev_id);
	cl_init err;

	cl_kernel vecinit_k = clCreateKernel(prog, "vecinit", &err);
	ocl_check(err, "create kernel vecinit");
	cl_kernel vecsum_k = clCreateKernel(prog, "vecinit", &err);
	ocl_check(err, "create kernel vecsum");

	//	   d_ = per device, sanity naming per il programmatore
	cl_mem d_v1 = NULL, cl_mem d_v2 = NULL, cl_mem d_vsum = NULL;

	d_v1 = cl_CreateBuffer(
							ctx,										 	// Contesto
							CL_MEM_READ_WRITE | CL_MEM_HOST_NOACCESS,		// Flags
							memsize,										// Size
							NULL,											// 
							&err);											// Messaggio di errore ritornato
	ocl_check(err, "create buffer d_v1");
	
	d_v2 = cl_CreateBuffer(
							ctx,										 
							CL_MEM_READ_WRITE | CL_MEM_HOST_NOACCESS,	
							memsize,									
							NULL,										
							&err);
	ocl_check(err, "create buffer d_v2");
	
	d_vsum = cl_CreateBuffer(
							ctx,										 
							CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,	
							memsize,									
							NULL,										
							&err);
	ocl_check(err, "create buffer d_vsum");

	

	if ( !v1 || !v2 || !vsum) 
	{
		fprintf(stderr, "failed to malloc arrays\n");
		exit(2);
	}


	clock_t start_init, end_init;
	clock_t start_sum, end_sum;

	start_init = clock();
	vecinit(vecinit_k, que ,d_v1, d_v2, nels);
	end_init = clock();

	start_sum = clock();
	vecsum(vecinit_k, que ,d_v1, d_v2, nels);
	end_sum = clock();

	verify(vsum, nels);

	double runtime_init_ms = (end_init - start_init)*1.0e3/CLOCKS_PER_SEC;
	double runtime_sum_ms = (end_sum - start_sum)*1.0e3/CLOCKS_PER_SEC;

	double init_bw_gbs = 2.0*memsize/1.0e6/runtime_init_ms;
	double sum_bw_gbs = 3.0*memsize/1.0e6/runtime_sum_ms;
	printf("init: %d int in %gms: %g GB/s\n",
		nels, runtime_init_ms, init_bw_gbs);
	printf("sum : %d int in %gms: %g GB/s\n",
		nels, runtime_sum_ms, sum_bw_gbs);

	free(vsum); vsum = NULL;
	free(v2); v2 = NULL;
	free(v1); v1 = NULL;
}
