#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

/*	const int i = get_global_id(0);
 *	Funzione built-in OpenCL, per ciascuna istanza, indica qual è l'indice globale per qualsiasi istanza.
 *  (int) = dimensione della griglia di lancio. 0 = monodim, 1 = bidim, 2 = tridim.
*/

// I kernel restituiscono sempre void
// Si deve specificare il tipo di memoria, i puntatori di solito risiedono in global memory
cl_event vecinit(	cl_kernel vecinit_k, 
					cl_command_queue que,
					cl_mem d_v1, 
					cl_mem d_v2, 
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

cl_event vecsum(	cl_kernel vecsum_k,
					cl_command_queue que,
					cl_mem d_vsum, 
					cl_mem d_v1, 
					cl_mem d_v2, 
					cl_int nels,
					cl_event init_evt)
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

	err = clEnqueueNDRangeKernel(que, vecsum_k, 1, NULL, gws, NULL, 0, &init_evt, &vecsum_evt);	// &init_evt = sto giocando con puntatori/array perché è uno solo
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
	cl_int err;

	cl_kernel vecinit_k = clCreateKernel(prog, "vecinit", &err);
	ocl_check(err, "create kernel vecinit");
	cl_kernel vecsum_k = clCreateKernel(prog, "vecinit", &err);
	ocl_check(err, "create kernel vecsum");

	//	   d_ = per device, sanity naming per il programmatore
	cl_mem d_v1 = NULL, d_v2 = NULL, d_vsum = NULL;

	d_v1 = clCreateBuffer(
							ctx,										 	// Contesto
							CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,		// Flags
							memsize,										// Size
							NULL,											// 
							&err);											// Messaggio di errore ritornato
	ocl_check(err, "create buffer d_v1");
	
	d_v2 = clCreateBuffer(
							ctx,										 
							CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,	
							memsize,									
							NULL,										
							&err);
	ocl_check(err, "create buffer d_v2");
	
	d_vsum = clCreateBuffer(
							ctx,										 
							CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,	
							memsize,									
							NULL,										
							&err);
	ocl_check(err, "create buffer d_vsum");

	

	if ( !d_v1 || !d_v2 || !d_vsum) 
	{
		fprintf(stderr, "failed to malloc arrays\n");
		exit(2);
	}


	clock_t start_init, end_init;
	clock_t start_sum, end_sum;
	cl_event init_evt, sum_evt, read_evt;

	start_init = clock();
	init_evt = vecinit(vecinit_k, que ,d_v1, d_v2, nels);
	end_init = clock();

	start_sum = clock();
	sum_evt = vecsum(vecinit_k, que, d_vsum, d_v1, d_v2, nels, init_evt);
	end_sum = clock();

	//	clFinish() 						Aspetta che tutti i comandi hanno finito, brutale punto di sincronizzazione. E' bloccante.
	//	clFlush()						Non è un punto di sincronizzazione, ma garantisce solo che la GPU abbia ricevuto tutti i comandi.
	//	clWaitForEvents(1, &sum_evt); 	Garanzia che vecsum ha concluso l'operazione

	int *h_vsum=malloc(memsize);
	if(!h_vsum) ocl_check(CL_OUT_OF_HOST_MEMORY, "alloc vsum host");

	err = clEnqueueReadBuffer(que, d_vsum, CL_FALSE, 0, memsize, h_vsum, 1, &sum_evt, &read_evt);
	//					command_queue, buffer, blocking?, offset, size, puntatore, eventi in lista, lista di eventi, evento nuovo)
	ocl_check(err, "read buffer vsum");
	
	clWaitForEvents(1, &read_evt);	// Garanzia che vecsum ha concluso l'operazione

	//verify(d_vsum, nels);

	double runtime_init_ms = (end_init - start_init)*1.0e3/CLOCKS_PER_SEC;
	double runtime_sum_ms = (end_sum - start_sum)*1.0e3/CLOCKS_PER_SEC;

	double init_bw_gbs = 2.0*memsize/1.0e6/runtime_init_ms;
	double sum_bw_gbs = 3.0*memsize/1.0e6/runtime_sum_ms;
	printf("init: %d int in %gms: %g GB/s\n",
		nels, runtime_init_ms, init_bw_gbs);
	printf("sum : %d int in %gms: %g GB/s\n",
		nels, runtime_sum_ms, sum_bw_gbs);

	free(h_vsum); 	h_vsum = NULL;
	// così o fai exit();
	clReleaseMemObject(d_vsum);		// Per i buffer.
	clReleaseMemObject(d_v1);		// Esistono i buffer classici (blocco contiguo indicizzato di memoria)
	clReleaseMemObject(d_v2);		// Oppure le 'immagini' (strutture opache, nella grafica sono usate come texture)
	clReleaseKernel(vecsum_k);		
	clReleaseKernel(vecinit_k);		
	clReleaseProgram(prog); 		
	clReleaseCommandQueue(que);		
	clReleaseContext(ctx);				
}
