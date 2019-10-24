#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

size_t gws_align_init;		// Struct per fare le cose meglio
size_t gws_align_sum;		// global variables bad

// I kernel restituiscono sempre void
// Si deve specificare il tipo di memoria, i puntatori di solito risiedono in global memory
cl_event vecinit(cl_kernel vecinit_k, cl_command_queue que, 
				 cl_mem d_v1,  cl_mem d_v2, cl_int nels)
{
	// Divisione intera tra due numeri arrotondando per eccesso: (a+b-1)/b => *b arrotonda al multiplo più vicino


	//static const size_t gws_align = 1024; 								// Allineamento del global work size
	const size_t gws[] = {round_mul_up(nels,gws_align_init)};				// Sarà sempre un multiplo di gws_align
	printf("init gws: %d | %zu = %zu\n",nels, gws_align_init, gws[0]);
	
	cl_event vecinit_evt;
	cl_int err;

	cl_uint i = 0;	//0, 1, 2 v
	err = clSetKernelArg(vecinit_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set vecinit arg_dv1", i-1);	// non è il massimo...
	err = clSetKernelArg(vecinit_k, i++, sizeof(d_v2), &d_v2);
	ocl_check(err, "set vecinit arg_dv2", i-1);
	err = clSetKernelArg(vecinit_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set vecinit arg_nels", i-1);
	
	err = clEnqueueNDRangeKernel(que, vecinit_k, 1, NULL, gws, NULL, 0, NULL, &vecinit_evt);
	ocl_check(err, "enqueue vecinit");
	return vecinit_evt;
}

cl_event vecsum(cl_kernel vecsum_k, cl_command_queue que, 
				cl_mem d_vsum, cl_mem d_v1, cl_mem d_v2, cl_int nels, cl_event init_evt)
{
	//static const size_t gws_align = 1024; 					// Allineamento del global work size
	const size_t gws[] = {round_mul_up(nels,gws_align_sum)};	// Sarà sempre un multiplo di gws_align
	printf("sum gws: %d | %zu = %zu\n",nels, gws_align_sum, gws[0]);

	cl_event vecsum_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(vecsum_k, i++, sizeof(d_vsum), &d_vsum);
	ocl_check(err, "set vecsum arg_dvsum", i-1);	// non è il massimo...
	err = clSetKernelArg(vecsum_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set vecsum arg_dv1", i-1);
	err = clSetKernelArg(vecsum_k, i++, sizeof(d_v2), &d_v2);
	ocl_check(err, "set vecsum arg_dv2", i-1);
	err = clSetKernelArg(vecsum_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set vecsum arg_nels", i-1);

	err = clEnqueueNDRangeKernel(que, vecsum_k, 1, NULL, gws, NULL, 1, &init_evt, &vecsum_evt);	// &init_evt = sto giocando con puntatori/array perché è uno solo
	ocl_check(err, "enqueue vecsum");
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
	cl_program prog = create_program("vecsum.ocl", ctx, dev_id);	// File dei kernel deve avere lo stesso identico nome
	cl_int err;

	cl_kernel vecinit_k = clCreateKernel(prog, "vecinit", &err);
	ocl_check(err, "create kernel vecinit");
	cl_kernel vecsum_k = clCreateKernel(prog, "vecsum", &err);
	ocl_check(err, "create kernel vecsum");

	err = clGetKernelWorkGroupInfo(vecinit_k, dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(gws_align_init), &gws_align_init, NULL);
	ocl_check(err, "Preferred wg multiple for init");
	err = clGetKernelWorkGroupInfo(vecsum_k, dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(gws_align_sum), &gws_align_sum, NULL);
	ocl_check(err, "Preferred wg multiple for sum");

	//	   d_ = per device, sanity naming per il programmatore
	cl_mem d_v1 = NULL, d_v2 = NULL, d_vsum = NULL;

	d_v1 = clCreateBuffer(
							ctx,										 	// Contesto
							CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,		// Flags
							memsize,										// Size
							NULL,											// Puntatore all'host
							&err);											// Messaggio di errore ritornato
	ocl_check(err, "create buffer d_v1");
	
	d_v2 = clCreateBuffer(
							ctx,										 
							CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,	
							memsize,									
							NULL,										
							&err);
	ocl_check(err, "create buffer d_v2");
	
	// Sappiamo che leggeremo i dati tramite map
	d_vsum = clCreateBuffer(
							ctx,										 
							CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,	
							memsize,									
							NULL,										
							&err);
	ocl_check(err, "create buffer d_vsum");

	clock_t start_init, end_init;
	clock_t start_sum, end_sum;
	cl_event init_evt, sum_evt, read_evt;

	start_init = clock();
	init_evt = vecinit(vecinit_k, que, d_v1, d_v2, nels);
	end_init = clock();

	start_sum = clock();
	sum_evt = vecsum(vecsum_k, que, d_vsum, d_v1, d_v2, nels, init_evt);
	end_sum = clock();

	//	clFinish() 						Aspetta che tutti i comandi hanno finito, brutale punto di sincronizzazione. E' bloccante.
	//	clFlush()						Non è un punto di sincronizzazione, ma garantisce solo che la GPU abbia ricevuto tutti i comandi.
	//	clWaitForEvents(1, &sum_evt); 	Garanzia che vecsum ha concluso l'operazione


#if 0
	Vecchio(e lento) modo di fare le cose con ReadBuffer:
	int *h_vsum=malloc(memsize);	// Copia estremamente dispendiosa
	if(!h_vsum) ocl_check(CL_OUT_OF_HOST_MEMORY, "alloc vsum host");

	//ReadBuffer = Copia (in modo opaco) i dati
	//MapBuffer  = Rende accessibili i contenuti del buffer all'host. Comando ASINCRONO (serve coda).
	
	err = clEnqueueReadBuffer(que, d_vsum, CL_FALSE, 0, memsize, h_vsum, 1, &sum_evt, &read_evt);
	//					command_queue, buffer, blocking?, offset, size, puntatore, eventi in lista, lista di eventi, evento nuovo)
	ocl_check(err, "read buffer vsum");
#else
	//Rende visibile all'host (CPU) il contenuto di un buffer
	cl_int * h_vsum = clEnqueueMapBuffer(que,d_vsum,CL_FALSE,CL_MAP_READ,							// Non eseguire mai kernel mentre il buffer è mappato, undefined behaviour, 
										0,memsize,1,&sum_evt, &read_evt /*"Map_Event"!*/, &err);	// idem se mappo su un readbuffer
#endif
	clWaitForEvents(1, &read_evt);	// Garanzia che vecsum ha concluso l'operazione

	verify(h_vsum, nels);

	const double runtime_init_ms = runtime_ms(init_evt);
	const double runtime_sum_ms  = runtime_ms(sum_evt );
	const double runtime_read_ms = runtime_ms(read_evt);

	const double init_bw_gbs = 2.0*memsize/1.0e6/runtime_init_ms;
	const double sum_bw_gbs = 3.0*memsize/1.0e6/runtime_sum_ms;
	const double read_bw_gbs = memsize/1.0e6/runtime_read_ms;

	printf("init: %d int in %gms: %g GB/s %g GE/s\n",
		nels, runtime_init_ms, init_bw_gbs, nels/1.0e6/runtime_init_ms);
	printf("sum : %d int in %gms: %g GB/s %g GE/s\n",
		nels, runtime_sum_ms, sum_bw_gbs,nels/1.0e6/runtime_sum_ms);
	printf("read : %d int in %gms: %g GB/s %g GE/s\n",
		nels, runtime_read_ms, read_bw_gbs,nels/1.0e6/runtime_read_ms);

	//	free(h_vsum); 	h_vsum = NULL; Con MapBuffer non si può più fare free 
	err = clEnqueueUnmapMemObject(que,d_vsum,h_vsum,0,NULL,NULL);				
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
