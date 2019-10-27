#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

size_t gws_align_init;
size_t gws_align_sum;		

cl_event vecinit(cl_kernel vecinit_k, cl_command_queue que, 
				 cl_mem d_v1,  cl_mem d_v2, cl_int nels)
{ 						
 	const size_t gws[] = {round_mul_up(nels,gws_align_init)};	 
	printf("init gws: %d | %zu = %zu\n",nels, gws_align_init, gws[0]);
	
	cl_event vecinit_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(vecinit_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set vecinit arg_dv1", i-1);
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
	
	const cl_int noct = nels/8;
	const size_t gws[] = {round_mul_up(noct,gws_align_sum)};
	printf("sum gws: %d | %zu = %zu\n",nels, gws_align_sum, gws[0]);

	cl_event vecsum_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(vecsum_k, i++, sizeof(d_vsum), &d_vsum);
	ocl_check(err, "set vecsum arg_dvsum", i-1);
	err = clSetKernelArg(vecsum_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set vecsum arg_dv1", i-1);
	err = clSetKernelArg(vecsum_k, i++, sizeof(d_v2), &d_v2);
	ocl_check(err, "set vecsum arg_dv2", i-1);
	err = clSetKernelArg(vecsum_k, i++, sizeof(noct), &noct);
	ocl_check(err, "set vecsum arg_noct", i-1);

	err = clEnqueueNDRangeKernel(que, vecsum_k, 1, NULL, gws, NULL, 1, &init_evt, &vecsum_evt);
	ocl_check(err, "enqueue vecsum");
	return vecsum_evt;
}

void verify(const int *vsum, int nels)
{
	for (int i = 0; i < nels; ++i) 
		if (vsum[i] != nels) 
		{
			fprintf(stderr, "mismatch @ %d : %d != %d\n", i, vsum[i], nels);
			exit(3);
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
	const size_t memsize = nels*sizeof(cl_int);

	if (nels & 7)	// Vale solo per potenze di 2 (per 4 gli ultimi due bit sono 0 se è 4)
	{
		fprintf(stderr, "nels must be multiple of 8\n");	// Non usare modulo.
		exit(1);
	}

	cl_platform_id plat_id = select_platform();
	cl_device_id dev_id = select_device(plat_id);
	cl_context ctx = create_context(plat_id, dev_id);
	cl_command_queue que = create_queue(ctx, dev_id);
	cl_program prog = create_program("vecsum.ocl", ctx, dev_id);
	cl_int err;

	cl_kernel vecinit_k = clCreateKernel(prog, "vecinit", &err);
	ocl_check(err, "create kernel vecinit");
	cl_kernel vecsum_k = clCreateKernel(prog, "vecsum4x2", &err);
	ocl_check(err, "create kernel vecsum");

	err = clGetKernelWorkGroupInfo(vecinit_k, dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(gws_align_init), &gws_align_init, NULL);
	ocl_check(err, "Preferred wg multiple for init");
	err = clGetKernelWorkGroupInfo(vecsum_k, dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(gws_align_sum), &gws_align_sum, NULL);
	ocl_check(err, "Preferred wg multiple for sum");

	//	   d_ = per device, sanity naming per il programmatore
	cl_mem d_v1 = NULL, d_v2 = NULL, d_vsum = NULL;

	d_v1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, memsize, NULL, &err);											
	ocl_check(err, "create buffer d_v1");
	
	d_v2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, memsize, NULL, &err);					
	ocl_check(err, "create buffer d_v2");
	
	// Sappiamo che leggeremo i dati tramite map
	d_vsum = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
	ocl_check(err, "create buffer d_vsum");

	cl_event init_evt, sum_evt, read_evt;

	init_evt = vecinit(vecinit_k, que, d_v1, d_v2, nels);

	sum_evt = vecsum(vecsum_k, que, d_vsum, d_v1, d_v2, nels, init_evt);

	cl_int * h_vsum = clEnqueueMapBuffer(que,d_vsum,CL_FALSE,CL_MAP_READ,
										0,memsize,1,&sum_evt, &read_evt, &err);
	clWaitForEvents(1, &read_evt);

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

	err = clEnqueueUnmapMemObject(que,d_vsum,h_vsum,0,NULL,NULL);
	ocl_check(err, "unmap vsum");				
	
	clReleaseMemObject(d_vsum);			// Per i buffer.
	clReleaseMemObject(d_v1);				// Esistono i buffer classici (blocco contiguo indicizzato di memoria)
	clReleaseMemObject(d_v2);				// Oppure le 'immagini' (strutture opache, nella grafica sono usate come texture)
	clReleaseKernel(vecsum_k);		
	clReleaseKernel(vecinit_k);		
	clReleaseProgram(prog); 		
	clReleaseCommandQueue(que);		
	clReleaseContext(ctx);				
}
