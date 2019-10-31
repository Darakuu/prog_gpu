#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

size_t gws_align_init;
size_t gws_align_smooth;		

cl_event matinit(cl_kernel matinit_k, cl_command_queue que, cl_int lws_cli, 
				 cl_mem d_v1, cl_int nrows, cl_int ncols)
{
	const size_t gws[] = {nrows, ncols};  			// Sarà sempre un multiplo del local work size, se specificato
	printf("init gws: %d | %zu = %zu\n",nels, gws_align_init, gws[0]);
	
	cl_event matinit_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(matinit_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set vecinit arg_dv1", i-1);
	err = clSetKernelArg(matinit_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set vecinit arg_nels", i-1);
	
	err = clEnqueueNDRangeKernel(que, matinit_k, 2, NULL, gws, NULL, 0, NULL, &matinit_evt);
	ocl_check(err, "enqueue vecinit");
	return matinit_evt;
}

void verify(const int *h_A, int nrows, int ncols)
{
  for (int r = 0; r < nrows; ++r)
  {
    for (int c = 0; c < ncols; ++c)
    {
      if (h_A[r*ncols+c] != r-c)
      {
        fprintf(stderr, "mismatch = (%d, %d) : %d != %d\n",
                r,c,h_A[r*ncols+c],r-c);
                exit(3);
      }
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
	const size_t memsize = nels*sizeof(cl_int);

	const int lws = argc > 2 ? atoi(argv[2] ) : 0; //specifico anche local work size

  cl_platform_id plat_id = select_platform();
	cl_device_id dev_id = select_device(plat_id);
	cl_context ctx = create_context(plat_id, dev_id);
	cl_command_queue que = create_queue(ctx, dev_id);
	cl_program prog = create_program("vecsmooth.ocl", ctx, dev_id);	// File dei kernel deve avere lo stesso identico nome
	cl_int err;

	cl_kernel vecinit_k = clCreateKernel(prog, "vecinit", &err);
	ocl_check(err, "create kernel vecinit");
	cl_kernel vecsmooth_k = clCreateKernel(prog, "vecsmooth", &err);
	ocl_check(err, "create kernel vecsmooth");

	err = clGetKernelWorkGroupInfo(vecinit_k, dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
				sizeof(gws_align_init), &gws_align_init, NULL);
	ocl_check(err, "Preferred wg multiple for init");
	err = clGetKernelWorkGroupInfo(vecsmooth_k, dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
				sizeof(gws_align_smooth), &gws_align_smooth, NULL);
	ocl_check(err, "Preferred wg multiple for smooth");

	// d_ = per device, sanity naming per il programmatore
	cl_mem d_A = NULL, d_vsmooth = NULL;

	d_A = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, memsize, NULL, &err);
	ocl_check(err, "create buffer d_v1");

	cl_event init_evt, smooth_evt, read_evt;

	init_evt = vecinit(vecinit_k, que, lws, d_A, nels);
	
	smooth_evt = vecsmooth(vecsmooth_k, que, lws, d_vsmooth, d_v1, nels, init_evt);
	
	cl_int * h_A = clEnqueueMapBuffer(que,d_vsmooth,CL_FALSE,CL_MAP_READ, 0,memsize,1,&smooth_evt, &read_evt , &err);
	clWaitForEvents(1, &read_evt);	// Garanzia che vecsmooth ha concluso l'operazione
	verify(h_vsmooth, nels);

	const double runtime_init_ms = runtime_ms(init_evt);
	const double runtime_smooth_ms  = runtime_ms(smooth_evt );
	const double runtime_read_ms = runtime_ms(read_evt);

	const double init_bw_gbs = 1.0*memsize/1.0e6/runtime_init_ms;
	const double smooth_bw_gbs = 4.0*memsize/1.0e6/runtime_smooth_ms; // Ignoriamo quello che succede all'ultimo elemento. 3 read, 1 write
	const double read_bw_gbs = memsize/1.0e6/runtime_read_ms;

	printf("init: %d int in %gms: %g GB/s %g GE/s\n",
		nels, runtime_init_ms, init_bw_gbs, nels/1.0e6/runtime_init_ms);
	printf("smooth : %d int in %gms: %g GB/s %g GE/s\n",
		nels, runtime_smooth_ms, smooth_bw_gbs,nels/1.0e6/runtime_smooth_ms);
	printf("read : %d int in %gms: %g GB/s %g GE/s\n",
		nels, runtime_read_ms, read_bw_gbs,nels/1.0e6/runtime_read_ms);

	err = clEnqueueUnmapMemObject(que,d_vsmooth,h_vsmooth,0,NULL,NULL);
	ocl_check(err, "unmap vsmooth");
  clReleaseMemObject(d_vsmooth);		// Rilascio per i buffer openCL, in questo caso blocchi contigui di memoria.
	clReleaseMemObject(d_v1);
	clReleaseKernel(vecsmooth_k);		
	clReleaseKernel(vecinit_k);		
	clReleaseProgram(prog); 		
	clReleaseCommandQueue(que);		
	clReleaseContext(ctx);				
}