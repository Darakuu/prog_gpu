#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

size_t gws_align_init;

cl_event matinit(cl_kernel matinit_k, cl_command_queue que, 
									cl_mem d_A, cl_int nrows, cl_int ncols, cl_int pitch_el)
{
	const size_t gws[] = {round_mul_up(ncols, gws_align_init), nrows};  			// Al contrario solo per mantenere la coalescenza
	cl_event matinit_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(matinit_k, i++, sizeof(d_A), &d_A);
	ocl_check(err, "set matinit arg_dv1", i-1);
	err = clSetKernelArg(matinit_k, i++, sizeof(ncols), &ncols);
	ocl_check(err, "set matinit arg_ncols", i-1);
	err = clSetKernelArg(matinit_k, i++, sizeof(nrows), &nrows);
	ocl_check(err, "set matinit arg_nrows", i-1);
  err = clSetKernelArg(matinit_k, i++, sizeof(pitch_el), &pitch_el);
	ocl_check(err, "set matinit arg_nrows", i-1);
	
	err = clEnqueueNDRangeKernel(que, matinit_k, 2, NULL, gws, NULL, 0, NULL, &matinit_evt);
	ocl_check(err, "enqueue matinit");
	return matinit_evt;
}

void verify(const int *h_A, int nrows, int ncols, int pitch_el)
{
  for (int r = 0; r < nrows; ++r)
    for (int c = 0; c < ncols; ++c)
      if (h_A[r*pitch_el+c] != r-c)
      {
        fprintf(stderr, "mismatch @ (%d, %d) : %d != %d\n",
                r,c,h_A[r*pitch_el+c],r-c);
                exit(3);
      }
}

int main(int argc, char *argv[])
{
	if (argc <= 2) {
		fprintf(stderr, "specify number of rows and columns\n");
		exit(1);
	}

	const int nrows = atoi(argv[1]);
	const int ncols = atoi(argv[2]);
	

  cl_platform_id plat_id = select_platform();
	cl_device_id dev_id = select_device(plat_id);
	cl_context ctx = create_context(plat_id, dev_id);
	cl_command_queue que = create_queue(ctx, dev_id);
	cl_program prog = create_program("matinit.ocl", ctx, dev_id);
	cl_int err;

	cl_kernel matinit_k = clCreateKernel(prog, "matinit_pitch", &err);
	ocl_check(err, "create kernel matinit_pitch");

	err = clGetKernelWorkGroupInfo(matinit_k, dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
				sizeof(gws_align_init), &gws_align_init, NULL);
	ocl_check(err, "Preferred wg multiple for init");

  // get info about the base address

  cl_uint base_addr_align;
  err = clGetDeviceInfo(dev_id, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
      sizeof(base_addr_align), &base_addr_align, NULL);
  ocl_check(err, "base address alignment(bits)\n");
  base_addr_align/=8;   // Da bit a byte
  printf("base address align (bytes):%d\n", base_addr_align);

  base_addr_align /= sizeof(cl_int);  // Da Byte a num. di elementi
  // Verificare sempre che riottengo il numero di byte stesso
  printf("column align (num_of_elements):%d\n", base_addr_align);

  // matrix pitch in elements
  const int pitch_el = round_mul_up(ncols, base_addr_align);
  printf("ncols: %d pitch: %d\n",ncols, pitch_el);

  const size_t memsize = nrows*pitch_el*sizeof(cl_int);

  cl_mem d_A = NULL;
	d_A = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, memsize, NULL, &err);
	ocl_check(err, "create buffer d_v1");

	cl_event init_evt, read_evt;
	init_evt = matinit(matinit_k, que, d_A, nrows, ncols, pitch_el);
	
	cl_int * h_A = clEnqueueMapBuffer(que,d_A,CL_FALSE,CL_MAP_READ, 0,memsize,1,&init_evt, &read_evt , &err);
	clWaitForEvents(1, &read_evt);
	verify(h_A, nrows,ncols, pitch_el);

	const double runtime_init_ms = runtime_ms(init_evt);
	const double runtime_read_ms = runtime_ms(read_evt);

	const double init_bw_gbs = 1.0*memsize/1.0e6/runtime_init_ms;
	const double read_bw_gbs = memsize/1.0e6/runtime_read_ms;

	printf("init: %dx%d int in %gms: %g GB/s %g GE/s\n",
		nrows, ncols, runtime_init_ms, init_bw_gbs, nrows*ncols/1.0e6/runtime_init_ms);
	printf("read: %dx%d int in %gms: %g GB/s %g GE/s\n",
		nrows, ncols, runtime_read_ms, read_bw_gbs, nrows*ncols/1.0e6/runtime_read_ms);

	err = clEnqueueUnmapMemObject(que,d_A,h_A,0,NULL,NULL);
	ocl_check(err, "unmap matrix");
  clReleaseMemObject(d_A);
	clReleaseKernel(matinit_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);				
}