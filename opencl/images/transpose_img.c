#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define CL_TARGET_OPENCL_VERSION 120
#include "../ocl_boiler.h"

size_t gws_align_init;
size_t gws_align_transpose;		


cl_event matinit(cl_kernel matinit_k, cl_command_queue que, 
									cl_mem d_A, cl_int nrows, cl_int ncols)
{
	const size_t gws[] = {round_mul_up(ncols, gws_align_init), nrows};  			// Al contrario solo per mantenere la coalescenza
	cl_event matinit_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(matinit_k, i++, sizeof(d_A), &d_A);
	ocl_check(err, "set matinit arg_dA", i-1);
	err = clSetKernelArg(matinit_k, i++, sizeof(ncols), &ncols);
	ocl_check(err, "set matinit arg_ncols", i-1);
	err = clSetKernelArg(matinit_k, i++, sizeof(nrows), &nrows);
	ocl_check(err, "set matinit arg_nrows", i-1);
	
	err = clEnqueueNDRangeKernel(que, matinit_k, 2, NULL, gws, NULL, 0, NULL, &matinit_evt);
	ocl_check(err, "enqueue matinit");
	return matinit_evt;
}

cl_event transpose(cl_kernel transpose_k, cl_command_queue que, 
									cl_mem d_T, cl_mem d_I, cl_int nrows_T, cl_int ncols_T, cl_event init_evt)
{
	const size_t gws[] = {round_mul_up(ncols_T, gws_align_init), nrows_T};
	cl_event trans_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(transpose_k, i++, sizeof(d_T), &d_T);
	ocl_check(err, "set tranpose arg_dT", i-1);
  err = clSetKernelArg(transpose_k, i++, sizeof(d_I), &d_I);
	ocl_check(err, "set tranpose arg_dA", i-1);
	// Tecnicamente non servono righe e colonne perché sono nella struct

	err = clEnqueueNDRangeKernel(que, transpose_k, 2, NULL, gws, NULL, 1, &init_evt, &trans_evt);
	ocl_check(err, "enqueue transpose");
	return trans_evt;
}

void verify(const int *h_T, int nrows_T, int ncols_T)
{
  for (int r = 0; r < nrows_T; ++r)
    for (int c = 0; c < ncols_T; ++c)
      if (h_T[r*ncols_T+c] != c-r)
      {
        fprintf(stderr, "mismatch = (%d, %d) : %d != %d\n",
                r,c,h_T[r*ncols_T+c],c-r);
                exit(3);
      }
}

int main(int argc, char *argv[])
{
	if (argc <= 2) {
		fprintf(stderr, "specify number of rows and columns\n");
		exit(1);
	}

	const int nrows_A = atoi(argv[1]);
	const int ncols_A = atoi(argv[2]);

	const char *trans_kernel_name = "transpose_img";

	const size_t memsize = nrows_A*ncols_A*sizeof(cl_int);
  const int nrows_T = ncols_A;
  const int ncols_T = nrows_A;

  cl_platform_id plat_id = select_platform();
	cl_device_id dev_id = select_device(plat_id);
	cl_context ctx = create_context(plat_id, dev_id);
	cl_command_queue que = create_queue(ctx, dev_id);
	cl_program prog = create_program("transpose_img.ocl", ctx, dev_id);
	cl_int err;

	cl_kernel matinit_k = clCreateKernel(prog, "matinit", &err);
	ocl_check(err, "create kernel matinit");
	cl_kernel transpose_k = clCreateKernel(prog, trans_kernel_name, &err);
	ocl_check(err, "create kernel transpose");

	err = clGetKernelWorkGroupInfo(matinit_k, dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
				sizeof(gws_align_init), &gws_align_init, NULL);
	err = clGetKernelWorkGroupInfo(transpose_k, dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
				sizeof(gws_align_transpose), &gws_align_transpose, NULL);
	ocl_check(err, "Preferred wg multiple for init");

	cl_mem d_A = NULL, d_T = NULL;
	cl_mem d_I = NULL; // image

	d_A = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, memsize, NULL, &err);
	ocl_check(err, "create buffer d_A");
  
	d_T = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, memsize, NULL, &err);
	ocl_check(err, "create buffer d_T");

	const cl_image_format fmt = {
			.image_channel_order = CL_R,	// Order in which data is interpreted / returned. IE: RGB, RGBA, ARGB... in this case x=R
			.image_channel_data_type = CL_SIGNED_INT32  
			};
	const cl_image_desc desc = {
			.image_type = CL_MEM_OBJECT_IMAGE2D,
			.image_width = ncols_A,
			.image_height =	nrows_A
	};
	d_I = clCreateImage(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,&fmt,&desc,NULL,&err);
	ocl_check(err, "create image d_I");

	cl_event init_evt, copy_evt, trans_evt, read_evt;	// copy_evt = inefficiente
	init_evt = matinit(matinit_k, que, d_A, nrows_A, ncols_A);
	
	const size_t img_origin[3] = {0,0,0};
	const size_t img_region[3] = {ncols_A, nrows_A, 1};
	err = clEnqueueCopyBufferToImage(que,
		d_A, d_I,
		0 /*offset src(d_A) buffer*/,
		img_origin	/*offset dest(d_I) buffer*/,
		img_region/*region*/,
		1, &init_evt, &copy_evt
		);
	ocl_check(err, "Copy d_A in d_I");

	trans_evt = transpose(transpose_k, que, d_T, d_A, nrows_T, ncols_T, copy_evt);
	
	cl_int * h_T = clEnqueueMapBuffer(que,d_T,CL_FALSE,CL_MAP_READ, 0,memsize,1,&trans_evt, &read_evt , &err);
	clWaitForEvents(1, &read_evt);
	verify(h_T, nrows_T,ncols_T);

	const double runtime_init_ms = runtime_ms(init_evt);
	const double runtime_copy_ms = runtime_ms(trans_evt);
  const double runtime_trans_ms = runtime_ms(trans_evt);
	const double runtime_read_ms = runtime_ms(read_evt);

	const double init_bw_gbs = 1.0*memsize/1.0e6/runtime_init_ms;
	const double copy_bw_gbs = 2.0*memsize/1.0e6/runtime_init_ms;
  const double trans_bw_gbs = 1.0*memsize/1.0e6/runtime_init_ms;
	const double read_bw_gbs = memsize/1.0e6/runtime_read_ms;

	printf("init: %dx%d int in %gms: %g GB/s %g GE/s\n",
		nrows_A, ncols_A, runtime_init_ms, init_bw_gbs, nrows_A*ncols_A/1.0e6/runtime_init_ms);
  printf("copy: %dx%d int in %gms: %g GB/s %g GE/s\n",
		nrows_T, ncols_T, runtime_copy_ms, copy_bw_gbs, nrows_T*ncols_T/1.0e6/runtime_copy_ms);
	printf("transpose: %dx%d int in %gms: %g GB/s %g GE/s\n",
		nrows_T, ncols_T, runtime_init_ms, init_bw_gbs, nrows_T*ncols_T/1.0e6/runtime_init_ms);
	printf("read: %dx%d int in %gms: %g GB/s %g GE/s\n",
		nrows_T, ncols_T, runtime_read_ms, read_bw_gbs, nrows_T*ncols_T/1.0e6/runtime_read_ms);

	err = clEnqueueUnmapMemObject(que,d_T,h_T,0,NULL,NULL);
	ocl_check(err, "unmap T");
	clReleaseMemObject(d_T);
  clReleaseMemObject(d_A);
	clReleaseKernel(transpose_k);
	clReleaseKernel(matinit_k);	
	clReleaseProgram(prog);	
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);		
}