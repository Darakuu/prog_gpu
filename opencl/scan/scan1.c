#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler2.h"

size_t gws_align_init;
size_t gws_align_sum;

cl_event vecinit(cl_kernel vecinit_k, cl_command_queue que, cl_mem d_v1, cl_int nels)
{
  const size_t gws[] = { round_mul_up(nels, gws_align_init)};
  printf("init gws: %d | %zu = %zu\n", nels, gws_align_init, gws[0]);
  cl_event vecinit_evt;
  cl_int err;

  cl_uint i = 0;
  err = clSetKernelArg(vecinit_k, i++, sizeof(d_v1), &d_v1);
  ocl_check(err, "set vecinit arg dv1", i-1);
  err = clSetKernelArg(vecinit_k, i++, sizeof(nels), &nels);
  ocl_check(err, "set vecinit arg nels", i-1);

  err = clEnqueueNDRangeKernel(que, vecinit_k, 1,
    NULL, gws, NULL, 0, NULL, &vecinit_evt);
  ocl_check(err, "enqueue vecinit");

  return vecinit_evt;
}

cl_event scan1(cl_kernel scan1_k, cl_command_queue que,
  cl_mem d_vsum, cl_mem d_v1, cl_int nels,
  cl_int lws_, cl_int nwg, cl_event init_evt)
{
  const size_t gws[] = {nwg*lws_};
  const size_t lws[] = {lws_};
  cl_event scan1_evt;
  cl_int err;

  cl_uint i = 0;
  err = clSetKernelArg(scan1_k, i++, sizeof(d_vsum), &d_vsum);
  ocl_check(err, "set scan1 arg dvsum", i-1);
  err = clSetKernelArg(scan1_k, i++, sizeof(d_v1), &d_v1);
  ocl_check(err, "set scan1 arg dv1", i-1);
  err = clSetKernelArg(scan1_k, i++, sizeof(cl_int)*lws[0], NULL);
  ocl_check(err, "set scan1 arg lws", i-1);
  err = clSetKernelArg(scan1_k, i++, sizeof(nels), &nels);
  ocl_check(err, "set scan1 arg nels", i-1);

  err = clEnqueueNDRangeKernel(que, scan1_k, 1,
    NULL, gws, lws, 1, &init_evt, &scan1_evt);
  ocl_check(err, "enqueue scan1_lmem");

  err = clFinish(que);
  ocl_check(err,"finish que");

  return scan1_evt;
}

void verify(const int *vsum, int nels)
{
	int scan = 0;
	for (int i = 0; i < nels; ++i)
	{
		scan += (i+1);
		if (vsum[i] != scan)
		{
			fprintf(stderr, "mismatch %d @ %d != %d\n",vsum[i],i, scan);
			exit(3);
		}
	}
}

int main(int argc, char * argv[])
{
  if (argc <= 3)
  {
    fprintf(stderr, "Use: %s <number of elements> <lws> <nwg>\n",argv[0]);
    exit(1);
  }

  const int nels = atoi(argv[1]);
	const int lws = atoi(argv[2]);
	const int nwg = atoi(argv[3]);
	const size_t memsize = nels*sizeof(cl_int);
	//const size_t nwg_mem = nwg*sizeof(cl_int);

	//if (nels & 3) ocl_check(CL_INVALID_VALUE, "nels not multiple of 4");
	if (nwg != 1) ocl_check(CL_INVALID_VALUE, "nwg is not 1");
	if (lws & (lws-1)) ocl_check(CL_INVALID_VALUE, "lws is not a power of 2");

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("scan1.ocl", ctx, d);
	cl_int err;

	cl_kernel vecinit_k = clCreateKernel(prog, "vecinit", &err);
	ocl_check(err, "create kernel vecinit");
	cl_kernel scan1_k = clCreateKernel(prog, "scan1", &err);
	ocl_check(err, "create kernel scan1");

	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(vecinit_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_init), &gws_align_init, NULL);
	ocl_check(err, "preferred wg multiple for init");
	err = clGetKernelWorkGroupInfo(scan1_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_sum), &gws_align_sum, NULL);
	ocl_check(err, "preferred wg multiple for sum");

	cl_mem d_v1 = NULL, d_v2 = NULL;

	d_v1 = clCreateBuffer(ctx,
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		memsize, NULL,
		&err);
	ocl_check(err, "create buffer d_v1");

	d_v2 = clCreateBuffer(ctx,
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,	//ALLOC_HOST for mapping purposes
		memsize, NULL,
		&err);
	ocl_check(err, "create buffer d_v2");

	cl_event init_evt, scan_evt[2], read_evt;

	init_evt = vecinit(vecinit_k, que, d_v1, nels);

  int nquarts = nels/4;

  // riduco datasize a nwg elementi:
  scan_evt[0] = scan1(scan1_k,que,d_v2,d_v1,nquarts,lws,nwg,init_evt);

  int *risultato = clEnqueueMapBuffer(que,d_v2,CL_TRUE,CL_MAP_READ,0,memsize,1,scan_evt,&read_evt,&err);
  err = clEnqueueReadBuffer(que, d_v2, CL_TRUE, 0, sizeof(risultato), &risultato,
		1, scan_evt + 1, &read_evt);
	ocl_check(err, "read result");
  verify(risultato, nels);

	const double runtime_init_ms = runtime_ms(init_evt);
	const double runtime_read_ms = runtime_ms(read_evt);

	const double init_bw_gbs = 1.0*memsize/1.0e6/runtime_init_ms;
	const double read_bw_gbs = sizeof(float)/1.0e6/runtime_read_ms;

	printf("init: %d int in %gms: %g GB/s %g GE/s\n",
		nels, runtime_init_ms, init_bw_gbs, nels/1.0e6/runtime_init_ms);

	{
		const double runtime_pass_ms = runtime_ms(scan_evt[0]);
		const double pass_bw_gbs = (memsize+memsize)/1.0e6/runtime_pass_ms;
		printf("reduce0 : %d float in %gms: %g GB/s %g GE/s\n",
			nels, runtime_pass_ms, pass_bw_gbs,
			nels/1.0e6/runtime_pass_ms);
	}
  /*
	if (nwg > 1)
	{
		const double runtime_pass_ms = runtime_ms(scan_evt[1]);
		const double pass_bw_gbs = (nwg_mem+sizeof(cl_int))/1.0e6/runtime_pass_ms;
		printf("reduce1 : %d float in %gms: %g GB/s %g GE/s\n",
			(lws*nwg), runtime_pass_ms, pass_bw_gbs,
			(lws*nwg)/1.0e6/runtime_pass_ms);
	}
*/
	const double runtime_reduction_ms = total_runtime_ms(scan_evt[0], scan_evt[1]);
	printf("reduce : %d float in %gms: %g GE/s\n",
		nels, runtime_reduction_ms, nels/1.0e6/runtime_reduction_ms);

	printf("read: 1 int in %gms: %g GB/s %g GE/s\n",
		runtime_read_ms, read_bw_gbs, memsize/1.0/1.0e6/runtime_read_ms);

	err = clEnqueueUnmapMemObject(que, d_v2, risultato, 0, NULL, NULL);
	ocl_check(err, "unmap risultato");
	clReleaseMemObject(d_v2);
	clReleaseMemObject(d_v1);
	clReleaseKernel(scan1_k);
	clReleaseKernel(vecinit_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}