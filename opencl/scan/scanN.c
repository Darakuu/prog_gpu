#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler2.h"

size_t gws_align_init;
size_t gws_align_sum;

cl_event vecinit(cl_kernel vecinit_k, cl_command_queue que,
	cl_mem d_v1, cl_int nels)
{
	const size_t gws[] = { round_mul_up(nels, gws_align_init) };
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
	cl_mem d_vsum, cl_mem d_tails, cl_mem d_v1, cl_int nels,
	cl_int lws_, cl_int nwg,
	cl_event init_evt)
{
	const size_t gws[] = { nwg*lws_ };
	const size_t lws[] = { lws_ };
	cl_event scan1_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(scan1_k, i++, sizeof(d_vsum), &d_vsum);
	ocl_check(err, "set scan1 arg dvsum", i-1);
	if (nwg > 1) 
  {
		err = clSetKernelArg(scan1_k, i++, sizeof(d_tails), &d_tails);
		ocl_check(err, "set scan1 arg dtails", i-1);
	}

	err = clSetKernelArg(scan1_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set scan1 arg dv1", i-1);
	err = clSetKernelArg(scan1_k, i++, sizeof(cl_int)*lws[0], NULL);
	ocl_check(err, "set scan1 arg lws", i-1);
	err = clSetKernelArg(scan1_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set scan1 arg nels", i-1);

	err = clEnqueueNDRangeKernel(que, scan1_k, 1,
		NULL, gws, lws, 1, &init_evt, &scan1_evt);
	ocl_check(err, "enqueue scan_lmem");
	err = clFinish(que);
	ocl_check(err, "finish que");
	return scan1_evt;
}

cl_event fixup(cl_kernel fixup_k, cl_command_queue que,
	cl_mem d_vsum, cl_mem d_tails, cl_int nels,
	cl_int lws_, cl_int nwg,
	cl_event init_evt)
{
	const size_t gws[] = { nwg*lws_ };
	const size_t lws[] = { lws_ };
	cl_event fixup_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(fixup_k, i++, sizeof(d_vsum), &d_vsum);
	ocl_check(err, "set fixup arg dvsum", i-1);
	err = clSetKernelArg(fixup_k, i++, sizeof(d_tails), &d_tails);
	ocl_check(err, "set fixup arg dtails", i-1);
	err = clSetKernelArg(fixup_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set fixup arg nels", i-1);

	err = clEnqueueNDRangeKernel(que, fixup_k, 1,
		NULL, gws, lws, 1, &init_evt, &fixup_evt);
	ocl_check(err, "enqueue scan_lmem");
	err = clFinish(que);
	ocl_check(err, "finish que");
	return fixup_evt;
}

void verify(const int *vsum, int nels)
{
	int scan = 0;
	for (int i = 0; i < nels; ++i) 
  {
		scan += (i+1);
		if (vsum[i] != scan) 
    {
			fprintf(stderr, "mismatch @ %d : %d != %d\n", i, vsum[i], scan);
			exit(3);
		}
	}
}


int main(int argc, char *argv[])
{
	if (argc <= 3) 
  {
		fprintf(stderr, "specify number of elements, lws, nwg\n");
		exit(1);
	}

	const int nels = atoi(argv[1]);
	const int lws = atoi(argv[2]);
	const int nwg = atoi(argv[3]);
	const size_t memsize = nels*sizeof(cl_int);
	const size_t nwg_mem = nwg*sizeof(cl_int);

	if (lws & (lws-1)) ocl_check(CL_INVALID_VALUE, "lws");

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("scanN.ocl", ctx, d);
	cl_int err;

	cl_kernel vecinit_k = clCreateKernel(prog, "vecinit", &err);
	ocl_check(err, "create kernel vecinit");
	cl_kernel scan1_k = clCreateKernel(prog, "scan1_lmem", &err);
	ocl_check(err, "create kernel scan1_lmem");
	cl_kernel scanN_k = clCreateKernel(prog, "scanN_lmem", &err);
	ocl_check(err, "create kernel scanN_lmem");
	cl_kernel fixup_k = clCreateKernel(prog, "scanN_fixup", &err);
	ocl_check(err, "create kernel scanN_fixup");

	/* get information about the preferred work-group size multiple */
	err = clGetKernelWorkGroupInfo(vecinit_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_init), &gws_align_init, NULL);
	ocl_check(err, "preferred wg multiple for init");
	err = clGetKernelWorkGroupInfo(scan1_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(gws_align_sum), &gws_align_sum, NULL);
	ocl_check(err, "preferred wg multiple for sum");

	cl_mem d_v1 = NULL, d_v2 = NULL, d_tails = NULL;

	d_v1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, memsize, NULL, &err);
	ocl_check(err, "create buffer d_v1");

	d_tails = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nwg_mem, NULL, &err);
	ocl_check(err, "create buffer d_tails");

	d_v2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
	ocl_check(err, "create buffer d_v2");

	cl_event init_evt, scan_evt[3], read_evt;

	init_evt = vecinit(vecinit_k, que, d_v1, nels);

	// riduco il datasize originale ad nwg elementi
	scan_evt[0] = scan1(nwg > 1 ? scanN_k : scan1_k, que, d_v2, d_tails, d_v1, nels,
		lws, nwg, init_evt);
	if (nwg > 1) 
  {
		scan_evt[1] = scan1(scan1_k, que, d_tails, NULL, d_tails, nwg,
			lws, 1, scan_evt[0]);
		scan_evt[2] = fixup(fixup_k, que, d_v2, d_tails, nels, lws, nwg,
			scan_evt[1]);
	} 
  else 
		scan_evt[2] = scan_evt[1] = scan_evt[0];

	int *risultato = clEnqueueMapBuffer(que, d_v2, CL_TRUE,
		CL_MAP_READ, 0, memsize,
		1, scan_evt, &read_evt, &err);
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
		printf("scan0 : %d float in %gms: %g GB/s %g GE/s\n",
			nels, runtime_pass_ms, pass_bw_gbs,
			nels/1.0e6/runtime_pass_ms);
	}

	if (nwg > 1)
	{
		const double runtime_pass_ms = runtime_ms(scan_evt[1]);
		const double pass_bw_gbs = (nwg_mem + nwg_mem)/1.0e6/runtime_pass_ms;
		printf("scan1 : %d float in %gms: %g GB/s %g GE/s\n",
			nwg, runtime_pass_ms, pass_bw_gbs,
			nwg/1.0e6/runtime_pass_ms);
	}
	if (nwg > 1)
	{
		const double runtime_pass_ms = runtime_ms(scan_evt[2]);
		const double pass_bw_gbs = (memsize + nwg_mem + memsize)/1.0e6/runtime_pass_ms;
		printf("scan2 : %d float in %gms: %g GB/s %g GE/s\n",
			nels, runtime_pass_ms, pass_bw_gbs,
			nels/1.0e6/runtime_pass_ms);
	}


	const double runtime_reduction_ms = total_runtime_ms(scan_evt[0], scan_evt[2]);
	printf("scan : %d float in %gms: %g GE/s\n",
		nels, runtime_reduction_ms, nels/1.0e6/runtime_reduction_ms);

	printf("read: %d int in %gms: %g GB/s %g GE/s\n", nels,
		runtime_read_ms, read_bw_gbs, memsize/1.0e6/runtime_read_ms);

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