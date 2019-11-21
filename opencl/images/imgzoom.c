#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#include "../ocl_boiler.h"
#include "pamalign.h"

size_t gws_align_imgzoom;

//nrows and ncols are taken from SOURCE image
cl_event imgzoom(cl_kernel imgzoom_k, cl_command_queue que, 
									cl_mem d_output, cl_mem d_input, cl_int nrows, cl_int ncols)
{
	const size_t gws[] = {round_mul_up(ncols, gws_align_imgzoom), nrows};  			// Al contrario solo per mantenere la coalescenza
	cl_event imgzoom_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(imgzoom_k, i++, sizeof(d_output), &d_output);
	ocl_check(err, "set imgzoom arg_dv1", i-1);
  err = clSetKernelArg(imgzoom_k, i++, sizeof(d_input), &d_input);
	ocl_check(err, "set imgzoom arg_dv1", i-1);
	
	err = clEnqueueNDRangeKernel(que, imgzoom_k, 2, NULL, gws, NULL, 0, NULL, &imgzoom_evt);
	ocl_check(err, "enqueue imgzoom");
	return imgzoom_evt;
}

int main(int argc, char *argv[])
{
	if (argc <= 1) {
		fprintf(stderr, "specify name of file\n");
		exit(1);
	}

  const char *input_fname = argv[1];
  const char *output_fname = "copia.pam";

  struct imgInfo src, dst;
  cl_int err = load_pam(input_fname, &src);
  if (err !=0)
  {
    fprintf(stderr, "error loading %s\n", input_fname);
		exit(1);
  }

  if (src.channels != 4)
  {
    fprintf(stderr, "source must have 4 channels\n");
		exit(1);
  }

  if (src.depth != 8)
  {
    fprintf(stderr, "source must have depth 8\n");
		exit(1);
  }
  dst = src; // THIS COPIES &data AS WELL
  dst.width     *= 2;
  dst.height    *= 2;
  dst.data_size *= 4;
  dst.data = NULL;

  
  
  cl_platform_id plat_id = select_platform();
	cl_device_id dev_id = select_device(plat_id);
	cl_context ctx = create_context(plat_id, dev_id);
	cl_command_queue que = create_queue(ctx, dev_id);
	cl_program prog = create_program("imgzoom.ocl", ctx, dev_id);
	cl_int err;
	cl_kernel imgzoom_k = clCreateKernel(prog, "imgzoom", &err);
	ocl_check(err, "create kernel imgzoom");

	err = clGetKernelWorkGroupInfo(imgzoom_k, dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
				sizeof(gws_align_imgzoom), &gws_align_imgzoom, NULL);
	ocl_check(err, "Preferred wg multiple for init");
  
	cl_mem d_input = NULL, d_out= NULL;
	const cl_image_format fmt = 
  {
    .image_channel_order = CL_RGBA,
    .image_channel_data_type = CL_UNORM_INT8
  };

  const cl_image_desc desc =
  {
    .image_type = CL_MEM_OBJECT_IMAGE2D,
    .image_width = src.width,
    .image_height = src.height,
  };
  
  d_input = clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY| CL_MEM_USE_HOST_PTR, &fmt, &desc, src.data, &err);
	ocl_check(err, "create image d_input");
  d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, dst.data_size,NULL, &err);
  ocl_check(err,"create buffer d_out");

	cl_event imgzoom_evt, map_evt;
	imgzoom_evt = imgzoom(imgzoom_k, que, d_out, d_input, src.height, src.width);
	
	dst.data = clEnqueueMapBuffer(que,d_out,CL_FALSE,CL_MAP_READ, 0, dst.data_size,1,&imgzoom_evt, &map_evt , &err);
  ocl_check(err, "enqueue map d_output");
	clWaitForEvents(1, &map_evt);
  ocl_check(err, "clfinish");
	

	const double runtime_imgzoom_ms = runtime_ms(imgzoom_evt);
	const double runtime_map_ms = runtime_ms(map_evt);

	const double imgzoom_bw_gbs = (2*dst.data_size)/1.0e6/runtime_imgzoom_ms;
	const double map_bw_gbs = dst.data_size/1.0e6/runtime_map_ms;

	printf("imgzoom: %dx%d int in %gms: %g GB/s %g GE/s\n",
		src.height, src.width, runtime_imgzoom_ms, imgzoom_bw_gbs, src.height*src.width/1.0e6/runtime_imgzoom_ms);
	printf("map: %dx%d int in %gms: %g GB/s %g GE/s\n",
		dst.height, dst.width, runtime_map_ms, map_bw_gbs, dst.height*dst.width/1.0e6/runtime_map_ms);

  err = save_pam(output_fname, &dst);
  if (err !=0)
  {
    fprintf(stderr, "error writing %s\n", output_fname);
		exit(1);
  }
	err = clEnqueueUnmapMemObject(que,d_out,dst.data,0,NULL,NULL);
	ocl_check(err, "unmap output");
  clReleaseMemObject(d_out);
  clReleaseMemObject(d_input);
	clReleaseKernel(imgzoom_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);				
/**/
}