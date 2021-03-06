#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <stdarg.h>
#include <cuda_runtime_api.h>
#include "pamalign.h"

using namespace std;

#define BUFSIZE 4096

void cuda_check(cudaError_t err, const char *msg, ...) 
{
  if (err != cudaSuccess) 
  {
		char msg_buf[BUFSIZE + 1];
		va_list ap;
		va_start(ap, msg);
		vsnprintf(msg_buf, BUFSIZE, msg, ap);
		va_end(ap);
		msg_buf[BUFSIZE] = '\0';
		fprintf(stderr, "%s - error %d (%s)\n", msg_buf, err, cudaGetErrorString(err));
		exit(1);
	}
}

//texture is a template
texture<uchar4, 2,cudaReadModeNormalizedFloat> tex;

__global__
void imgcopy(uchar4 * out, int width, int height, int output_pitch_el)
{
  int row = blockDim.y*blockIdx.y + threadIdx.y;
  int col = blockDim.x*blockIdx.x + threadIdx.x;

  if (row < height && col < width)
  {
    float4 px = tex2D(tex, col, row);
    out[row*output_pitch_el+col] = make_uchar4(px.x*255, px.y*255, px.z*255, px.w*255);
  }
}

int main(int argc, char *argv[])
{
  if (argc <= 1) 
  {
    fprintf(stderr, "specify name of input file\n");
    exit(1);
  }

  const char *input_fname = argv[1];
  const char *output_fname = "copia.pam";

  struct imgInfo src; 
  struct imgInfo dst;
  
  cudaError_t err;
  int pam_err = load_pam(input_fname, &src);

  if (pam_err != 0)
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
    fprintf(stderr, "source must have 8-bit channels\n");
    exit(1);
  }

  dst = src; // THIS COPIES &data AS WELL
  dst.data = NULL;

  uchar4 *d_input, *d_output;
  size_t input_pitch, output_pitch;
  int output_pitch_el;
  
  size_t src_pitch = src.data_size/src.height;
  size_t dst_pitch = src_pitch; // only applies in this case

  err = cudaMallocPitch(&d_input, &input_pitch, src_pitch, src.height);
  cuda_check(err, "alloc d_input");
  err = cudaMallocPitch(&d_output, &output_pitch, dst_pitch, dst.height);
  cuda_check(err, "alloc d_output");

  printf("pitch: %zu -> %zu\n", src_pitch, output_pitch);
  output_pitch_el = output_pitch/sizeof(uchar4);

  dim3 blockSize;
  blockSize.x = blockSize.y = 16;
  blockSize.z = 1;
  
  dim3 numBlocks;
  numBlocks.x = (src.width + blockSize.x -1)/blockSize.x; // Round up
  numBlocks.y = (src.height + blockSize.y -1)/blockSize.y;

  cudaEvent_t pre_upload, post_upload;
  cudaEvent_t pre_download, post_download;
  cudaEvent_t pre_copy, post_copy;
  
  err = cudaEventCreate(&pre_upload, 0);
  cuda_check(err, "pre_upload event create");
  err = cudaEventCreate(&post_upload, 0);
  cuda_check(err, "post_upload event create");
  
  err = cudaEventCreate(&pre_copy, 0);
  cuda_check(err, "pre_copy event create");
  err = cudaEventCreate(&post_copy, 0);
  cuda_check(err, "post_copy event create");
  
  err = cudaEventCreate(&pre_download, 0);
  cuda_check(err, "pre_download event create");
  err = cudaEventCreate(&post_download, 0);
  cuda_check(err, "post_download event create");
  
  err = cudaEventRecord(pre_upload);
  cuda_check(err, "record pre_upload");
  cudaMemcpy2D(d_input, input_pitch, src.data, src_pitch, src_pitch, src.height, cudaMemcpyHostToDevice);
  err = cudaEventRecord(post_upload);
  cuda_check(err, "record post_upload");

  err = cudaBindTexture2D(NULL, tex, d_input, tex.channelDesc, src.width, src.height, input_pitch);
  cuda_check(err, "texture binding");
  
  err = cudaEventRecord(pre_copy);
  cuda_check(err, "record pre_copy");
  imgcopy<<< numBlocks, blockSize>>>(d_output, src.width, src.height, output_pitch_el);
  err = cudaEventRecord(post_copy);
  cuda_check(err, "record post_copy");
  
  err = cudaHostAlloc(&dst.data, dst.data_size, cudaHostAllocPortable);
  cuda_check(err, "host alloc");
  
  err = cudaEventRecord(pre_download);
  cuda_check(err, "record pre_download");
  err = cudaMemcpy2D(dst.data, dst_pitch, d_output, output_pitch, dst_pitch/*width in byte*/, dst.height, cudaMemcpyDeviceToHost);
  cuda_check(err, "download output");
  err = cudaEventRecord(post_download);
  cuda_check(err, "record post_download");    

  err = cudaEventSynchronize(post_download);
  cuda_check (err, "sync post_download");

  float upload_time, copy_time, download_time;
  err = cudaEventElapsedTime(&upload_time, pre_upload, post_upload);
  cuda_check(err, "get upload time");
  err = cudaEventElapsedTime(&copy_time, pre_copy, post_copy);
  cuda_check(err, "get copy time");
  err = cudaEventElapsedTime(&download_time, pre_download, post_download);
  cuda_check(err, "get download time");


  printf("upload: %dx%d els in %6.4gms: %6.4gGB/s\n",
    src.width, src.height, upload_time, src.data_size/upload_time/1.0e6);
  printf("copy: %dx%d els in %6.4gms: %6.4gGB/s\n",
    src.width, src.height, copy_time, 2*src.data_size/copy_time/1.0e6);
  printf("download: %dx%d els in %6.4gms: %6.4gGB/s\n",
    src.width, src.height, download_time, src.data_size/download_time/1.0e6);

  pam_err = save_pam(output_fname, &dst);
  if (pam_err != 0) 
    fprintf(stderr, "error writing %s\n", output_fname), exit(1);

  cudaEventDestroy(pre_upload);
  cudaEventDestroy(post_upload);
  cudaEventDestroy(pre_copy);
  cudaEventDestroy(post_copy);
  cudaEventDestroy(pre_download);
  cudaEventDestroy(post_download);
  cudaFreeHost(dst.data);
  cudaFree(d_output); d_output = NULL;
  cudaFree(d_input); d_input = NULL;
}