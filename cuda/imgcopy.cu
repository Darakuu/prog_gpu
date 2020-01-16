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

__host__ __device__   // NVCC compiles this code for both device and host
int4 operator+(int4 const & a, int4 const & b)
{
  return make_int4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__global__
void vecinit(int * __restrict__ v1, int * __restrict__ v2, int nels)
{
  // threadIdx is a int3 index (xyz). Local workitem index <=> get_local_id()
  // blockIdx is the workgroup's index in the launch grid. <=> get_group_id()
  // blockDim is the workgroup dimension                   <=> get_group_size()
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  v1[i] = i;
	v2[i] = nels - i;
}

// Performance normale
__global__
void vecsum(int *__restrict__ vsum, const int *__restrict__ v1, const int *__restrict__ v2, int nels)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=nels) return;
	vsum[i] = v1[i] + v2[i];
}


void verify(const int *vsum, int nels)
{
	for (int i = 0; i < nels; ++i) 
		if (vsum[i] != nels) 
			fprintf(stderr, "mismatch @ %d : %d != %d\n", i, vsum[i], nels), exit(3);
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
    float4 px = tex20(tex, col, row);
    out[row*output_pitch_el+col] = make_uchar4(px.x*255, px.y*255, px.z*255, px.w*255);
  }
}

int main(int argc, char *argv[])
{
  int main(int argc, char *argv[])
  {
    if (argc <= 1) 
    {
      fprintf(stderr, "specify name of file\n");
      exit(1);
    }
  
    const char *input_fname = argv[1];
    const char *output_fname = "copia.pam";
  
    struct imgInfo src; 
    struct imgInfo dst;
    cudaError_t err;
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
    dst.data = NULL;

    cudaArray_t d_input;
    uchar4 *d_output;
    size_t src_pitch = src.data_size/src.height;
    size_t output_pitch;
    int output_pitch_el;

    err = cudaMallocArray(&d_input, &tex.channelDesc, src.width, src.height);
    cuda_check(err, "alloc d_input");
    err = cudaMallocPitch(&d_output, &output_pitch, dst_pitch, dst_height);
    cuda_check(err, "alloc d_output");

    prtinf("pitch: %zu -> %zu\n", src_pitch, output_pitch);
    output_pitch_el = output_pitch/sizeof(uchar4);

    dim3 blockSize;
    blockSize.x = blockSize.y = 16;
    blockSize.z = 1;
    
    dim3 numBlocks;
    numBlocks.x = (src.width + blockSize.x -1)/blockSize.x // Round up
    numBlocks.y = (src.height + blockSize.y -1)/blockSize.y

    cudaEvent_t pre_upload, post_upload;
    cudaEvent_t pre_download, post_download;
    cudaEvent_t pre_copy, post_copy;
    
    err = cudaEventCreate(&pre_upload, 0);
    cuda_check(err, "pre_upload event create");
    cudaMemcpy2DToArray(d_input, 0, 0, src.data, src_pitch, src_pitch, src.height, cudaMemcpyHostToDevice);
    err = cudaEventCreate(&post_upload, 0);
    cuda_check(err, "post_upload event create");

    err = cudaBindTextureToArray(tex,d_input, tex.channelDesc);
    cuda_check(err, "texture binding");

    err = cudaEventCreate(&pre_copy, 0);
    cuda_check(err, "pre_copy event create");
    imgcopy<<< numBlocks, blockSize>>>(d_output, src.width, , cudaMemcpyHostToDevice);  // TODO
    err = cudaEventCreate(&post_copy, 0);
    cuda_check(err, "post_copy event create");
    
    err = cudaHostAlloc(&dst.data, dst.data_size, cudaHostAllocPortable);
    cuda_check(err, "host alloc")

    err = cudaEventCreate(&pre_download, 0);
    cuda_check(err, "pre_download event create");
    cudaMemcpy2D(dst.data, dst_pitch, d_output, output_pitch, dst_pitch/*width in byte*/, dst_height, cudaMemcpyDeviceToHost);
    err = cudaEventCreate(&post_download, 0);
    cuda_check(err, "post_download event create");

    err = cudaEventSynchronize(post_download);
    cuda_check (err, "sync post_download");

    float upload_time, copy_time, download_time;

    printf("upload: %dx%d els in &6.4gms: %6.4gGB/s\n", src.width, src.height, upload_time, data_size/uploadtime/1.0e6);

    cudaEventDestroy(pre_upload);

    pam_err = save_pam();

  }