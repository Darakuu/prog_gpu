#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <stdarg.h>
#include <cuda_runtime_api.h>

using namespace std;

#define BUFSIZE 4096

void cuda_check(cudaError_t err, const char *msg, ...) {
	if (err != cudaSuccess) {
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
  return make_int(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
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
  
  int *d_v1 = NULL, *d_v2 = NULL, *d_vsum = NULL;

  size_t memsize = nels*sizeof(*d_v1);
  
  cudaError_t err;

  err = cudaMalloc(&d_v1, memsize); // cudaMalloc returns a device pointer
  cuda_check(err,"alloc v1");
  err = cudaMalloc(&d_v2, memsize);
  cuda_check(err,"alloc v2");
  err = cudaMalloc(&d_vsum, memsize);
  cuda_check(err,"alloc vsum");
  
  int blockSize = 256;
  int numBlocks = (nels + blockSize - 1)/blockSize;

  printf("%d blocchi di %d threads\n",numBlocks,blockSize);

  cudaEvent_t pre_init, post_init;
  cudaEvent_t pre_sum, post_sum;
  cudaEvent_t pre_copy, post_copy;

  err = cudaEventCreate(&pre_init, 0);
  cuda_check(err, "pre_init event create");
  err = cudaEventCreate(&post_init, 0);
  cuda_check(err, "post_init event create");
  err = cudaEventCreate(&pre_sum, 0);
  cuda_check(err, "pre_sum event create");
  err = cudaEventCreate(&post_sum, 0);
  cuda_check(err, "post_sum event create");
  err = cudaEventCreate(&pre_copy, 0);
  cuda_check(err, "post_copy event create");
  err = cudaEventCreate(&post_copy, 0);
  cuda_check(err, "post_copy event create");

  err = cudaEventRecord(pre_init);
  cuda_check(err, "pre_init record");
  vecinit<<< numBlocks, blockSize >>>(d_v1,d_v2,nels);
  err = cudaEventRecord(post_init);
  cuda_check(err, "post_init record");

  err = cudaEventRecord(pre_sum);
  cuda_check(err, "pre_sum record");
  vecsum<<< numBlocks, blockSize >>>(d_vsum,d_v1,d_v2,nels);
  err = cudaEventRecord(post_sum);
  cuda_check(err, "post_sum record");

  int * h_vsum;
  err = cudaHostAlloc(&h_vsum, memsize, cudaHostAllocPortable); //DMA


  if (!h_vsum)
  {
    fprintf(stderr, "out of memory on host!\n");
    exit(1);
  }

  err = cudaEventRecord(pre_copy);
  cuda_check(err, "pre_copy record");
  err = cudaMemcpy(h_vsum,d_vsum,memsize, cudaMemcpyDeviceToHost);
  cuda_check(err, "copy vsum");
  err = cudaEventRecord(post_copy);
  cuda_check(err, "post_copy record");

	verify(h_vsum, nels);

  cudaEventSynchronize(post_copy);
  cuda_check(err, "sync post_copy");

  float init_time, sum_time, copy_time;
  err = cudaEventElapsedTime(&init_time, pre_init, post_init);  //in ms
  cuda_check(err, "get init time");
  err = cudaEventElapsedTime(&sum_time, pre_sum, post_sum);  //in ms
  cuda_check(err, "get init time");
  err = cudaEventElapsedTime(&copy_time, pre_copy, post_copy);  //in ms
  cuda_check(err, "get init time");

  printf("init: %d els in &6.4gms: %6.4gGB/s, %6.4gGE/s\n", nels, init_time,2*memsize/init_time/1.0e6, nels/init_time/1.0e6);
  printf("sum: %d els in &6.4gms: %6.4gGB/s, %6.4gGE/s\n", nels, sum_time,3*memsize/sum_time/1.0e6, nels/sum_time/1.0e6);
  printf("copy: %d els in &6.4gms: %6.4gGB/s, %6.4gGE/s\n", nels, copy_time,memsize/copy_time/1.0e6, nels/copy_time/1.0e6);

  cudaEventDestroy(pre_init);
  cudaEventDestroy(post_init);
  cudaEventDestroy(pre_sum);
  cudaEventDestroy(post_sum);
  cudaEventDestroy(pre_copy);
  cudaEventDestroy(post_copy);

  
  cudaFreeHost(h_vsum);     h_vsum = NULL;
  cudaFree(d_vsum); d_vsum = NULL;
  cudaFree(d_v2);   d_v2 = NULL;
  cudaFree(d_v1);   d_v1 = NULL;
}