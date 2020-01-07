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
  // blockDim is the workgroup dimensione                  <=> get_group_size()
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  v1[i] = i;
	v2[i] = nels - i;
}

// Performance normale
__global__
void vecsum(int *__restrict__ vsum, const int *__restrict__ v1, const int *__restrict__ v2, int nels)
{
	const int i = get_global_id(0);

    if(i>=nels)  return;

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
  
  cl_mem v1 = NULL, v2 = NULL, vsum = NULL;

  size_t memsize = nels*sizeof(*v1);
  
  cudaError_t err;

  err = (int*)cudaMalloc(&v1, memsize); // cudaMalloc returns a device pointer
  cuda_check(err,"alloc v1");
  err = (int*)cudaMalloc(&v2, memsize);
  cuda_check(err,"alloc v2");
  err = (int*)cudaMalloc(&vsum, memsize);
  cuda_check(err,"alloc vsum");
  
  int blockSize = 256;
  int numBlocks = (nels + blockSize - 1)/blockSize;

  vecinit<<< blockSize, numBlocks >>>(v1,v2,nels);
  
  vecsum<<< blockSize, numBlocks >>>(vsum,v1,v2,nels);
	
	verify(h_vsum, nels);

  cudaFree(vsum); vsum = NULL;
  cudaFree(v2);   v2 = NULL;
  cudaFree(v1);   v1 = NULL;
}