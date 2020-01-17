#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <stdarg.h>
#include <cuda_runtime_api.h>

#define BUFSIZE 4096

typedef unsigned int uint;

using namespace std;

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

__host__ __device__   // NVCC compiles this code for both device and host
void operator+=(int4 & a, int b)
{
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}

__device__
void microscan(int4 &q)
{
  q.y += q.x;
  q.z += q.y;
  q.w += q.z;
}

// a / b, rounding up
__host__ __device__
uint div_up(uint a, uint b) {return (a + b - 1)/b;}

// round a to the next multiple of b
__host__ __device__
uint round_mul_up(uint a, uint b) {return div_up(a, b)*b;}

__device__
int get_global_id() {return blockIdx.x * blockDim.x + threadIdx.x;}

__global__
void vecinit(int *out, int n)
{
	const int i = get_global_id();
	if (i < n) out[i] = (i+1);
}

// CUDA Shared Memory
extern __shared__ int lmem[];

// scan that accesses global memory directly
__device__
int scan_pass(int gi, int nels,
	 int * __restrict__ out,
	 const int * __restrict__ in,
	 int * __restrict__ lmem,
	 int corr)
{
	const uint li = threadIdx.x;
	const uint lws = blockDim.x;
	int acc = (gi < nels ? in[gi] : 0);

	uint write_mask = 1U;
	uint read_mask = ~0U;

	lmem[li] = acc;
	while (write_mask < lws) 
  {
		__syncthreads();
		if (li & write_mask) 
    {
			acc += lmem[(li & read_mask) - 1];
			lmem[li] = acc;
		}
		write_mask <<= 1;
		read_mask <<= 1;
	}

	acc += corr;
	if (gi < nels)
		out[gi] = acc;

  __syncthreads();
	corr += lmem[lws - 1];

	// ensure that lmem[i] on the next cycle from the last work-item does not
	// overwrite lmem[lws-1] before all other work-item read it
	__syncthreads();
	return corr;
}

// scan pass in which we pass the values that need to be scanned
__device__
int scan_pass(int gi, int global_nquarts, 
    int4 * __restrict__ out, 
    int4 &q, 
    int * __restrict__ lmem, 
    int corr)
{
	const uint li = threadIdx.x;
	const uint lws = blockDim.x;
	int acc = q.w;

	uint write_mask = 1U;
	uint read_mask = ~0U;

	lmem[li] = acc;
	while (write_mask < lws) 
  {
		__syncthreads();
		if (li & write_mask) 
    {
			acc += lmem[(li & read_mask) - 1];
			lmem[li] = acc;
		}
		write_mask <<= 1;
		read_mask <<= 1;
	}

  // complete work-group scan
  __syncthreads();
  if (li > 0)
    q += lmem[li-1];
  
  // sliding window correction
  q+= corr;

	if (gi < global_nquarts)
		out[gi] = q;

	corr += lmem[lws - 1];

  // ensure that lmem[i] on the next cycle
  // from the last work-item does not
	// overwrite lmem[lws-1] before all other work-item read it
	__syncthreads();
	return corr;
}

/* single-work-group version: used to scan the tails of the partial scans */
__global__
void scan1_lmem(int * __restrict__ out,
	const int * __restrict__ in,
	uint nels)
{
	const uint lws = blockDim.x;
	const uint limit = round_mul_up(nels, lws);
  uint gi = get_global_id();
  int corr = 0;
  
	while (gi < limit) // 4x4 loop
  {
    corr = scan_pass(gi,nels,out,in,lmem,corr);
    gi += lws;
	}
}

/* multi-work-group, vectorized (x4) version 
*  each work-item reads and writes a int4.
*  once a work-item is done, it save a single int.
*/
__global__
void scan4_lmem(int4 * __restrict__ out,
	int * __restrict__ tails,
	const int4 * __restrict__ in, 
	uint global_nquarts)
{
	const uint lws = blockDim.x;

	// number of quarts for the single work-group:
	// start by dividing the total number of elements by the number of groups,
	// rounding up
	uint local_nquarts = div_up(global_nquarts, gridDim.x);
	// round up to the next multiple of lws
	local_nquarts = round_mul_up(local_nquarts, lws);

	const uint begin = blockIdx.x*local_nquarts;
	const uint end = min(begin + local_nquarts, global_nquarts);
	const uint limit = round_mul_up(end, lws);
	int corr = 0;

	uint gi = begin + threadIdx.x;
	while (gi < limit)
  {
    int g0 = gi + 0*lws;
    int g1 = gi + 1*lws;
    int g2 = gi + 2*lws;
    int g3 = gi + 3*lws;

    int4 q0 = g0 < global_nquarts ? in[g0] : make_int4(0,0,0,0);
    int4 q1 = g1 < global_nquarts ? in[g1] : make_int4(0,0,0,0);
    int4 q2 = g2 < global_nquarts ? in[g2] : make_int4(0,0,0,0);
    int4 q3 = g3 < global_nquarts ? in[g3] : make_int4(0,0,0,0);

    microscan(q0);
    microscan(q1);
    microscan(q2);
    microscan(q3);

    corr = scan_pass(g0, global_nquarts, out, q0, lmem, corr);
    corr = scan_pass(g1, global_nquarts, out, q1, lmem, corr);
    corr = scan_pass(g2, global_nquarts, out, q2, lmem, corr);
    corr = scan_pass(g3, global_nquarts, out, q3, lmem, corr);
    
    gi += 4*lws;
    /*  
    ALTERNATIVE: exploit instruction level parameters (ILP).
    If the hardware allows it, these are not 4 sums but 2 parallelized sums.
    q.y += q.x
    q.w += q.z // these 2 are indipendent sums
    q.z += q.y
    q.w += q.y // idem
    */
	}

	if (threadIdx.x == 0)
		tails[blockIdx.x] = corr;
}

/* fixup the partial scans with the scanned tails */
__global__
void scanN_fixup(int4 * __restrict__ out,
	const int * __restrict__ tails,
	uint global_nquarts)
{
	if (blockIdx.x == 0) return;

	const uint lws = blockDim.x;

	// number of elements for the single work-group:
	// start by dividing the total number of elements by the number of groups,
	// rounding up
	uint local_nquarts = div_up(global_nquarts, gridDim.x);
	// round up to the next multiple of lws
	local_nquarts = round_mul_up(local_nquarts, lws);

	const uint begin = blockIdx.x*local_nquarts;
	const uint end = min(begin + local_nquarts, global_nquarts);
	const int corr = tails[blockIdx.x-1];

	uint gi = begin + threadIdx.x;
	while (gi < end) 
  {
		out[gi] += corr;
		gi += lws;
	}
}

void verify(const int *vsum, int nels)
{
	int scan = 0;
	for (int i = 0; i < nels; ++i) 
  {
		scan += (i+1);
		if (vsum[i] != scan) 
			fprintf(stderr, "mismatch @ %d : %d != %d\n", i, vsum[i], scan), exit(3);
	}
}


int main(int argc, char *argv[])
{
	if (argc <= 3)
		fprintf(stderr, "specify number of elements, lws, nwg\n"), exit(1);

  const int nels = atoi(argv[1]);
  const int lws = atoi(argv[2]);
  const int nwg = atoi(argv[3]);
  const size_t memsize = nels*sizeof(int);
  const size_t nwg_mem = nwg*sizeof(int);
 
  if (nels & 3) cuda_check(cudaErrorInvalidValue, "nels must be multiple of 4");
  if (lws & (lws-1)) cuda_check(cudaErrorInvalidValue, "lws"); // this should be invalid value, not invalid device(!!!)
  
  int *d_v1 = NULL, *d_v2 = NULL, *d_tails = NULL;
  
  cudaError_t err;

  err = cudaMalloc(&d_v1, memsize); // cudaMalloc returns a device pointer
  cuda_check(err,"alloc v1");
  err = cudaMalloc(&d_v2, memsize);
  cuda_check(err,"alloc v2");
  err = cudaMalloc(&d_tails, nwg_mem);
  cuda_check(err,"alloc vsum");

  cudaEvent_t pre_init, post_init;
  cudaEvent_t pre_scan1, post_scan1;
  cudaEvent_t pre_scan_tails, post_scan_tails;
  cudaEvent_t pre_fixup, post_fixup;
  cudaEvent_t pre_copy, post_copy;

  err = cudaEventCreate(&pre_init, 0);
  cuda_check(err, "pre_init event create");
  err = cudaEventCreate(&post_init, 0);
  cuda_check(err, "post_init event create");
  
  err = cudaEventCreate(&pre_scan1, 0);
  cuda_check(err, "pre_scan1 event create");
  err = cudaEventCreate(&post_scan1, 0);
  cuda_check(err, "post_scan1 event create");
  
  err = cudaEventCreate(&pre_scan_tails, 0);
  cuda_check(err, "pre_scan_tails event create");
  err = cudaEventCreate(&post_scan_tails, 0);
  cuda_check(err, "post_scan_tails event create");
  
  err = cudaEventCreate(&pre_fixup, 0);
  cuda_check(err, "pre_fixup event create");
  err = cudaEventCreate(&post_fixup, 0);
  cuda_check(err, "post_fixup event create");

  err = cudaEventCreate(&pre_copy, 0);
  cuda_check(err, "pre_copy event create");
  err = cudaEventCreate(&post_copy, 0);
  cuda_check(err, "post_copy event create");

  err = cudaEventRecord(pre_init);
  cuda_check(err, "pre_init record");
  vecinit<<< div_up(nels, lws), lws >>>(d_v1,nels);
  err = cudaEventRecord(post_init);
  cuda_check(err, "post_init record");

  int nquarts = nels/4;
  err = cudaEventRecord(pre_scan1);
  cuda_check(err, "pre_scan1 record");
  if (nwg>1)
    scan4_lmem<<< nwg, lws, lws*sizeof(int) >>>((int4*)d_v2, d_tails, (int4*)d_v1, nquarts);
  else
    scan1_lmem<<< 1, lws,lws*sizeof(int) >>>(d_v2, d_v1, nels);
  err = cudaEventRecord(post_scan1);
  cuda_check(err, "post_scan1 record");

  if (nwg > 1)
  {
    err = cudaEventRecord(pre_scan_tails);
    cuda_check(err, "record pre_scan_tails");
    scan1_lmem<<< 1, lws,lws*sizeof(int) >>>(d_tails, d_tails, nwg);
    err = cudaEventRecord(post_scan_tails);
    cuda_check(err, "record post_scan_tails");

    err = cudaEventRecord(pre_fixup);
    cuda_check(err, "record pre_fixup");
    scanN_fixup<<< nwg, lws >>>((int4*)d_v2, d_tails, nquarts);
    err = cudaEventRecord(post_fixup);
    cuda_check(err, "record post_fixup");
  }

  int * h_v2;
  err = cudaHostAlloc(&h_v2, memsize, cudaHostAllocPortable); //DMA
  cuda_check(err, "host alloc");

  err = cudaEventRecord(pre_copy);
  cuda_check(err, "pre_copy record");
  err = cudaMemcpy(h_v2,d_v2,memsize, cudaMemcpyDeviceToHost);
  cuda_check(err, "copy v2");
  err = cudaEventRecord(post_copy);
  cuda_check(err, "post_copy record");

	verify(h_v2, nels);

  cudaEventSynchronize(post_copy);
  cuda_check(err, "sync post_copy");

  float init_time, scan1_time, scan_tails_time, fixup_time, copy_time;
  float scan_time_total;
  err = cudaEventElapsedTime(&init_time, pre_init, post_init);  //in ms
  cuda_check(err, "get init time");
  err = cudaEventElapsedTime(&scan1_time, pre_scan1, post_scan1);  //in ms
  cuda_check(err, "get scan1 time");
  if (nwg > 1)
  {
    err = cudaEventElapsedTime(&scan_tails_time, pre_scan_tails, post_scan_tails);  //in ms
    cuda_check(err, "get scan tails time");
    err = cudaEventElapsedTime(&fixup_time, pre_fixup, post_fixup);  //in ms
    cuda_check(err, "get fixup time");
  }
  err = cudaEventElapsedTime(&copy_time, pre_copy, post_copy);  //in ms
  cuda_check(err, "get copy time");
  err = cudaEventElapsedTime(&scan_time_total, pre_scan1, nwg > 1 ? post_fixup : post_scan1);
  cuda_check(err, "get scan time total");

  printf("init: %d els in %6.4gms: %6.4gGB/s, %6.4gGE/s\n", nels, init_time, memsize/init_time/1.0e6, nels/init_time/1.0e6);
  printf("scan0:  %d els in %6.4gms: %6.4gGB/s, %6.4gGE/s\n", nels, scan1_time,2*memsize/scan1_time/1.0e6, nels/scan1_time/1.0e6);
  if(nwg>1)
  {
    printf("scan tails:  %d els in %6.4gms: %6.4gGB/s, %6.4gGE/s\n", nwg, scan_tails_time,2*nwg_mem/scan_tails_time/1.0e6, nwg/scan_tails_time/1.0e6);
    printf("scan fixup:  %d els in %6.4gms: %6.4gGB/s, %6.4gGE/s\n", nels, fixup_time,2*(memsize - lws*sizeof(int))/fixup_time/1.0e6, nels/fixup_time/1.0e6);
  }
  printf("copy: %d els in %6.4gms: %6.4gGB/s, %6.4gGE/s\n", nels, copy_time,memsize/copy_time/1.0e6, nels/copy_time/1.0e6);
  printf("scan total: %d els in %6.4gms: %6.4gGB/s, %6.4gGE/s\n", nels, scan_time_total,memsize/scan_time_total/1.0e6, nels/scan_time_total/1.0e6);

  cudaEventDestroy(pre_init);
  cudaEventDestroy(post_init);
  cudaEventDestroy(pre_scan1);
  cudaEventDestroy(post_scan1);
  cudaEventDestroy(pre_scan_tails);
  cudaEventDestroy(post_scan_tails);
  cudaEventDestroy(pre_fixup);
  cudaEventDestroy(post_fixup);
  cudaEventDestroy(pre_copy);
  cudaEventDestroy(post_copy);

  cudaFreeHost(h_v2);     h_v2 = NULL;
  cudaFree(d_v2);   d_v2 = NULL;
  cudaFree(d_tails); d_tails = NULL;
  cudaFree(d_v1);   d_v1 = NULL;
}