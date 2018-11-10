
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


const int N = 16;
const int a[N] = {10,1,8,-1,0,-2,3,5,-2,-3,2,7,0,11,0,2};
const int b[N] = {10,1,8,-1,0,-2,3,5,-2,-3,2,7,0,11,0,2};


__global__ void reduce1(int *a)
{
	int tid = threadIdx.x;
	for (int i = 1; i < blockDim.x; i *= 2)
	{
		if (tid % (2 * i) == 0)
			a[tid] += a[tid + i];
	}
}

__global__ void reduce2(int *a)
{
	int tid = threadIdx.x;
	for (int i = 1; i < blockDim.x; i *= 2)
	{
		int idx = 2 * i * tid;

		if (idx < blockDim.x)
			a[idx] += a[idx + i];
	}
}

__global__ void reduce3(int *a)
{
	int tid = threadIdx.x;
	for (unsigned int i = blockDim.x / 2; i > 0; i = i / 2)
	{

		if (tid < i)
			a[tid] += a[tid + i];
	}

}

__global__ void dot(int *a, int *b)
{
	int tid = threadIdx.x;
	a[tid] *= b[tid];
	for (int i = 1; i < blockDim.x; i *= 2)
	{
		int idx = 2 * i * tid;

		if (idx < blockDim.x)
			a[idx] += a[idx + i];
	}
}

int main()
{
    int c[N] = { 0 };
	int *dev_a = 0;
	int *dev_b = 0;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	dot <<<1, N>>>(dev_a, dev_b);
	cudaMemcpy(c, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", c[0]);//395 expected

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	reduce1 << <1, N >>>(dev_a);
	cudaMemcpy(c, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", c[0]);//41 expected

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	reduce2 << <1, N >>>(dev_a);
	cudaMemcpy(c, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", c[0]);//41 expected

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	reduce3 << <1, N >>>(dev_a);
	cudaMemcpy(c, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", c[0]); //41 expected


    return 0;
}
