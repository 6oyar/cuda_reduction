#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


const int N = 512;
//const float a[N] = {10,1,8,-1,0,-2,3,5,-2,-3,2,7,0,11,0,2};
//const float b[N] = {10,1,8,-1,0,-2,3,5,-2,-3,2,7,0,11,0,2};
//const float xx[N] = { -4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5 };


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

__global__ void reduce3(float *x)
{
	int tid = threadIdx.x;

	x[tid] = x[tid] * x[tid] * (x[tid + 1] - x[tid]);
	x[N - 1] = 0;

	for (unsigned int i = blockDim.x / 2; i > 0; i = i / 2)
	{

		if (tid < i)
			x[tid] += x[tid + i];
	}

}

float*  integration(float low, float high) 
{
	float x[N] = {0};
	float delta = (high - low) / (N - 1);

	for (size_t i = 0; i < N; i++)
	{
		x[i] = low + i * delta;
		/*printf("%.1f\n", x[i]);*/
	}

	float c[N] = { 0 }; 
	float *dev_a = 0;
	cudaMalloc((void**)&dev_a, N * sizeof(float));
	cudaMemcpy(dev_a, x, N * sizeof(float), cudaMemcpyHostToDevice);
	reduce3 << <1, N >> >(dev_a);
	cudaMemcpy(c, dev_a, N * sizeof(float), cudaMemcpyDeviceToHost);

	printf("%.1f\n", c[0]);


	return x;

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

__global__ void reduce(int* input, int* output) 
{
	__shared__ int* data;

	int tid = threadIdx.x;
	data[tid] = input[tid];

	__syncthreads();

	for (int i = blockDim.x / 2; i > 0; i = i / 2)
	{
		if (tid < i)
		{
			data[tid] += data[tid + i];
		}
		__syncthreads();
	}

	if (tid == 0) output[blockIdx.x] = data[0];
}

int main()
{
	//float c[N] = { 0 };
	//float *dev_a = 0;
	//float *dev_b = 0;

	//cudaMalloc((void**)&dev_a, N * sizeof(float));
	//cudaMalloc((void**)&dev_b, N * sizeof(float));

	//cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
	//dot <<<1, N>>>(dev_a, dev_b);
	//cudaMemcpy(c, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);
	//printf("%d\n", c[0]);//395 expected

	//cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	//reduce1 <<<1, N >>>(dev_a);
	//cudaMemcpy(c, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);
	//printf("%d\n", c[0]);//41 expected

	//cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	//reduce2 <<<1, N >>>(dev_a);
	//cudaMemcpy(c, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);
	//printf("%d\n", c[0]);//41 expected


	integration(-4, 4);

	//cudaMemcpy(dev_a, xx, N * sizeof(float), cudaMemcpyHostToDevice);
	//reduce3 <<<1, N >>>(dev_a);
	//cudaMemcpy(c, dev_a, N * sizeof(float), cudaMemcpyDeviceToHost);
	//printf("%.1f\n", c[0]);

	//cudaMemcpy(dev_b, xx, N * sizeof(float), cudaMemcpyHostToDevice);
	//integration << <1, N >> > (dev_b);
	//cudaMemcpy(c, dev_b, N * sizeof(float), cudaMemcpyDeviceToHost);

	

	//cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	//reduce <<<2, N / 2>>>(dev_a, dev_b);
	//cudaMemcpy(c, dev_b, N * sizeof(int), cudaMemcpyDeviceToHost);
	//printf("%d\n", c[0]); //41 expected


    return 0;
}
