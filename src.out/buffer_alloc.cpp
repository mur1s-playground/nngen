#include "buffer_alloc.hpp"

#include <cuda_runtime.h>
#include <cstdlib>

#include "util.hpp" 

float *dev_ptr_0_0 = nullptr;
float *dev_ptr_0_1 = nullptr;
float *dev_ptr_1_0 = nullptr;
float *dev_ptr_1_1 = nullptr;

float *nsyodpqxejuvckwh_dev_ptr = nullptr;
float *yxffajqbtoswqrpb_dev_ptr = nullptr;
float *xdrgbxgkxqftpjqa_dev_ptr = nullptr;
float *asfljvkjnakjvbad_dev_ptr = nullptr;
float *mngghxgyiykuvxdv_dev_ptr = nullptr;


void buffer_alloc() {
	cudaMalloc(&nsyodpqxejuvckwh_dev_ptr, 144 * sizeof(float));
	cudaMalloc(&yxffajqbtoswqrpb_dev_ptr, 36 * sizeof(float));
	cudaMalloc(&xdrgbxgkxqftpjqa_dev_ptr, 10 * sizeof(float));
	cudaMalloc(&asfljvkjnakjvbad_dev_ptr, 20 * sizeof(float));
	cudaMalloc(&mngghxgyiykuvxdv_dev_ptr, 10 * sizeof(float));
	cudaMalloc(&dev_ptr_0_0, 108160000 * sizeof(float));
	cudaMalloc(&dev_ptr_0_1, 108160000 * sizeof(float));
	cudaMalloc(&dev_ptr_1_0, 1200000 * sizeof(float));
	cudaMalloc(&dev_ptr_1_1, 1200000 * sizeof(float));

}

void buffer_fill() {
	{
		float *tmp_ptr = (float *) malloc(144 * sizeof(float));
		util_file_read("params/nsyodpqxejuvckwh.params", 0, 144 * sizeof(float), (char *)tmp_ptr);
		cudaMemcpy(nsyodpqxejuvckwh_dev_ptr, tmp_ptr, 144 * sizeof(float), cudaMemcpyHostToDevice);
		free(tmp_ptr);
	}
	{
		float *tmp_ptr = (float *) malloc(36 * sizeof(float));
		util_file_read("params/yxffajqbtoswqrpb.params", 0, 36 * sizeof(float), (char *)tmp_ptr);
		cudaMemcpy(yxffajqbtoswqrpb_dev_ptr, tmp_ptr, 36 * sizeof(float), cudaMemcpyHostToDevice);
		free(tmp_ptr);
	}
	{
		float *tmp_ptr = (float *) malloc(10 * sizeof(float));
		util_file_read("params/xdrgbxgkxqftpjqa.params", 0, 10 * sizeof(float), (char *)tmp_ptr);
		cudaMemcpy(xdrgbxgkxqftpjqa_dev_ptr, tmp_ptr, 10 * sizeof(float), cudaMemcpyHostToDevice);
		free(tmp_ptr);
	}
	{
		float *tmp_ptr = (float *) malloc(20 * sizeof(float));
		util_file_read("params/asfljvkjnakjvbad.params", 0, 20 * sizeof(float), (char *)tmp_ptr);
		cudaMemcpy(asfljvkjnakjvbad_dev_ptr, tmp_ptr, 20 * sizeof(float), cudaMemcpyHostToDevice);
		free(tmp_ptr);
	}
	{
		float *tmp_ptr = (float *) malloc(10 * sizeof(float));
		util_file_read("params/mngghxgyiykuvxdv.params", 0, 10 * sizeof(float), (char *)tmp_ptr);
		cudaMemcpy(mngghxgyiykuvxdv_dev_ptr, tmp_ptr, 10 * sizeof(float), cudaMemcpyHostToDevice);
		free(tmp_ptr);
	}

}
