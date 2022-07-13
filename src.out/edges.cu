#include "edges.hpp"

#include "buffer_alloc.hpp"
#include "cuda_streams.hpp"

__global__ void nsyodpqxejuvckwh_0_kernel(const float *in, const float *params, float *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 10000) {
		const unsigned int output_base_idx = i * 2704;
		
		{
			for (int f = 0; f < 4; f++) {
				for (int i_c = 0; i_c < 1; i_c++) {
					for (int o_r = 0; o_r < 26; o_r++) {
						for (int o_c = 0; o_c < 26; o_c++) {
							float value_f = 0.0f;
							for (int k_x = 0; k_x < 3; k_x += 1) {
								for (int k_y = 0; k_y < 3; k_y += 1) {
									float kernel_f = params[0 + f * 3 * 3 + k_y * 3 + k_x];
									unsigned int input_idx = i * 1 * 28 * 28 + i_c * 28 * 28 + (o_r * 1 + k_y) * 28 + (o_c * 1 + k_x);
									value_f += kernel_f * in[input_idx];
								}
							}
							out[output_base_idx + 0 + f * 1 * 26 * 26 + i_c * 26 * 26 +  o_r * 26 + o_c] = value_f;
						}
					}
				}
			}
		}

	} 
}

void launch_nsyodpqxejuvckwh_0_kernel() {
	const int threads_per_block = 256;
    const int blocks_per_grid = (10000 + threads_per_block - 1)/threads_per_block;
    

    nsyodpqxejuvckwh_0_kernel <<< blocks_per_grid, threads_per_block, 0, cuda_streams[2] >>> (dev_ptr_0_1 + 0, nsyodpqxejuvckwh_dev_ptr, dev_ptr_0_0 + 0);

}
__global__ void ejouwxcpljiftpmt_0_kernel(const float *in, const float *params, float *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 10000) {
		const unsigned int output_base_idx = i * 2704;
		
		for (int o_r = 0; o_r < 2704; o_r++) {
			out[output_base_idx + o_r] = (in[output_base_idx + o_r] >= 0) * in[output_base_idx + o_r]; 
		}

	} 
}

void launch_ejouwxcpljiftpmt_0_kernel() {
	const int threads_per_block = 256;
    const int blocks_per_grid = (10000 + threads_per_block - 1)/threads_per_block;
    

    ejouwxcpljiftpmt_0_kernel <<< blocks_per_grid, threads_per_block, 0, cuda_streams[2] >>> (dev_ptr_0_0 + 0, nullptr, dev_ptr_0_1 + 0);
	cudaEventRecord(ejouwxcpljiftpmt_finished_event, cuda_streams[2]);

}
__global__ void yxffajqbtoswqrpb_0_kernel(const float *in, const float *params, float *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 10000) {
		const unsigned int output_base_idx = i * 256;
		
		{
			for (int f = 0; f < 1; f++) {
				for (int i_c = 0; i_c < 4; i_c++) {
					for (int o_r = 0; o_r < 8; o_r++) {
						for (int o_c = 0; o_c < 8; o_c++) {
							float value_f = 0.0f;
							for (int k_x = 0; k_x < 3; k_x += 1) {
								for (int k_y = 0; k_y < 3; k_y += 1) {
									float kernel_f = params[0 + f * 3 * 3 + k_y * 3 + k_x];
									unsigned int input_idx = i * 4 * 26 * 26 + i_c * 26 * 26 + (o_r * 3 + k_y) * 26 + (o_c * 3 + k_x);
									float value_tmp = kernel_f * in[input_idx];
									value_f = (value_tmp > value_f) * value_tmp + (value_tmp <= value_f) * value_f;
								}
							}
							out[output_base_idx + 0 + f * 4 * 8 * 8 + i_c * 8 * 8 +  o_r * 8 + o_c] = value_f;
						}
					}
				}
			}
		}

	} 
}

void launch_yxffajqbtoswqrpb_0_kernel() {
	const int threads_per_block = 256;
    const int blocks_per_grid = (10000 + threads_per_block - 1)/threads_per_block;
    
	cudaStreamWaitEvent(cuda_streams[2], ejouwxcpljiftpmt_finished_event);

    yxffajqbtoswqrpb_0_kernel <<< blocks_per_grid, threads_per_block, 0, cuda_streams[2] >>> (dev_ptr_0_1 + 0, yxffajqbtoswqrpb_dev_ptr, dev_ptr_0_0 + 0);

}
__global__ void kltmpmjhrmgrtjhg_0_kernel(const float *in, const float *params, float *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 10000) {
		const unsigned int output_base_idx = i * 256;
		
		for (int o_r = 0; o_r < 256; o_r++) {
			out[output_base_idx + o_r] = (in[output_base_idx + o_r] >= 0) * in[output_base_idx + o_r]; 
		}

	} 
}

void launch_kltmpmjhrmgrtjhg_0_kernel() {
	const int threads_per_block = 256;
    const int blocks_per_grid = (10000 + threads_per_block - 1)/threads_per_block;
    

    kltmpmjhrmgrtjhg_0_kernel <<< blocks_per_grid, threads_per_block, 0, cuda_streams[2] >>> (dev_ptr_0_0 + 0, nullptr, dev_ptr_0_1 + 0);

}
__global__ void xdrgbxgkxqftpjqa_0_kernel(const float *in, const float *params, float *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 10000) {
		const unsigned int output_base_idx = i * 10;
		
		const unsigned int bias_offset = 10 * 256;
		for (int o_r = 0; o_r < 10; o_r++) {
			out[output_base_idx + o_r] = params[bias_offset + o_r];
			for (int i_c = 0; i_c < 256; i_c++) {
				out[output_base_idx + o_r] += params[o_r * 256 + i_c] * in[i * 256 + i_c]; 
			}
		}

	} 
}

void launch_xdrgbxgkxqftpjqa_0_kernel() {
	const int threads_per_block = 256;
    const int blocks_per_grid = (10000 + threads_per_block - 1)/threads_per_block;
    

    xdrgbxgkxqftpjqa_0_kernel <<< blocks_per_grid, threads_per_block, 0, cuda_streams[2] >>> (dev_ptr_0_1 + 0, xdrgbxgkxqftpjqa_dev_ptr + 0, dev_ptr_0_0 + 0);

}
__global__ void kmdwtwruixhmxkvl_0_kernel(const float *in, const float *params, float *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 10000) {
		const unsigned int output_base_idx = i * 30;
		
		for (int o_r = 0; o_r < 30; o_r++) {
			out[output_base_idx + o_r] = (in[output_base_idx + o_r] >= 0) * in[output_base_idx + o_r]; 
		}

	} 
}

void launch_kmdwtwruixhmxkvl_0_kernel() {
	const int threads_per_block = 256;
    const int blocks_per_grid = (10000 + threads_per_block - 1)/threads_per_block;
    

    kmdwtwruixhmxkvl_0_kernel <<< blocks_per_grid, threads_per_block, 0, cuda_streams[2] >>> (dev_ptr_0_0 + 0, nullptr, dev_ptr_0_1 + 0);
	cudaEventRecord(kmdwtwruixhmxkvl_finished_event, cuda_streams[2]);

}
__global__ void asfljvkjnakjvbad_0_kernel(const float *in, const float *params, float *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 10000) {
		const unsigned int output_base_idx = i * 20;
		
		const unsigned int bias_offset = 20 * 2704;
		for (int o_r = 0; o_r < 20; o_r++) {
			out[output_base_idx + o_r] = params[bias_offset + o_r];
			for (int i_c = 0; i_c < 2704; i_c++) {
				out[output_base_idx + o_r] += params[o_r * 2704 + i_c] * in[i * 2704 + i_c]; 
			}
		}

	} 
}

void launch_asfljvkjnakjvbad_0_kernel() {
	const int threads_per_block = 256;
    const int blocks_per_grid = (10000 + threads_per_block - 1)/threads_per_block;
    
	cudaStreamWaitEvent(cuda_streams[3], ejouwxcpljiftpmt_finished_event);

    asfljvkjnakjvbad_0_kernel <<< blocks_per_grid, threads_per_block, 0, cuda_streams[3] >>> (dev_ptr_0_1 + 0, asfljvkjnakjvbad_dev_ptr + 0, dev_ptr_1_0 + 0);

}
__global__ void cjkbavdkjbadkjbv_0_kernel(const float *in, const float *params, float *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 10000) {
		const unsigned int output_base_idx = i * 30;
		
		for (int o_r = 0; o_r < 30; o_r++) {
			out[output_base_idx + o_r] = (in[output_base_idx + o_r] >= 0) * in[output_base_idx + o_r]; 
		}

	} 
}

void launch_cjkbavdkjbadkjbv_0_kernel() {
	const int threads_per_block = 256;
    const int blocks_per_grid = (10000 + threads_per_block - 1)/threads_per_block;
    

    cjkbavdkjbadkjbv_0_kernel <<< blocks_per_grid, threads_per_block, 0, cuda_streams[3] >>> (dev_ptr_1_0 + 0, nullptr, dev_ptr_0_1 + 10);
	cudaEventRecord(cjkbavdkjbadkjbv_finished_event, cuda_streams[3]);

}
__global__ void vvbabehgakhgshhg_0_kernel(const float *in, const float *params, float *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 10000) {
		const unsigned int output_base_idx = i * 30;
		
		for (int o_r = 0; o_r < 30; o_r++) {
			out[output_base_idx + o_r] = in[output_base_idx + o_r]; 
		}

	} 
}

void launch_vvbabehgakhgshhg_0_kernel() {
	const int threads_per_block = 256;
    const int blocks_per_grid = (10000 + threads_per_block - 1)/threads_per_block;
    
	cudaStreamWaitEvent(cuda_streams[2], kmdwtwruixhmxkvl_finished_event);
	cudaStreamWaitEvent(cuda_streams[2], cjkbavdkjbadkjbv_finished_event);

    vvbabehgakhgshhg_0_kernel <<< blocks_per_grid, threads_per_block, 0, cuda_streams[2] >>> (dev_ptr_0_1 + 0, nullptr, dev_ptr_0_0 + 0);

}
__global__ void mngghxgyiykuvxdv_0_kernel(const float *in, const float *params, float *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 10000) {
		const unsigned int output_base_idx = i * 10;
		
		const unsigned int bias_offset = 10 * 30;
		for (int o_r = 0; o_r < 10; o_r++) {
			out[output_base_idx + o_r] = params[bias_offset + o_r];
			for (int i_c = 0; i_c < 30; i_c++) {
				out[output_base_idx + o_r] += params[o_r * 30 + i_c] * in[i * 30 + i_c]; 
			}
		}

	} 
}

void launch_mngghxgyiykuvxdv_0_kernel() {
	const int threads_per_block = 256;
    const int blocks_per_grid = (10000 + threads_per_block - 1)/threads_per_block;
    

    mngghxgyiykuvxdv_0_kernel <<< blocks_per_grid, threads_per_block, 0, cuda_streams[2] >>> (dev_ptr_0_0 + 0, mngghxgyiykuvxdv_dev_ptr + 0, dev_ptr_0_1 + 0);

}
__global__ void aewooyypjykeaonm_0_kernel(const float *in, const float *params, float *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < 10000) {
		const unsigned int output_base_idx = i * 10;
		
		{
		float summed = 0.0f;
        for (int c = 0; c < 10; c++) {
                out[output_base_idx + c] = exp(in[output_base_idx + c]);
                summed += out[output_base_idx + c];
        }

        for (int c = 0; c < 10; c++) {
        	if (summed > 0) out[output_base_idx + c] /= summed;
        }
	}

	} 
}

void launch_aewooyypjykeaonm_0_kernel() {
	const int threads_per_block = 256;
    const int blocks_per_grid = (10000 + threads_per_block - 1)/threads_per_block;
    

    aewooyypjykeaonm_0_kernel <<< blocks_per_grid, threads_per_block, 0, cuda_streams[2] >>> (dev_ptr_0_1 + 0, nullptr, dev_ptr_0_0 + 0);
	cudaEventRecord(aewooyypjykeaonm_finished_event, cuda_streams[2]);

}

