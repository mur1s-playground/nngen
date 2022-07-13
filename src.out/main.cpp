#include <cuda_runtime.h>

#include "cuda_streams.hpp"
#include "buffer_alloc.hpp"
#include "edges.hpp"
#include "input_stream.hpp"
#include "mutex.h"
#include "util.hpp"

int main() {
	cuda_streams_init(4);
	buffer_alloc();
	buffer_fill();
	
	struct input_stream is;
	input_stream_init(&is);

	unsigned int offset = 0;
	while(true) {
		int cur_buffer = -1;
		mutex_wait_for(&is.buffer_lock);
		for (int o = 0; o < is.buffer_slot_count; o++) {
			int o_id = (offset + o) % is.buffer_slot_count;
			offset++;
			if (is.buffer_state[o_id] == IS_B_S_WAITING_FOR_OUTPUT) {
				cur_buffer = o_id;
				break;
			}
		}
		mutex_release(&is.buffer_lock);
		
		if (cur_buffer > -1) {
			cudaMemcpy(dev_ptr_0_0, &is.buffer[is.buffer_slot_size], is.batch_input_size, cudaMemcpyHostToDevice);
			launch_nsyodpqxejuvckwh_0_kernel();//0, 0, 1
			launch_ejouwxcpljiftpmt_0_kernel();//0, 0, 2
			launch_yxffajqbtoswqrpb_0_kernel();//1, 0, 5
			launch_asfljvkjnakjvbad_0_kernel();//1, 1, 17
			launch_kltmpmjhrmgrtjhg_0_kernel();//1, 0, 6
			launch_cjkbavdkjbadkjbv_0_kernel();//1, 1, 18
			launch_xdrgbxgkxqftpjqa_0_kernel();//1, 0, 7
			launch_kmdwtwruixhmxkvl_0_kernel();//1, 0, 8
			launch_vvbabehgakhgshhg_0_kernel();//2, 0, 11
			launch_mngghxgyiykuvxdv_0_kernel();//2, 0, 12
			launch_aewooyypjykeaonm_0_kernel();//2, 0, 13


			while (true) {
				if (cudaEventQuery(aewooyypjykeaonm_finished_event) == cudaSuccess) break;
                util_sleep(33);
            }
			cudaMemcpy(&is.buffer[is.buffer_slot_size], dev_ptr_0_0, is.batch_output_size, cudaMemcpyDeviceToHost);

			mutex_wait_for(&is.buffer_lock);
			is.buffer_state[cur_buffer] = IS_B_S_OUTPUT_READY;
			mutex_release(&is.buffer_lock);
		} else {
			util_sleep(33);
		}
	}

	return 0;
}
