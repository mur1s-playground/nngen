#include <cuda_runtime.h>

#include "cuda_streams.hpp"
#include "buffer_alloc.hpp"
#include "edges.hpp"
#include "util.hpp"

int main() {
	cuda_streams_init(2);
	buffer_alloc();
	buffer_fill();

	while(true) {
		//if input ready {
			launch_nsyodpqxejuvckwh_0_kernel();//0, 0, 1
			launch_ejouwxcpljiftpmt_0_kernel();//0, 0, 2
			launch_yxffajqbtoswqrpb_0_kernel();//1, 0, 5
			cudaDeviceSynchronize();
			launch_asfljvkjnakjvbad_0_kernel();//1, 1, 17
			cudaDeviceSynchronize();
			launch_kltmpmjhrmgrtjhg_0_kernel();//1, 0, 6
			launch_cjkbavdkjbadkjbv_0_kernel();//1, 1, 18
			launch_xdrgbxgkxqftpjqa_0_kernel();//1, 0, 7
			launch_kmdwtwruixhmxkvl_0_kernel();//1, 0, 8
			launch_vvbabehgakhgshhg_0_kernel();//2, 0, 11
			cudaDeviceSynchronize();
			launch_mngghxgyiykuvxdv_0_kernel();//2, 0, 12
			launch_aewooyypjykeaonm_0_kernel();//2, 0, 13
			cudaDeviceSynchronize();

		//} else {
			util_sleep(33);
		//}
	}

	return 0;
}
