/*
 * cuda_streams.cpp
 *
 *  Created on: Jul 5, 2022
 *      Author: mur1
 */

#include "cuda_streams.hpp"

cudaStream_t *cuda_streams = nullptr;

cudaEvent_t ejouwxcpljiftpmt_finished_event;
cudaEvent_t kmdwtwruixhmxkvl_finished_event;
cudaEvent_t cjkbavdkjbadkjbv_finished_event;
cudaEvent_t aewooyypjykeaonm_finished_event;


void cuda_streams_init(unsigned int count) {
	cuda_streams = new cudaStream_t[count];

	for (int i = 0; i < count; i++) {
		cudaStreamCreate(&cuda_streams[i]);
	}

	cudaEventCreateWithFlags(&ejouwxcpljiftpmt_finished_event, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&kmdwtwruixhmxkvl_finished_event, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&cjkbavdkjbadkjbv_finished_event, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&aewooyypjykeaonm_finished_event, cudaEventDisableTiming);

}




