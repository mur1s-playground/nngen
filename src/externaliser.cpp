/*
 * externaliser.cpp
 *
 *  Created on: Jun 27, 2022
 *      Author: mur1
 */


#include "externaliser.hpp"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

void externaliser_init(struct externaliser *e, unsigned int preallocate) {
	e->position = 0;
	e->alloc_c = preallocate;
	if (preallocate > 0) {
		e->out_ext = (char *) malloc(preallocate);
	} else {
		e->out_ext = nullptr;
	}
}

void externaliser_destroy(struct externaliser *e) {
	e->position = 0;

	if (e->alloc_c > 0) {
		free(e->out_ext);
	}
	e->alloc_c = 0;
}


void externaliser_alloc_on_demand(struct externaliser *e, unsigned int size_to_add) {
	if (e->position + size_to_add < e->alloc_c) {
		if (e->alloc_c == 0) {
			e->out_ext = (char *) malloc(size_to_add);
		} else {
			e->out_ext = (char *) realloc(e->out_ext, e->alloc_c + size_to_add);
		}
		e->alloc_c += size_to_add;
	}
}

void externaliser_int_add(struct externaliser *e, int to_add) {
	externaliser_alloc_on_demand(e, sizeof(int));
	memcpy(&e->out_ext[e->position], &to_add, sizeof(int));
	e->position += sizeof(int);
}

void externaliser_raw_add(struct externaliser *e, void *raw, int size_to_add) {
	externaliser_alloc_on_demand(e, size_to_add);
	memcpy(&e->out_ext[e->position], raw, size_to_add);
	e->position += size_to_add;
}

void externaliser_str_add(struct externaliser *e, char *str, unsigned int len) {
	externaliser_int_add(e, len);
	externaliser_raw_add(e, str, len);
}

void externaliser_2device(struct externaliser *e, void **device_ptr) {
	unsigned int size_req = e->position;
	cudaMalloc(device_ptr, size_req);
	cudaMemcpy(*device_ptr, e->out_ext, size_req, cudaMemcpyHostToDevice);
}

