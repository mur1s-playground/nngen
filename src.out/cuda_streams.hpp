/*
 * cuda_streams.hpp
 *
 *  Created on: Jul 5, 2022
 *      Author: mur1
*/

#pragma once

#include <cuda_runtime.h>

extern cudaStream_t *cuda_streams;

void cuda_streams_init(unsigned int count);



