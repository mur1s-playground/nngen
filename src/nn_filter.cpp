/*
 * nn_filter.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "nn_filter.hpp"

#include <string>

void nn_filter_parse(struct nn_filter *nn_f, std::vector<std::string> vec, int *current_param_in_out) {
	int params_position = *current_param_in_out;
	nn_f->filters_c			= std::stoi(vec[params_position].c_str());	params_position++;
	nn_f->kernel_size[0] 	= std::stoi(vec[params_position].c_str());	params_position++;
	nn_f->kernel_size[1] 	= std::stoi(vec[params_position].c_str());	params_position++;
	nn_f->padding[0]	 	= std::stoi(vec[params_position].c_str());	params_position++;
	nn_f->padding[1]	 	= std::stoi(vec[params_position].c_str());	params_position++;
	nn_f->stride[0]	 		= std::stoi(vec[params_position].c_str());	params_position++;
	nn_f->stride[1]	 		= std::stoi(vec[params_position].c_str());	params_position++;
	nn_f->dilation[0]		= std::stoi(vec[params_position].c_str());	params_position++;
	nn_f->dilation[1]		= std::stoi(vec[params_position].c_str());	params_position++;
	*current_param_in_out = params_position;
}

unsigned int nn_filter_mem_req(struct nn_filter *nn_f) {
	return nn_f->filters_c * nn_f->kernel_size[0] * nn_f->kernel_size[1] * sizeof(float);
}

void nn_filter_dump(struct nn_filter *nn_f) {
	printf("filters_c: %d, kernel_size (%d %d), padding (%d %d), stride (%d %d), dilation (%d %d)",
			nn_f->filters_c, nn_f->kernel_size[0], nn_f->kernel_size[1],  nn_f->padding[0],
			nn_f->padding[1],nn_f->stride[0], nn_f->stride[1], nn_f->dilation[0], nn_f->dilation[1]);
}

