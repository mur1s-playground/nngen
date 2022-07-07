/*
 * nn_filter.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_FILTER_HPP_
#define SRC_NN_FILTER_HPP_

#include <vector>
#include <string>

struct nn_filter {
	unsigned int filters_c;
	unsigned int kernel_size[2];
	unsigned int stride[2];
	unsigned int padding[2];
	unsigned int dilation[2];
};

void nn_filter_parse(struct nn_filter *nn_f, std::vector<std::string> vec, int *current_param_in_out);
unsigned int nn_filter_mem_req(struct nn_filter *nn_f);

void nn_filter_dump(struct nn_filter *nn_f);

#endif /* SRC_NN_FILTER_HPP_ */
