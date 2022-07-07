/*
 * nn_node.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_NODE_HPP_
#define SRC_NN_NODE_HPP_

#include "m_array.hpp"
#include "nn_dimension.hpp"
#include "externaliser.hpp"

enum nn_node_type {
	NN_N_T_UNSPEC,
	NN_N_T_INPUT,
	NN_N_T_THROUGHPUT,
	NN_N_T_OUTPUT
};

struct nn_node {
	char 							*id;
	enum nn_node_type 				type;
	m_array<struct nn_dimension>	dimensions;
	unsigned int					output_mem_req;
	unsigned int					edges_out_c;
	unsigned int					edges_in_c;

	int 							ui_grid_position[2];

	char							*last_used_buffer_str;
};

void nn_node_init(struct nn_node *nn_n);
void nn_node_dump(struct nn_node *nn_n);

unsigned int nn_node_layer_width_get(struct nn_node *nn_n);

void nn_node_externalise_evo(struct nn_node *nn_n, struct externaliser *e);

#endif /* SRC_NN_NODE_HPP_ */
