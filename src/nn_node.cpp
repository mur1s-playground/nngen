/*
 * nn_node.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "nn_node.hpp"

#include "util.hpp"

#include <stdio.h>

void nn_node_init(struct nn_node *nn_n) {
	nn_n->id 	= nullptr;
	nn_n->type 	= NN_N_T_UNSPEC;
	m_array_init(&nn_n->dimensions);
	nn_n->output_mem_req = 0;
	nn_n->edges_out_c = 0;
	nn_n->edges_in_c = 0;

	nn_n->ui_grid_position[0] = -1;
	nn_n->ui_grid_position[1] = -1;

	nn_n->last_used_buffer_str = nullptr;
}

void nn_node_dump(struct nn_node *nn_n) {
	printf("node (%d)->(%s)->(%d): ", nn_n->edges_in_c, nn_n->id, nn_n->edges_out_c);
	switch (nn_n->type) {
		case NN_N_T_UNSPEC:
			printf("UNSPEC");
			break;
		case NN_N_T_INPUT:
			printf("INPUT");
			break;
		case NN_N_T_THROUGHPUT:
			printf("THROUGHPUT");
			break;
		case NN_N_T_OUTPUT:
			printf("OUTPUT");
			break;
	}
	printf(", mem_req: ");
	util_size_format_print(nn_n->output_mem_req);
	printf(",");
	int dims_c = m_array_size(&nn_n->dimensions);
	for (int d = 0; d < dims_c; d++) {
		struct nn_dimension *nn_d = &nn_n->dimensions.data[d];
		if (nn_d->assoc_id_in == nullptr) {
			printf(" (%d %d %d [null, %s])", nn_d->rows, nn_d->cols, nn_d->channels, nn_d->assoc_id_out);
		} else {
			printf(" (%d %d %d [%s, %s])", nn_d->rows, nn_d->cols, nn_d->channels, nn_d->assoc_id_in, nn_d->assoc_id_out);
		}
		if (d < dims_c - 1) printf(",");
	}
	printf("\n");
}

unsigned int nn_node_layer_width_get(struct nn_node *nn_n) {
	int dims_c = m_array_size(&nn_n->dimensions);
	unsigned int layer_width = 0;
	for (int d = 0; d < dims_c; d++) {
		layer_width += nn_n->dimensions.data[d].rows * nn_n->dimensions.data[d].cols * nn_n->dimensions.data[d].channels;
	}
	return layer_width;
}

void nn_node_externalise_evo(struct nn_node *nn_n, struct externaliser *e) {
	int dims_c = m_array_size(&nn_n->dimensions);
	int total_len =
			//total_len
			sizeof(int) +
			//id
			sizeof(int) + strlen(nn_n->id) +
			//node_type
			sizeof(int) +
			//output_mem_req
			sizeof(int) +
			//dims_c + dimensions
			sizeof(int) + dims_c * sizeof(struct nn_dimension);

	externaliser_init(e, total_len);

	externaliser_int_add(e, total_len);
	externaliser_str_add(e, nn_n->id, strlen(nn_n->id));
	externaliser_int_add(e, (int)nn_n->type);
	externaliser_int_add(e, nn_n->output_mem_req);
	externaliser_m_array_add(e, &nn_n->dimensions);
}
