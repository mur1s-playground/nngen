/*
 * nn_edge_operation_conv2d.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "nn_edge_operation_conv2d.hpp"

#include <stdlib.h>
#include <cstring>

#include "nn_filter.hpp"

void nn_edge_operation_conv2d_init(struct nn_edge_operation *nn_e_op) {
	nn_e_op->type 				= NN_E_OP_T_CONV2D;
	nn_e_op->operation_params 	= (void *) malloc(sizeof(struct nn_edge_operation_conv2d));
	struct nn_edge_operation_conv2d *op_conv2d = (struct nn_edge_operation_conv2d *) nn_e_op->operation_params;
	m_array_init(&op_conv2d->filters);
	nn_e_op->operation_mem_req	= 0;
}

void nn_edge_operation_conv2d_filter_add(struct nn_edge_operation_conv2d *nn_e_op_conv2d, struct nn_filter nn_f) {
	m_array_push_back(&nn_e_op_conv2d->filters, nn_f);
}

void nn_edge_operation_conv2d_dump(struct nn_edge_operation_conv2d *nn_e_op_conv2d) {
	int filters_c = m_array_size(&nn_e_op_conv2d->filters);
	printf("CONV2D, total_filters_c: %d, ", filters_c);
	for (int f = 0; f < filters_c; f++) {
		nn_filter_dump(&nn_e_op_conv2d->filters.data[f]);
		if (f < filters_c - 1) printf(", ");
	}
}
