/*
 * nn_edge_operation_fullyconnected.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "nn_edge_operation_fullyconnected.hpp"

#include <stdlib.h>
#include <stdio.h>

void nn_edge_operation_fullyconnected_init(struct nn_edge_operation *nn_e_op) {
	nn_e_op->type 				= NN_E_OP_T_FULLYCONNECTED;
	nn_e_op->operation_params 	= (void *) malloc(sizeof(struct nn_edge_operation_fullyconnected));
	nn_e_op->operation_mem_req	= 0;
}

void nn_edge_operation_fullyconnected_param_set(struct nn_edge_operation *nn_e_op, int nodes_c) {
	struct nn_edge_operation_fullyconnected *nn_e_op_fc = (struct nn_edge_operation_fullyconnected *) nn_e_op->operation_params;
	nn_e_op_fc->nodes_c = nodes_c;
}

void nn_edge_operation_fullyconnected_dump(struct nn_edge_operation_fullyconnected *nn_e_op_fc) {
	printf("FULLYCONNECTED, nodes_c: %d", nn_e_op_fc->nodes_c);
}
