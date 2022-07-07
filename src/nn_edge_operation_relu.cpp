/*
 * nn_edge_operation_relu.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "nn_edge_operation_relu.hpp"

void nn_edge_operation_relu_init(struct nn_edge_operation *nn_e_op) {
	nn_e_op->type 				= NN_E_OP_T_RELU;
	nn_e_op->operation_params 	= nullptr;
	nn_e_op->operation_mem_req	= 0;
}

