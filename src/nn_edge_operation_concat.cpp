/*
 * nn_edge_operation_concat.cpp
 *
 *  Created on: Jun 26, 2022
 *      Author: mur1
 */

#include "nn_edge_operation_concat.hpp"

void nn_edge_operation_concat_init(struct nn_edge_operation *nn_e_op) {
	nn_e_op->type 				= NN_E_OP_T_CONCAT;
	nn_e_op->operation_params 	= nullptr;
	nn_e_op->operation_mem_req	= 0;
}



