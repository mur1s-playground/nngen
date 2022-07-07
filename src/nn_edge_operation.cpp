/*
 * nn_edge_operation.cpp
 *
 *  Created on: Jun 25, 2022
 *      Author: mur1
 */

#include "nn_edge_operation.hpp"

#include <stdio.h>

#include "nn_edge_operation_conv2d.hpp"
#include "nn_edge_operation_maxpool2d.hpp"
#include "nn_edge_operation_fullyconnected.hpp"
#include "util.hpp"

void nn_edge_operation_dump(struct nn_edge_operation *nn_e_op) {
	printf("mem_req: ");
	util_size_format_print(nn_e_op->operation_mem_req);
	printf(", ");
	switch (nn_e_op->type) {
		case NN_E_OP_T_CONV2D:
			nn_edge_operation_conv2d_dump((struct nn_edge_operation_conv2d *) nn_e_op->operation_params);
			break;
		case NN_E_OP_T_MAXPOOL2D:
			nn_edge_operation_maxpool2d_dump((struct nn_edge_operation_maxpool2d *) nn_e_op->operation_params);
			break;
		case NN_E_OP_T_FULLYCONNECTED:
			nn_edge_operation_fullyconnected_dump((struct nn_edge_operation_fullyconnected *) nn_e_op->operation_params);
			break;
		case NN_E_OP_T_RELU:
			printf("RELU");
			break;
		case NN_E_OP_T_SOFTMAX:
			printf("SOFTMAX");
			break;
		case NN_E_OP_T_CONCAT:
			printf("CONCAT");
			break;
	}
}

