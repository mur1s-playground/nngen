/*
 * nn_edge_operation.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_EDGE_OPERATION_HPP_
#define SRC_NN_EDGE_OPERATION_HPP_

enum nn_edge_operation_type {
	NN_E_OP_T_CONV2D,
	NN_E_OP_T_MAXPOOL2D,
	NN_E_OP_T_FULLYCONNECTED,
	NN_E_OP_T_RELU,
	NN_E_OP_T_SOFTMAX,
	NN_E_OP_T_CONCAT
};

struct nn_edge_operation {
	enum nn_edge_operation_type type;

	unsigned int operation_mem_req;
	void *operation_params;
};

void nn_edge_operation_dump(struct nn_edge_operation *nn_e_op);

#endif /* SRC_NN_EDGE_OPERATION_HPP_ */
