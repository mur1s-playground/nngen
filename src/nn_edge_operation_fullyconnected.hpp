/*
 * nn_edge_operation_fullyconnected.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_EDGE_OPERATION_FULLYCONNECTED_HPP_
#define SRC_NN_EDGE_OPERATION_FULLYCONNECTED_HPP_

#include "nn_edge_operation.hpp"

struct nn_edge_operation_fullyconnected {
	unsigned int nodes_c;
};

void nn_edge_operation_fullyconnected_init(struct nn_edge_operation *nn_e_op);
void nn_edge_operation_fullyconnected_param_set(struct nn_edge_operation *nn_e_op, int nodes_c);

void nn_edge_operation_fullyconnected_dump(struct nn_edge_operation_fullyconnected *nn_e_op_fc);

#endif /* SRC_NN_EDGE_OPERATION_FULLYCONNECTED_HPP_ */
