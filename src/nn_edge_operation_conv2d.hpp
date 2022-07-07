/*
 * nn_edge_operation_conv2d.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_EDGE_OPERATION_CONV2D_HPP_
#define SRC_NN_EDGE_OPERATION_CONV2D_HPP_

#include "nn_edge_operation.hpp"
#include "nn_filter.hpp"
#include "m_array.hpp"

struct nn_edge_operation_conv2d {
	struct m_array<struct nn_filter> filters;
};

void nn_edge_operation_conv2d_init(struct nn_edge_operation *nn_e_op);
void nn_edge_operation_conv2d_filter_add(struct nn_edge_operation_conv2d *nn_e_op_conv2d, struct nn_filter nn_f);

void nn_edge_operation_conv2d_dump(struct nn_edge_operation_conv2d *nn_e_op_conv2d);

#endif /* SRC_NN_EDGE_OPERATION_CONV2D_HPP_ */
