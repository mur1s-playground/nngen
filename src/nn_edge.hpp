/*
 * nn_edge.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_EDGE_HPP_
#define SRC_NN_EDGE_HPP_

#include "m_array.hpp"
#include "nn_edge_operation.hpp"
#include "externaliser.hpp"

struct nn_edge {
	char *id;
	char *id_from;
	char *id_to;

	struct m_array<struct nn_edge_operation> operations;
};

void nn_edge_init(struct nn_edge *nn_e);
void nn_edge_operation_add(struct nn_edge *nn_e, struct nn_edge_operation nn_edge_op);

void nn_edge_dump(struct nn_edge *nn_e);

void nn_edge_externalise_evo(struct nn_edge *nn_e, struct externaliser *e);


#endif /* SRC_NN_EDGE_HPP_ */
