/*
 * nn_edge.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "nn_edge.hpp"

void nn_edge_init(struct nn_edge *nn_e) {
	nn_e->id 		= nullptr;
	nn_e->id_from 	= nullptr;
	nn_e->id_to		= nullptr;
	m_array_init(&nn_e->operations);
}

void nn_edge_operation_add(struct nn_edge *nn_e, struct nn_edge_operation nn_edge_op) {
	m_array_push_back(&nn_e->operations, nn_edge_op);
}

void nn_edge_dump(struct nn_edge *nn_e) {
	int operations_c = m_array_size(&nn_e->operations);
	printf("edge (%s):  %s -> %s, ops_c: %d, ", nn_e->id, nn_e->id_from, nn_e->id_to, operations_c);
	for (int o = 0; o < operations_c; o++) {
		nn_edge_operation_dump(&nn_e->operations.data[o]);
		if (o < operations_c - 1) printf(", ");
	}
	printf("\n");
}

void nn_edge_externalise_evo(struct nn_edge *nn_e, struct externaliser *e) {
	int ops_c = m_array_size(&nn_e->operations);
	int total_len =
			//total_len
			sizeof(int) +
			//id
			sizeof(int) + strlen(nn_e->id) +
			//id_from
			sizeof(int) + strlen(nn_e->id_from) +
			//id_to
			sizeof(int) + strlen(nn_e->id_to) +
			//ops_c + ops_c * operations
			sizeof(int) + ops_c * sizeof(struct nn_edge_operation);

	externaliser_init(e, total_len);

	externaliser_int_add(e, total_len);
	externaliser_str_add(e, nn_e->id, strlen(nn_e->id));
	externaliser_str_add(e, nn_e->id_from, strlen(nn_e->id_from));
	externaliser_str_add(e, nn_e->id_to, strlen(nn_e->id_to));
	externaliser_m_array_add(e, &nn_e->operations);
}
