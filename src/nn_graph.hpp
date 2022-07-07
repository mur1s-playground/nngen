/*
 * nn_graph.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_GRAPH_HPP_
#define SRC_NN_GRAPH_HPP_


#include "nn_node.hpp"
#include "nn_edge.hpp"
#include "m_array.hpp"
#include "nn_id_factory.hpp"
#include "nn_dimension.hpp"

struct nn_graph {
	char 							*directory;
	char 							*prefix;

	struct nn_id_factory 			nn_id_f;

	struct m_array<struct nn_node>	nodes;
	struct m_array<struct nn_edge>	edges;

	unsigned int					output_mem_req;
	unsigned int					ops_mem_req;

	struct m_array<int>				segments;
};

void nn_graphs_init();
struct nn_graph *nn_graph_get(unsigned int id);

void nn_graph_init(struct nn_graph *nn_g, const char *directory, const char *prefix);
void nn_graph_node_add(struct nn_graph *nn_g, const char *node_id, enum nn_node_type type, struct nn_dimension nn_dim);
struct nn_edge *nn_graph_edge_add(struct nn_graph *nn_g, const char *edge_id, const char *node_from_id, const char *node_to_id);
struct nn_node *nn_graph_node_get_by_id(struct nn_graph *nn_g, const char *node_id, unsigned int *out_nid = nullptr);
struct nn_edge *nn_graph_edge_find_outgoing(struct nn_graph *nn_g, const char *node_id, unsigned int *out_eid = nullptr, int offset = 0);

void nn_graph_recalculate_dimensions(struct nn_graph *nn_g);
void nn_graph_segments(struct nn_graph *nn_g);

void nn_graph_dump(struct nn_graph *nn_g);

#endif /* SRC_NN_GRAPH_HPP_ */
