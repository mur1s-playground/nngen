/*
 * nn_schedule_evolution.hpp
 *
 *  Created on: Jun 27, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_SCHEDULE_EVOLUTION_HPP_
#define SRC_NN_SCHEDULE_EVOLUTION_HPP_

#include "m_array.hpp"
#include "nn_scheduler.hpp"

/*
struct nn_schedule_genetic {

};

struct nn_schedule_evolution {
	struct nn_scheduler *nn_s;

	m_array<struct nn_schedule_genetic> pool;
};
*/



#endif /* SRC_NN_SCHEDULE_EVOLUTION_HPP_ */

/*
#include "nn_schedule_evolution.hpp"

#include "externaliser.hpp"
#include "nn_edge.hpp"
#include "nn_node.hpp"
/*

void nn_schedule_evolve(struct nn_schedule_evolution *nn_s_e) {
	struct nn_graph *nn_g = nn_s_e->nn_s->nn_g;

	//all nodes
	m_array<struct externaliser> nodes_externalisers;
	m_array_init(&nodes_externalisers);

	int nodes_size_total = 0;
	int nodes_c = m_array_size(&nn_g->nodes);
	for (int n = 0; n < nodes_c; n++) {
		struct nn_node *nn_n = &nn_g->nodes.data[n];
		struct externaliser e;
		nn_node_externalise_evo(nn_n, &e);
		m_array_push_back(&nodes_externalisers, e);
		nodes_size_total += e.position;
	}

	struct externaliser all_nodes; //<---
	externaliser_init(&all_nodes, sizeof(int) + nodes_size_total);
	externaliser_int_add(&all_nodes, sizeof(int) + nodes_size_total);
	for (int n = 0; n < nodes_c; n++) {
		externaliser_raw_add(&all_nodes, nodes_externalisers.data[n].out_ext, nodes_externalisers.data[n].position);
		externaliser_destroy(&nodes_externalisers.data[n]);
	}

	m_array_destroy(&nodes_externalisers);

	//all edges
	m_array<struct externaliser> edges_externalisers;
	m_array_init(&edges_externalisers);

	int edges_size_total = 0;
	int edges_c = m_array_size(&nn_g->edges);
	for (int e = 0; e < edges_c; e++) {
		struct nn_edge *nn_e = &nn_g->edges.data[e];
		struct externaliser e;
		nn_edge_externalise_evo(nn_e, &e);
		m_array_push_back(&edges_externalisers, e);
		edges_size_total += e.position;
	}

	struct externaliser all_edges; //<---
	externaliser_init(&all_edges, sizeof(int) + edges_size_total);
	externaliser_int_add(&all_edges, sizeof(int) + edges_size_total);
	for (int e = 0; e < edges_c; e++) {
		externaliser_raw_add(&all_edges, edges_externalisers.data[e].out_ext, edges_externalisers.data[e].position);
		externaliser_destroy(&edges_externalisers.data[e]);
	}

	m_array_destroy(&edges_externalisers);

	char *all_nodes_dev = nullptr;
	externaliser_2device(&all_nodes, (void **)&all_nodes_dev);

	char *all_edges_dev = nullptr;
	externaliser_2device(&all_edges, (void **)&all_edges_dev);

	// [(h) load r-n] [(h->d) r-n] [(h) load r-op] [(h->d) r-op],
	// edge proc,
	// [(d->0) r-n] [(d->0) r-op] [(d->h) n-r [(d->0) n-r] [(h) save n-r [(h->0) n-r]]
	//
	//

	int segments_c = m_array_size(&nn_g->segments);
	m_array<int> segments_starts;
	m_array_init(&segments_starts);
	for (int s = 0; s < segments_c; s++) {
		int segment_value = nn_g->segments.data[s];
		if (segment_value != -3 && segment_value < 0) {
			m_array_push_back(&segments_starts, s);
		}
	}
	int segments_starts_c = m_array_size(&segments_starts);


}

*/
