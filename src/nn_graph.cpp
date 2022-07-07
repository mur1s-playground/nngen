/*
 * nn_graph.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "nn_graph.hpp"

#include <vector>
#include <string>
#include <sstream>

#include "nn_edge_operation_conv2d.hpp"
#include "nn_edge_operation_maxpool2d.hpp"
#include "nn_edge_operation_relu.hpp"
#include "nn_edge_operation_fullyconnected.hpp"
#include "nn_edge_operation_softmax.hpp"
#include "nn_edge_operation_concat.hpp"
#include "nn_filter.hpp"
#include "nn_edge_operation.hpp"

#include "util.hpp"

struct m_array<struct nn_graph *> nn_graphs;

void nn_graphs_init() {
	m_array_init(&nn_graphs);
}

struct nn_graph *nn_graph_get(unsigned int id) {
	return nn_graphs.data[id];
}

void nn_graph_init(struct nn_graph *nn_g, const char *directory, const char *prefix) {
	nn_id_factory_init(&nn_g->nn_id_f);

	util_chararray_from_const(directory, &nn_g->directory);
	util_chararray_from_const(prefix, &nn_g->prefix);

	m_array_init(&nn_g->nodes);
	m_array_init(&nn_g->edges);

	m_array_init(&nn_g->segments);

	std::stringstream nodes_file_path;
	nodes_file_path << directory << "/" << prefix << "nodes";
	std::vector<std::string> nodes_file = util_file_read(nodes_file_path.str().c_str());
	for (int nf = 0; nf < nodes_file.size(); nf++) {
		if (nodes_file[nf].length() == 0) continue;
		std::vector<std::string> nodes_params = util_split(nodes_file[nf], ",");
		const char *node_id 	= nodes_params[0].c_str();
		const char *type_spec 	= nodes_params[1].c_str();
		enum nn_node_type type 	= NN_N_T_UNSPEC;
		struct nn_dimension nn_dim = {0, 0, 0};
		if (strcmp(type_spec, "input") == 0) {
			type = NN_N_T_INPUT;
			nn_dim.rows 	= std::stoi(nodes_params[2].c_str());
			nn_dim.cols 	= std::stoi(nodes_params[3].c_str());
			nn_dim.channels = std::stoi(nodes_params[4].c_str());
			nn_dim.assoc_id_in = nullptr;
			nn_id_factory_id_get(&nn_g->nn_id_f, &nn_dim.assoc_id_out);
		} else if (strcmp(type_spec, "throughput") == 0) {
			type = NN_N_T_THROUGHPUT;
		} else if (strcmp(type_spec, "output") == 0) {
			type = NN_N_T_OUTPUT;
		}
		printf("adding node to graph %s %d\n", node_id, m_array_size(&nn_g->nodes));
		nn_graph_node_add(nn_g, node_id, type, nn_dim);
	}

	std::stringstream edges_file_path;
	edges_file_path << directory << "/" << prefix << "edges";
	std::vector<std::string> edges_file = util_file_read(edges_file_path.str().c_str());
	for (int ef = 0; ef < edges_file.size(); ef++) {
		if (edges_file[ef].length() == 0) continue;
		std::vector<std::string> edges_params = util_split(edges_file[ef], ",");
		const char *edge_id 			= edges_params[0].c_str();
		const char *node_from_id 		= edges_params[1].c_str();
		const char *node_to_id			= edges_params[2].c_str();
		unsigned int operations_c		= std::stoi(edges_params[3].c_str());
		struct nn_edge *edge_current 	= nn_graph_edge_add(nn_g, edge_id, node_from_id, node_to_id);
		int params_position = 4;
		for (int o = 0; o < operations_c; o++) {
			struct nn_edge_operation edge_op;
			const char *op_type = edges_params[params_position].c_str();						params_position++;
			if (strcmp(op_type, "convolution_2d") == 0) {
				nn_edge_operation_conv2d_init(&edge_op);
				struct nn_edge_operation_conv2d *op_conv2d = (struct nn_edge_operation_conv2d *) edge_op.operation_params;
				int filters_total			= std::stoi(edges_params[params_position].c_str());	params_position++;
				int filters_counted			= 0;
				while (filters_counted < filters_total) {
					struct nn_filter nn_f;
					nn_filter_parse(&nn_f, edges_params, &params_position);
					nn_edge_operation_conv2d_filter_add(op_conv2d, nn_f);
					filters_counted += nn_f.filters_c;
				}
			} else if (strcmp(op_type, "maxpool_2d") == 0) {
				nn_edge_operation_maxpool2d_init(&edge_op);
				struct nn_edge_operation_maxpool2d *op_maxpool2d = (struct nn_edge_operation_maxpool2d *) edge_op.operation_params;
				int filters_total			= std::stoi(edges_params[params_position].c_str());	params_position++;
				int filters_counted			= 0;
				while (filters_counted < filters_total) {
					struct nn_filter nn_f;
					nn_filter_parse(&nn_f, edges_params, &params_position);
					nn_edge_operation_maxpool2d_filter_add(op_maxpool2d, nn_f);
					filters_counted += nn_f.filters_c;
				}
			} else if (strcmp(op_type, "relu") == 0) {
				nn_edge_operation_relu_init(&edge_op);
			} else if (strcmp(op_type, "fullyconnected") == 0) {
				nn_edge_operation_fullyconnected_init(&edge_op);
				int nodes_c = std::stoi(edges_params[params_position].c_str());	params_position++;
				nn_edge_operation_fullyconnected_param_set(&edge_op, nodes_c);
			} else if (strcmp(op_type, "softmax") == 0) {
				nn_edge_operation_softmax_init(&edge_op);
			} else if (strcmp(op_type, "concat") == 0) {
				nn_edge_operation_concat_init(&edge_op);
			}
			nn_edge_operation_add(edge_current, edge_op);
		}
	}
	m_array_push_back(&nn_graphs, nn_g);
}

void nn_graph_node_add(struct nn_graph *nn_g, const char *node_id, enum nn_node_type type, struct nn_dimension nn_dim) {
	int id_nid = nn_id_factory_id_add(&nn_g->nn_id_f, node_id);
	if (id_nid > -1) {
		struct nn_node nn_n;
		nn_node_init(&nn_n);
		nn_n.id = nn_g->nn_id_f.ids.data[id_nid];
		nn_n.type = type;
		if (nn_dim.rows != 0 && nn_dim.cols != 0 && nn_dim.channels != 0) {
			m_array_push_back(&nn_n.dimensions, nn_dim);
		}

		m_array_push_back(&nn_g->nodes, nn_n);
	} else {
		printf("error (nn_graph): node_id already exists\n");
	}
}

struct nn_edge *nn_graph_edge_add(struct nn_graph *nn_g, const char *edge_id, const char *node_from_id, const char *node_to_id) {
	int id_from_nid = nn_id_factory_nid_get(&nn_g->nn_id_f, node_from_id);
	int id_to_nid = nn_id_factory_nid_get(&nn_g->nn_id_f, node_to_id);
	if (id_from_nid == -1) {
		printf("error (nn_graph): node_from_id (%s) does not exist on edge (%s)\n", node_from_id, edge_id);
		return nullptr;
	}
	if (id_to_nid == -1) {
		printf("error (nn_graph): node_to_id (%s) does not exist on edge (%s)\n", node_to_id, edge_id);
		return nullptr;
	}
	int id_nid = nn_id_factory_id_add(&nn_g->nn_id_f, edge_id);
	if (id_nid > -1) {
		struct nn_edge nn_e;
		nn_edge_init(&nn_e);
		nn_e.id 		= nn_g->nn_id_f.ids.data[id_nid];
		nn_e.id_from 	= nn_g->nn_id_f.ids.data[id_from_nid];
		nn_e.id_to 		= nn_g->nn_id_f.ids.data[id_to_nid];

		m_array_push_back(&nn_g->edges, nn_e);
		int size = m_array_size(&nn_g->edges);
		return &nn_g->edges.data[size - 1];
	} else {
		printf("error (nn_graph): edge_id already exists\n");
	}
	return nullptr;
}

struct nn_node *nn_graph_node_get_by_id(struct nn_graph *nn_g, const char *node_id, unsigned int *out_nid) {
	struct nn_node *nn_res = nullptr;

	int nodes_c = m_array_size(&nn_g->nodes);
	for (int n = 0; n < nodes_c; n++) {
		if (strcmp(nn_g->nodes.data[n].id, node_id) == 0) {
			nn_res = &nn_g->nodes.data[n];
			if (out_nid != nullptr) {
				*out_nid = n;
			}
			break;
		}
	}
	return nn_res;
}

struct nn_edge *nn_graph_edge_find_outgoing(struct nn_graph *nn_g, const char *node_id, unsigned int *out_eid, int offset) {
	struct nn_edge *nn_e = nullptr;
	int edges_c = m_array_size(&nn_g->edges);
	for (int e = offset; e < edges_c; e++) {
		if (strcmp(nn_g->edges.data[e].id_from, node_id) == 0) {
			nn_e = &nn_g->edges.data[e];
			if (out_eid != nullptr) {
				*out_eid = e;
			}
			break;
		}
	}
	return nn_e;
}

void nn_graph_dump(struct nn_graph *nn_g) {
	int nodes_c = m_array_size(&nn_g->nodes);
	for (int n = 0; n < nodes_c; n++) {
		nn_node_dump(&nn_g->nodes.data[n]);
	}

	int edges_c = m_array_size(&nn_g->edges);
	for (int e = 0; e < edges_c; e++) {
		nn_edge_dump(&nn_g->edges.data[e]);
	}

	printf("output_mem_req: (1)->");
	util_size_format_print(nn_g->output_mem_req);
	printf(", (100)->");
	util_size_format_print(nn_g->output_mem_req * 100);
	printf(", (1000)->");
	util_size_format_print(nn_g->output_mem_req * 1000);
	printf(", (10000)->");
	util_size_format_print(nn_g->output_mem_req * 10000);
	printf(", (100000)->");
	util_size_format_print(nn_g->output_mem_req * 100000);
	printf("\nops_mem_req: ");
	util_size_format_print(nn_g->ops_mem_req);
	printf("\n");

	nn_graph_segments(nn_g);
}

void nn_graph_recalculate_dimensions(struct nn_graph *nn_g) {
	bool found_unspec_dim = true;
	int nodes_c = m_array_size(&nn_g->nodes);
	int edges_c = m_array_size(&nn_g->edges);

	nn_g->output_mem_req = 0;
	nn_g->ops_mem_req = 0;

	bool *set = (bool *) malloc(nodes_c * sizeof(bool));
	for (int n = 0; n < nodes_c; n++) {
		set[n] = false;
		struct nn_node *nn_n = &nn_g->nodes.data[n];
		nn_n->edges_in_c = 0;
		nn_n->edges_out_c = 0;
	}
	while (found_unspec_dim) {
		found_unspec_dim = false;
		for (int n = 0; n < nodes_c; n++) {
			struct nn_node *nn_n = &nn_g->nodes.data[n];
			if (nn_n->type == NN_N_T_INPUT) {
				set[n] = true;
				continue;
			}
			if (!set[n]) {
				found_unspec_dim = true;
				int edges_con_c 	= 0;
				int edges_con_c_rdy	= 0;
				std::vector<struct nn_edge *> from_edges = std::vector<struct nn_edge *>();
				std::vector<struct nn_node *> from_nodes = std::vector<struct nn_node *>();
				for (int e = 0; e < edges_c; e++) {
					struct nn_edge *nn_e = &nn_g->edges.data[e];
					if (strcmp(nn_e->id_to, nn_n->id) == 0) {
						edges_con_c++;
						unsigned int from_nid = -1;
						struct nn_node *nn_from = nn_graph_node_get_by_id(nn_g, nn_e->id_from, &from_nid);

						if (!set[from_nid]) break;

						from_edges.push_back(nn_e);
						from_nodes.push_back(nn_from);
						edges_con_c_rdy++;
					}
				}
				if (edges_con_c == edges_con_c_rdy) {
					for (int e = 0; e < from_edges.size(); e++) {
						struct nn_edge *nn_e 	= from_edges[e];
						struct nn_node *nn_from = from_nodes[e];
						nn_n->edges_in_c++;
						nn_from->edges_out_c++;

						unsigned int edge_ops_c = m_array_size(&nn_e->operations);

						unsigned int from_dims_c = m_array_size(&nn_from->dimensions);


						//for (int op = 0; op < edge_ops_c; op++) { //TMP: edge_ops_c == 1
						int op = 0;
						struct nn_edge_operation *nn_op = &nn_e->operations.data[op];
						nn_op->operation_mem_req = 0;

						for (int d = 0; d < from_dims_c; d++) {
								struct nn_dimension *nn_from_dim_cur = &nn_from->dimensions.data[d];
								if (nn_op->type == NN_E_OP_T_RELU || nn_op->type == NN_E_OP_T_SOFTMAX) {
									if (op == 0) {
										struct nn_dimension nn_dim_calc = {0, 0, 0};
										nn_dim_calc.rows 		= nn_from_dim_cur->rows;
										nn_dim_calc.cols 		= nn_from_dim_cur->cols;
										nn_dim_calc.channels 	= nn_from_dim_cur->channels;
										nn_dim_calc.assoc_id_in = nn_from_dim_cur->assoc_id_out;
										nn_id_factory_id_get(&nn_g->nn_id_f, &nn_dim_calc.assoc_id_out);

										m_array_push_back(&nn_n->dimensions, nn_dim_calc);
									} //else { }
								} else if (nn_op->type == NN_E_OP_T_CONV2D) {
									struct nn_edge_operation_conv2d *conv2d = (struct nn_edge_operation_conv2d *) nn_op->operation_params;
									int filters_t_c = m_array_size(&conv2d->filters);
									if (op == 0) {
										for (int f = 0; f < filters_t_c; f++) {
											struct nn_filter *nn_f = &conv2d->filters.data[f];
											struct nn_dimension nn_dim_calc = {0, 0, 0};
											nn_dim_calc.rows		= (nn_from_dim_cur->rows - (nn_f->kernel_size[0]/nn_f->dilation[0]) + 1) / nn_f->stride[0];
											nn_dim_calc.cols		= (nn_from_dim_cur->cols - (nn_f->kernel_size[1]/nn_f->dilation[1]) + 1) / nn_f->stride[1];
											nn_dim_calc.channels 	= nn_from_dim_cur->channels * nn_f->filters_c;
											nn_dim_calc.assoc_id_in = nn_from_dim_cur->assoc_id_out;
											nn_id_factory_id_get(&nn_g->nn_id_f, &nn_dim_calc.assoc_id_out);

											m_array_push_back(&nn_n->dimensions, nn_dim_calc);
											if (d == 0) nn_op->operation_mem_req += nn_filter_mem_req(nn_f);
										}
									}
								} else if (nn_op->type == NN_E_OP_T_MAXPOOL2D) {
									struct nn_edge_operation_maxpool2d *maxpool2d = (struct nn_edge_operation_maxpool2d *) nn_op->operation_params;
									int filters_t_c = m_array_size(&maxpool2d->filters);
									if (op == 0) {
										for (int f = 0; f < filters_t_c; f++) {
											struct nn_filter *nn_f = &maxpool2d->filters.data[f];
											struct nn_dimension nn_dim_calc = {0, 0, 0};
											nn_dim_calc.rows		= (nn_from_dim_cur->rows - (nn_f->kernel_size[0]/nn_f->dilation[0]) + 1) / nn_f->stride[0];
											nn_dim_calc.cols		= (nn_from_dim_cur->cols - (nn_f->kernel_size[1]/nn_f->dilation[1]) + 1) / nn_f->stride[1];
											nn_dim_calc.channels 	= nn_from_dim_cur->channels * nn_f->filters_c;
											nn_dim_calc.assoc_id_in = nn_from_dim_cur->assoc_id_out;
											nn_id_factory_id_get(&nn_g->nn_id_f, &nn_dim_calc.assoc_id_out);

											m_array_push_back(&nn_n->dimensions, nn_dim_calc);
											if (d == 0) nn_op->operation_mem_req += nn_filter_mem_req(nn_f);
										}
									}
								} else if (nn_op->type == NN_E_OP_T_FULLYCONNECTED) {
									struct nn_edge_operation_fullyconnected *fullyconnected = (struct nn_edge_operation_fullyconnected *) nn_op->operation_params;
									int fc_nodes_c = fullyconnected->nodes_c;
									if (op == 0) {
										struct nn_dimension nn_dim_calc = {0, 0, 0};
										nn_dim_calc.rows 		= fc_nodes_c;
										nn_dim_calc.cols 		= 1;
										nn_dim_calc.channels 	= 1;
										nn_dim_calc.assoc_id_in = nn_from_dim_cur->assoc_id_out;
										nn_id_factory_id_get(&nn_g->nn_id_f, &nn_dim_calc.assoc_id_out);

										m_array_push_back(&nn_n->dimensions, nn_dim_calc);
										nn_op->operation_mem_req += (fc_nodes_c * nn_from_dim_cur->rows * nn_from_dim_cur->cols * nn_from_dim_cur->channels + fc_nodes_c) * sizeof(float);
									}
								} else if (nn_op->type == NN_E_OP_T_CONCAT) {
									if (op == 0) {
										if (d == 0) {
											struct nn_dimension nn_dim_calc = {0, 0, 0};
											nn_dim_calc.rows 		= nn_from_dim_cur->rows;
											nn_dim_calc.cols 		= nn_from_dim_cur->cols;
											nn_dim_calc.channels 	= nn_from_dim_cur->channels;
											nn_dim_calc.assoc_id_in = nn_from_dim_cur->assoc_id_out;
											nn_id_factory_id_get(&nn_g->nn_id_f, &nn_dim_calc.assoc_id_out);

											m_array_push_back(&nn_n->dimensions, nn_dim_calc);
										} else {
											struct nn_dimension *nn_dim_calc = &nn_n->dimensions.data[0];
											if (nn_dim_calc->rows == nn_from_dim_cur->rows && nn_dim_calc->cols == nn_from_dim_cur->cols) {
												nn_dim_calc->channels += nn_from_dim_cur->channels;
											} else {
												nn_dim_calc->rows *= nn_dim_calc->cols * nn_dim_calc->channels;
												nn_dim_calc->cols = 1;
												nn_dim_calc->channels = 1;
												nn_dim_calc->rows += nn_from_dim_cur->rows * nn_from_dim_cur->cols * nn_from_dim_cur->channels;
											}
										}
									}
								}

						}

						nn_g->ops_mem_req += nn_op->operation_mem_req;
						//}
					}
					set[n] = true;
				}
			}
		}
	}
	for (int n = 0; n < nodes_c; n++) {
		struct nn_node *nn_n = &nn_g->nodes.data[n];
		unsigned int dims_c = m_array_size(&nn_n->dimensions);
		nn_n->output_mem_req = 0;
		for (int d = 0; d < dims_c; d++) {
			struct nn_dimension *nn_dim = &nn_n->dimensions.data[d];
			nn_n->output_mem_req += nn_dim->rows * nn_dim->cols * nn_dim->channels * sizeof(float);
		}
		nn_g->output_mem_req += nn_n->output_mem_req;
	}
}

#include "nn_scheduler.hpp"

void nn_graph_segments_dump(struct nn_graph *nn_g) {
	m_array<int> *segments = &nn_g->segments;
	int segments_c = m_array_size(segments);
	for (int s = 0; s < segments_c; s++) {
		int segment_id = segments->data[s];
		if (segment_id == -1) {
			printf("\ninput->");
		} else if (segment_id == -2) {
			printf("\ninter->");
		} else if (segment_id == -3) {
			printf("output");
		} else {
			struct nn_node *nn_n = &nn_g->nodes.data[segment_id];
			printf("[%d]:%s ", segment_id, nn_n->id);
		}
	}
	printf("\n");

	struct nn_scheduler nn_s;
	nn_scheduler_init(&nn_s, nn_g, NN_S_T_INFERENCE, 0, 0, 0, 0, 0, 0);
	nn_scheduler_schedule(&nn_s);
}

void nn_graph_traverse_till_barrier(struct nn_graph *nn_g, m_array<int> *barrier, struct nn_edge *nn_e, m_array<int> *starts);

void nn_graph_segments(struct nn_graph *nn_g) {
	m_array<int> graph_barriers;
	m_array_init(&graph_barriers);

	m_array<int> inputs;
	m_array_init(&inputs);

	m_array<int> starts;
	m_array_init(&starts);

	m_array<int> *segments = &nn_g->segments;
	m_array_destroy(segments);

	int nodes_c = m_array_size(&nn_g->nodes);
	for (int n = 0; n < nodes_c; n++) {
		struct nn_node *nn_n = &nn_g->nodes.data[n];
		if (nn_n->edges_in_c > 1 || nn_n->edges_out_c > 1 || nn_n->edges_out_c == 0) {
			m_array_push_back(&graph_barriers, n);
		}
		if (nn_n->edges_in_c == 0) {
			m_array_push_back(&inputs, n);
		}
	}

	int inputs_c = m_array_size(&inputs);
	for (int i = 0; i < inputs_c; i++) {
		int input_idx = inputs.data[i];
		struct nn_node *nn_n = &nn_g->nodes.data[input_idx];
//		printf("input->%s", nn_n->id);
		m_array_push_back(segments, -1);
		m_array_push_back(segments, input_idx);

		unsigned int out_eid = -1;
		struct nn_edge *nn_e_out = nn_graph_edge_find_outgoing(nn_g, nn_n->id, &out_eid, 0);
		while (nn_e_out != nullptr) {
			nn_graph_traverse_till_barrier(nn_g, &graph_barriers, nn_e_out, &starts);

			nn_e_out = nn_graph_edge_find_outgoing(nn_g, nn_n->id, &out_eid, out_eid+1);
		}
	}

	m_array_destroy(&graph_barriers);
	m_array_destroy(&inputs);
	m_array_destroy(&starts);

	nn_graph_segments_dump(nn_g);
}

void nn_graph_traverse_till_barrier(struct nn_graph *nn_g, m_array<int> *barrier, struct nn_edge *nn_e, m_array<int> *starts) {
	m_array<int> *segments = &nn_g->segments;
	unsigned int next_nid = -1;
	struct nn_node *nn_next = nn_graph_node_get_by_id(nn_g, nn_e->id_to, &next_nid);
	if (nn_next != nullptr) {
		m_array_push_back(segments, (int)next_nid);
//		printf("->%s", nn_next->id);
	}
	if (m_array_contains(barrier, (int)next_nid) > -1) {
		if (nn_next->edges_out_c == 0) {
			m_array_push_back(segments, -3);
//			printf("->output\n");
		} else {
//			printf("\n");
			if (m_array_contains(starts, (int)next_nid) == -1) {
				m_array_push_back(starts, (int)next_nid);
				unsigned int out_eid = -1;
				struct nn_edge *nn_e_out = nn_graph_edge_find_outgoing(nn_g, nn_next->id, &out_eid, 0);
				while (nn_e_out != nullptr) {
					m_array_push_back(segments, -2);
					m_array_push_back(segments, (int) next_nid);
//					printf("inter->%s", nn_next->id);
					nn_graph_traverse_till_barrier(nn_g, barrier, nn_e_out, starts);
					nn_e_out = nn_graph_edge_find_outgoing(nn_g, nn_next->id, &out_eid, out_eid+1);
				}
			}
		}
	} else {
		unsigned int out_eid = -1;
		struct nn_edge *nn_e_out = nn_graph_edge_find_outgoing(nn_g, nn_next->id, &out_eid, 0);
		if (nn_e_out != nullptr) {
			nn_graph_traverse_till_barrier(nn_g, barrier, nn_e_out, starts);
		}
	}
}

void nn_graph_generate_kernels(struct nn_graph *nn_g) {

}
