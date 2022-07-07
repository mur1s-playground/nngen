/*
 * nn_scheduler.cpp
 *
 *  Created on: Jun 27, 2022
 *      Author: mur1
 */

#include "nn_scheduler.hpp"

#include "m_array.hpp"
#include "util.hpp"
#include "nn_edge_operation.hpp"
#include "nn_edge_operation_fullyconnected.hpp"
#include "nn_edge_operation_conv2d.hpp"
#include "nn_edge_operation_maxpool2d.hpp"

#include <sstream>
#include <map>

#include <iostream>

void nn_scheduler_init(struct nn_scheduler *nn_s, struct nn_graph *nn_g, enum nn_scheduler_target target, unsigned int total_memory_host_max, unsigned int total_memory_device_max_buffer, unsigned int total_memory_device_max_ops, unsigned int graph_branches_parallel_max, unsigned int input_batch_min, unsigned int input_batch_max) {
	nn_s->nn_g 								= nn_g;
	nn_s->target 							= target;
	/*
	nn_s->total_memory_host_max				= total_memory_host_max;
	nn_s->total_memory_device_max_buffer	= total_memory_device_max_buffer;
	nn_s->total_memory_device_max_ops		= total_memory_device_max_ops;
	nn_s->graph_branches_parallel_max		= graph_branches_parallel_max;
	nn_s->input_batch_min					= input_batch_min;
	nn_s->input_batch_max					= input_batch_max;
	*/
}

void nn_scheduler_schedule(struct nn_scheduler *nn_s) {
	struct nn_graph *nn_g = nn_s->nn_g;
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

	m_array<int> segments_groups;
	m_array_init(&segments_groups);
	m_array_push_back(&segments_groups, 1);
	m_array_push_back(&segments_groups, 0);
	int segments_inserted = 0;
	bool *inserted = (bool *) malloc(segments_starts_c * sizeof(bool));
	for (int ss = 0; ss < segments_starts_c; ss++) {
		inserted[ss] = false;
	}

	int segment_group_count_idx = 0;
	int last_segment_group_start = 1;
	int max_elements_per_group = 0;
	while (segments_inserted < segments_starts_c) {
		printf("segments_inserted %d, segment_starts_c %d\n", segments_inserted, segments_starts_c);
		for (int ss = 0; ss < segments_starts_c; ss++) {
			if (!inserted[ss]) {
				int segment_start = segments_starts.data[ss];
				if (nn_g->segments.data[segment_start] == -1) { //input
					printf("inserting input\n");
					m_array_push_back(&segments_groups, ss);
					segments_groups.data[last_segment_group_start]++;
					if (segments_groups.data[last_segment_group_start] > max_elements_per_group) max_elements_per_group = segments_groups.data[last_segment_group_start];
					inserted[ss] = true;
					segments_inserted++;
				} else if (nn_g->segments.data[segment_start] == -2) { //inter
					int nid_req = nn_g->segments.data[segment_start+1];
					struct nn_node *nn_n = &nn_g->nodes.data[nid_req];
					int nid_req_c = nn_n->edges_in_c;
					int nid_req_found = 0;

					printf("nid_req: %d, nid_req_c: %d\n", nid_req, nid_req_c);

					int segment_groups_c = segments_groups.data[segment_group_count_idx]-1;
					int sg_pos = 1;
					for (int sg = 0; sg < segment_groups_c; sg++) {
						int segment_group_elements_c = segments_groups.data[sg_pos];	sg_pos++;
						printf("checking group %d/%d %d ", sg, segment_groups_c, segment_group_elements_c);
						for (int seg = 0; seg < segment_group_elements_c; seg++) {
							int segment_start_idx = segments_groups.data[sg_pos];		sg_pos++;
							int first_node_idx = segments_starts.data[segment_start_idx] + 1;
							for (int ssi = first_node_idx; ssi < segments_c; ssi++) {
								if (nn_g->segments.data[ssi] < 0) break;
								printf("%d ", nn_g->segments.data[ssi]);
								if (nn_g->segments.data[ssi] == nid_req) {
									nid_req_found++;
								}
								if (nid_req_found == nid_req_c) {
									m_array_push_back(&segments_groups, ss);
									segments_groups.data[last_segment_group_start]++;
									if (segments_groups.data[last_segment_group_start] > max_elements_per_group) max_elements_per_group = segments_groups.data[last_segment_group_start];
									inserted[ss] = true;
									segments_inserted++;
									printf("inserting inter\n");
									break;
								}
							}
							if (nid_req_found == nid_req_c) break;
						}
						printf("\n");
						if (nid_req_found == nid_req_c) break;
					}
					printf("\n");
				}
			}
		}
		if (segments_inserted < segments_starts_c) {
			segments_groups.data[segment_group_count_idx]++;
			m_array_push_back(&segments_groups, 0);
			last_segment_group_start = m_array_size(&segments_groups) - 1;
			if (segments_groups.data[segment_group_count_idx] > 6) break;
		}
	}

	m_array<int> max_buffer_sizes;
	m_array_init(&max_buffer_sizes);
	for (int ms = 0; ms < max_elements_per_group; ms++) {
		m_array_push_back(&max_buffer_sizes, 0);
		m_array_push_back(&max_buffer_sizes, 0);
	}

	int segment_groups_c = segments_groups.data[segment_group_count_idx];
	int sg_pos = 1;
	for (int sg = 0; sg < segment_groups_c; sg++) {
		int segment_group_elements_c = segments_groups.data[sg_pos];	sg_pos++;
		for (int seg = 0; seg < segment_group_elements_c; seg++) {
			int segment_start_idx = segments_groups.data[sg_pos];		sg_pos++;
			int value0 = max_buffer_sizes.data[2 * seg + 0];
			int value1 = max_buffer_sizes.data[2 * seg + 1];
			int first_node_idx = segments_starts.data[segment_start_idx] + 1;
			for (int ssi = first_node_idx; ssi < segments_c; ssi++) {
				printf("%d %d %d %d\n", sg, seg, ssi, nn_g->segments.data[ssi]);
				if (nn_g->segments.data[ssi] < 0) break;
				struct nn_node *nn_n = &nn_g->nodes.data[nn_g->segments.data[ssi]];
				if (nn_n->output_mem_req > value0) max_buffer_sizes.data[2 * seg + 0] = nn_n->output_mem_req;
				if (nn_n->output_mem_req > value1) max_buffer_sizes.data[2 * seg + 1] = nn_n->output_mem_req;
			}
		}
	}

	sg_pos = 1;
	printf("segment_groups_c %d\n", segment_groups_c);
	for (int sg = 0; sg < segment_groups_c; sg++) {
		int segment_group_elements_c = segments_groups.data[sg_pos];	sg_pos++;
		printf("\tsegment_group %d, c %d\n", sg, segment_group_elements_c);
		for (int seg = 0; seg < segment_group_elements_c; seg++) {
			int segment_start_idx = segments_groups.data[sg_pos];		sg_pos++;
			printf("\t\telement %d", segment_start_idx);
			int value0 = max_buffer_sizes.data[2 * seg + 0];
			int value1 = max_buffer_sizes.data[2 * seg + 1];
			printf(" req0 req1 %d %d ", value0, value1);
		}
		printf("\n");
	}

	int batch_size = 10000;

	std::stringstream main_cpp;
	std::stringstream net_execution;

	std::vector<std::string> main_cpp_in = util_file_read("src.in/main.cpp.in");

	//edges
	std::stringstream edges_hpp;
	std::stringstream edges_hpp_edges_functions;
	std::stringstream edges_cu;
	std::stringstream edges_cu_edges_functions;

	std::vector<std::string> edges_hpp_edges_functions_in = util_file_read("src.in/edges.hpp.edges_functions.in");
	std::vector<std::string> edges_cu_edges_functions_fullyconnected_in = util_file_read("src.in/edges.cu.edges_functions.fullyconnected.in");
	std::vector<std::string> edges_cu_edges_functions_relu_in = util_file_read("src.in/edges.cu.edges_functions.relu.in");
	std::vector<std::string> edges_cu_edges_functions_softmax_in = util_file_read("src.in/edges.cu.edges_functions.softmax.in");
	std::vector<std::string> edges_cu_edges_functions_concat_in = util_file_read("src.in/edges.cu.edges_functions.concat.in");
	std::vector<std::string> edges_cu_edges_functions_filter_in = util_file_read("src.in/edges.cu.edges_functions.filter.in");
	std::vector<std::string> edges_cu_edges_functions_maxpool2d_in = util_file_read("src.in/edges.cu.edges_functions.maxpool2d.in");
	std::vector<std::string> edges_cu_edges_functions_in = util_file_read("src.in/edges.cu.edges_functions.in");

	std::stringstream buffer_alloc_cpp_edges_params_ptrs;
	std::stringstream buffer_alloc_hpp_edges_params_ptrs;
	std::stringstream buffer_alloc_content;
	std::stringstream buffer_fill_content;

	std::vector<std::string> buffer_alloc_cpp_edges_params_ptrs_in = util_file_read("src.in/buffer_alloc.cpp.edges_params_ptrs.in");
	std::vector<std::string> buffer_alloc_hpp_edges_params_ptrs_in = util_file_read("src.in/buffer_alloc.hpp.edges_params_ptrs.in");
	std::vector<std::string> buffer_alloc_content_in = util_file_read("src.in/buffer_alloc.cpp.buffer_alloc.in");
	std::vector<std::string> buffer_fill_content_in = util_file_read("src.in/buffer_alloc.cpp.buffer_fill.in");

	std::map<char *, unsigned int> edge_id_2_kernel_c = std::map<char *, unsigned int>();

	int edges_c = m_array_size(&nn_g->edges);
	sg_pos = 1;
	for (int sg = 0; sg < segment_groups_c; sg++) {
		int segment_group_elements_c = segments_groups.data[sg_pos];	sg_pos++;
		for (int seg = 0; seg < segment_group_elements_c; seg++) {
			std::vector<std::string> buffer_swap_ptrs;
			std::stringstream buffer_swap_ptrs_0;
			buffer_swap_ptrs_0 << "dev_ptr_" << seg << "_" << 0;
			std::stringstream buffer_swap_ptrs_1;
			buffer_swap_ptrs_1 << "dev_ptr_" << seg << "_" << 1;
			buffer_swap_ptrs.push_back(buffer_swap_ptrs_0.str());
			buffer_swap_ptrs.push_back(buffer_swap_ptrs_1.str());
			bool swap = true;

			std::stringstream cuda_stream_name;
			cuda_stream_name << "cuda_streams[" << seg << "]";

			int segment_start_idx = segments_groups.data[sg_pos];		sg_pos++;
			int first_node_idx = segments_starts.data[segment_start_idx] + 1;
			for (int ssi = first_node_idx; ssi < segments_c - 1; ssi++) {
				if (nn_g->segments.data[ssi+1] < 0) break;
				struct nn_node *nn_n_from 	= &nn_g->nodes.data[nn_g->segments.data[ssi    ]];
				struct nn_node *nn_n_to 	= &nn_g->nodes.data[nn_g->segments.data[ssi + 1]];
				for (int e = 0; e < edges_c; e++) {
					struct nn_edge *nn_e = &nn_g->edges.data[e];
					/*
					printf("edge_id_from: %s, ", nn_e->id_from);
					printf("node_id_from: %s, ", nn_n_from->id);
					printf("edge_id_to: %s, ", nn_e->id_to);
					printf("node_nid_to: %d, \n", nn_g->segments.data[ssi + 1]);

					printf("node_id_to: %s\n", nn_n_to->id);
					*/
					if (strcmp(nn_e->id_from, nn_n_from->id) == 0 && strcmp(nn_e->id_to, nn_n_to->id) == 0) {
						std::string edge_id_str(nn_e->id);

						struct nn_edge_operation *nn_e_op = &nn_e->operations.data[0];

						if (nn_e_op->type == NN_E_OP_T_CONV2D) {
							struct nn_edge_operation_conv2d *conv2d = (struct nn_edge_operation_conv2d *) nn_e_op->operation_params;
							unsigned int kernels_c = 0;

							unsigned int output_layer_width = nn_node_layer_width_get(nn_n_to);

							for (int l = 0; l < buffer_alloc_cpp_edges_params_ptrs_in.size(); l++) {
								buffer_alloc_cpp_edges_params_ptrs << util_replace_into(buffer_alloc_cpp_edges_params_ptrs_in[l], "{:edge_id:}", edge_id_str) << "\n";
							}
							for (int l = 0; l < buffer_alloc_hpp_edges_params_ptrs_in.size(); l++) {
								buffer_alloc_hpp_edges_params_ptrs << util_replace_into(buffer_alloc_hpp_edges_params_ptrs_in[l], "{:edge_id:}", edge_id_str) << "\n";
							}
							std::stringstream dev_ptr_name;
							dev_ptr_name << edge_id_str << "_dev_ptr";
							for (int l = 0; l < buffer_alloc_content_in.size(); l++) {
								std::string repl = util_replace_into(buffer_alloc_content_in[l], "{:dev_ptr:}", dev_ptr_name.str());
								repl = util_replace_into(repl, "{:size:}", util_string_from_uint(nn_e_op->operation_mem_req));

								buffer_alloc_content << repl << "\n";
							}
							std::stringstream params_filename_ss;
							params_filename_ss << "\"params/" << edge_id_str << ".params\"";
							for (int l = 0; l < buffer_fill_content_in.size(); l++) {
								std::string repl = util_replace_into(buffer_fill_content_in[l], "{:params_size:}", util_string_from_uint(nn_e_op->operation_mem_req));
								repl = util_replace_into(repl, "{:params_filename:}", params_filename_ss.str());
								repl = util_replace_into(repl, "{:params_dev_ptr:}", dev_ptr_name.str());

								buffer_fill_content << repl << "\n";
							}

							for (int d_from = 0; d_from < m_array_size(&nn_n_from->dimensions); d_from++) {
								const char *assoc_id_from = nn_n_from->dimensions.data[d_from].assoc_id_out;
								for (int d_to = 0; d_to < m_array_size(&nn_n_to->dimensions); d_to++) {
									const char *assoc_id_to = nn_n_to->dimensions.data[d_to].assoc_id_in;
									if (strcmp(assoc_id_from, assoc_id_to) == 0) {
										int dim_offset_in = 0;
										for (int d_from_ = 0; d_from_ < d_from; d_from_++) {
											dim_offset_in += nn_n_from->dimensions.data[d_from_].rows * nn_n_from->dimensions.data[d_from_].cols * nn_n_from->dimensions.data[d_from_].channels;
										}
										int dim_offset_out = 0;
										for (int d_to_ = 0; d_to_ < d_to; d_to_++) {
											dim_offset_out += nn_n_to->dimensions.data[d_to_].rows * nn_n_to->dimensions.data[d_to_].cols * nn_n_to->dimensions.data[d_to_].channels;
										}
										unsigned int output_nodes_count = nn_n_to->dimensions.data[d_to].rows * nn_n_to->dimensions.data[d_to].cols * nn_n_to->dimensions.data[d_to].channels;


										std::stringstream edge_id_ss;
										edge_id_ss << nn_e->id << "_" << kernels_c;
										kernels_c++;

										std::stringstream filter_content;

										unsigned int output_prefix_sum = 0;
										unsigned int kernel_prefix_sum = 0;

										unsigned int filters_c = m_array_size(&conv2d->filters);
										for (int f = 0; f < filters_c; f++) {
											struct nn_filter *nn_f = &conv2d->filters.data[f];

											unsigned int filter_output_rows = (nn_n_from->dimensions.data[d_from].rows - (nn_f->kernel_size[0]/nn_f->dilation[0]) + 1) / nn_f->stride[0];
											unsigned int filter_output_cols = (nn_n_from->dimensions.data[d_from].cols - (nn_f->kernel_size[1]/nn_f->dilation[1]) + 1) / nn_f->stride[1];
											unsigned int input_channels = nn_n_from->dimensions.data[d_from].channels;

											for (int l = 0; l < edges_cu_edges_functions_filter_in.size(); l++) {
												std::string repl = util_replace_into(edges_cu_edges_functions_filter_in[l], "{:filters_c:}", util_string_from_uint(nn_f->filters_c));
												repl = util_replace_into(repl, "{:input_rows:}", util_string_from_uint(nn_n_from->dimensions.data[d_from].rows));
												repl = util_replace_into(repl, "{:input_cols:}", util_string_from_uint(nn_n_from->dimensions.data[d_from].cols));
												repl = util_replace_into(repl, "{:input_channels:}", util_string_from_uint(nn_n_from->dimensions.data[d_from].channels));
												repl = util_replace_into(repl, "{:filter_output_rows:}", util_string_from_uint(filter_output_rows));
												repl = util_replace_into(repl, "{:filter_output_cols:}", util_string_from_uint(filter_output_cols));
												repl = util_replace_into(repl, "{:kernel_size_x:}", util_string_from_uint(nn_f->kernel_size[0]));
												repl = util_replace_into(repl, "{:kernel_size_y:}", util_string_from_uint(nn_f->kernel_size[1]));
												repl = util_replace_into(repl, "{:kernel_dilation_x:}", util_string_from_uint(nn_f->dilation[0]));
												repl = util_replace_into(repl, "{:kernel_dilation_y:}", util_string_from_uint(nn_f->dilation[1]));
												repl = util_replace_into(repl, "{:kernel_stride_x:}", util_string_from_uint(nn_f->stride[0]));
												repl = util_replace_into(repl, "{:kernel_stride_y:}", util_string_from_uint(nn_f->stride[1]));
												repl = util_replace_into(repl, "{:kernel_prefix_sum:}", util_string_from_uint(kernel_prefix_sum));
												repl = util_replace_into(repl, "{:output_prefix_sum:}", util_string_from_uint(output_prefix_sum));

												filter_content << repl << "\n";
											}

											output_prefix_sum += nn_f->filters_c * input_channels * filter_output_rows * filter_output_cols;
											kernel_prefix_sum += nn_f->filters_c * nn_f->kernel_size[0] + nn_f->kernel_size[1];
										}

										for (int l = 0; l < edges_cu_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_cu_edges_functions_in[l], "{:layer_output_width:}", util_string_from_uint(output_layer_width));
											repl = util_replace_into(repl, "{:batch_size:}", util_string_from_uint(batch_size));
											repl = util_replace_into(repl, "{:kernel_content:}", filter_content.str());
											repl = util_replace_into(repl, "{:launch_cuda_stream:}", cuda_stream_name.str());
											repl = util_replace_into(repl, "{:edge_id:}", edge_id_ss.str());
											if (sg > 0 && ssi == first_node_idx) {
												//assert(nn_n_from->last_used_buffer_str != nullptr);
												if (nn_n_from->last_used_buffer_str == nullptr) {
													printf("error: last used buffer == nullptr\n");
													repl = util_replace_into(repl, "{:dev_ptr_input:}", buffer_swap_ptrs[swap]);
												} else {
													std::stringstream input_buffer_offset_ptr;
													input_buffer_offset_ptr << nn_n_from->last_used_buffer_str << " + " << dim_offset_in;
													repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
												}
											} else {
												std::stringstream input_buffer_offset_ptr;
												input_buffer_offset_ptr << buffer_swap_ptrs[swap] << " + " << dim_offset_in;
												repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
											}

											if (nn_n_to->last_used_buffer_str == nullptr) {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << buffer_swap_ptrs[!swap] << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
												util_chararray_from_const(buffer_swap_ptrs[!swap].c_str(), &nn_n_to->last_used_buffer_str);
											} else {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << nn_n_to->last_used_buffer_str << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
											}

											std::stringstream dev_ptr_params_offset_ptr;
											dev_ptr_params_offset_ptr << dev_ptr_name.str();// << " + " << params_offset;
											repl = util_replace_into(repl, "{:dev_ptr_params:}", dev_ptr_params_offset_ptr.str());

											edges_cu_edges_functions << repl << "\n";
										}

										for (int l = 0; l < edges_hpp_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_hpp_edges_functions_in[l], "{:edge_id:}", edge_id_ss.str());
											edges_hpp_edges_functions << repl << "\n";
										}
									}
								}
							}

							edge_id_2_kernel_c.insert(std::pair<char *, unsigned int>(nn_e->id, kernels_c));

						} else if (nn_e_op->type == NN_E_OP_T_MAXPOOL2D) {
							struct nn_edge_operation_maxpool2d *maxpool2d = (struct nn_edge_operation_maxpool2d *) nn_e_op->operation_params;

							unsigned int kernels_c = 0;

							unsigned int output_layer_width = nn_node_layer_width_get(nn_n_to);

							for (int l = 0; l < buffer_alloc_cpp_edges_params_ptrs_in.size(); l++) {
								buffer_alloc_cpp_edges_params_ptrs << util_replace_into(buffer_alloc_cpp_edges_params_ptrs_in[l], "{:edge_id:}", edge_id_str) << "\n";
							}
							for (int l = 0; l < buffer_alloc_hpp_edges_params_ptrs_in.size(); l++) {
								buffer_alloc_hpp_edges_params_ptrs << util_replace_into(buffer_alloc_hpp_edges_params_ptrs_in[l], "{:edge_id:}", edge_id_str) << "\n";
							}
							std::stringstream dev_ptr_name;
							dev_ptr_name << edge_id_str << "_dev_ptr";
							for (int l = 0; l < buffer_alloc_content_in.size(); l++) {
								std::string repl = util_replace_into(buffer_alloc_content_in[l], "{:dev_ptr:}", dev_ptr_name.str());
								repl = util_replace_into(repl, "{:size:}", util_string_from_uint(nn_e_op->operation_mem_req));

								buffer_alloc_content << repl << "\n";
							}
							std::stringstream params_filename_ss;
							params_filename_ss << "\"params/" << edge_id_str << ".params\"";
							for (int l = 0; l < buffer_fill_content_in.size(); l++) {
								std::string repl = util_replace_into(buffer_fill_content_in[l], "{:params_size:}", util_string_from_uint(nn_e_op->operation_mem_req));
								repl = util_replace_into(repl, "{:params_filename:}", params_filename_ss.str());
								repl = util_replace_into(repl, "{:params_dev_ptr:}", dev_ptr_name.str());

								buffer_fill_content << repl << "\n";
							}

							for (int d_from = 0; d_from < m_array_size(&nn_n_from->dimensions); d_from++) {
								const char *assoc_id_from = nn_n_from->dimensions.data[d_from].assoc_id_out;
								for (int d_to = 0; d_to < m_array_size(&nn_n_to->dimensions); d_to++) {
									const char *assoc_id_to = nn_n_to->dimensions.data[d_to].assoc_id_in;
									if (strcmp(assoc_id_from, assoc_id_to) == 0) {
										int dim_offset_in = 0;
										for (int d_from_ = 0; d_from_ < d_from; d_from_++) {
											dim_offset_in += nn_n_from->dimensions.data[d_from_].rows * nn_n_from->dimensions.data[d_from_].cols * nn_n_from->dimensions.data[d_from_].channels;
										}
										int dim_offset_out = 0;
										for (int d_to_ = 0; d_to_ < d_to; d_to_++) {
											dim_offset_out += nn_n_to->dimensions.data[d_to_].rows * nn_n_to->dimensions.data[d_to_].cols * nn_n_to->dimensions.data[d_to_].channels;
										}
										unsigned int output_nodes_count = nn_n_to->dimensions.data[d_to].rows * nn_n_to->dimensions.data[d_to].cols * nn_n_to->dimensions.data[d_to].channels;


										std::stringstream edge_id_ss;
										edge_id_ss << nn_e->id << "_" << kernels_c;
										kernels_c++;

										std::stringstream filter_content;

										unsigned int output_prefix_sum = 0;
										unsigned int kernel_prefix_sum = 0;

										unsigned int filters_c = m_array_size(&maxpool2d->filters);
										for (int f = 0; f < filters_c; f++) {
											struct nn_filter *nn_f = &maxpool2d->filters.data[f];

											unsigned int filter_output_rows = (nn_n_from->dimensions.data[d_from].rows - (nn_f->kernel_size[0]/nn_f->dilation[0]) + 1) / nn_f->stride[0];
											unsigned int filter_output_cols = (nn_n_from->dimensions.data[d_from].cols - (nn_f->kernel_size[1]/nn_f->dilation[1]) + 1) / nn_f->stride[1];
											unsigned int input_channels = nn_n_from->dimensions.data[d_from].channels;

											for (int l = 0; l < edges_cu_edges_functions_maxpool2d_in.size(); l++) {
												std::string repl = util_replace_into(edges_cu_edges_functions_maxpool2d_in[l], "{:filters_c:}", util_string_from_uint(nn_f->filters_c));
												repl = util_replace_into(repl, "{:input_rows:}", util_string_from_uint(nn_n_from->dimensions.data[d_from].rows));
												repl = util_replace_into(repl, "{:input_cols:}", util_string_from_uint(nn_n_from->dimensions.data[d_from].cols));
												repl = util_replace_into(repl, "{:input_channels:}", util_string_from_uint(nn_n_from->dimensions.data[d_from].channels));
												repl = util_replace_into(repl, "{:filter_output_rows:}", util_string_from_uint(filter_output_rows));
												repl = util_replace_into(repl, "{:filter_output_cols:}", util_string_from_uint(filter_output_cols));
												repl = util_replace_into(repl, "{:kernel_size_x:}", util_string_from_uint(nn_f->kernel_size[0]));
												repl = util_replace_into(repl, "{:kernel_size_y:}", util_string_from_uint(nn_f->kernel_size[1]));
												repl = util_replace_into(repl, "{:kernel_dilation_x:}", util_string_from_uint(nn_f->dilation[0]));
												repl = util_replace_into(repl, "{:kernel_dilation_y:}", util_string_from_uint(nn_f->dilation[1]));
												repl = util_replace_into(repl, "{:kernel_stride_x:}", util_string_from_uint(nn_f->stride[0]));
												repl = util_replace_into(repl, "{:kernel_stride_y:}", util_string_from_uint(nn_f->stride[1]));
												repl = util_replace_into(repl, "{:kernel_prefix_sum:}", util_string_from_uint(kernel_prefix_sum));
												repl = util_replace_into(repl, "{:output_prefix_sum:}", util_string_from_uint(output_prefix_sum));

												filter_content << repl << "\n";
											}

											output_prefix_sum += nn_f->filters_c * input_channels * filter_output_rows * filter_output_cols;
											kernel_prefix_sum += nn_f->filters_c * nn_f->kernel_size[0] + nn_f->kernel_size[1];
										}

										for (int l = 0; l < edges_cu_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_cu_edges_functions_in[l], "{:layer_output_width:}", util_string_from_uint(output_layer_width));
											repl = util_replace_into(repl, "{:batch_size:}", util_string_from_uint(batch_size));
											repl = util_replace_into(repl, "{:kernel_content:}", filter_content.str());
											repl = util_replace_into(repl, "{:launch_cuda_stream:}", cuda_stream_name.str());
											repl = util_replace_into(repl, "{:edge_id:}", edge_id_ss.str());
											if (sg > 0 && ssi == first_node_idx) {
												//assert(nn_n_from->last_used_buffer_str != nullptr);
												if (nn_n_from->last_used_buffer_str == nullptr) {
													printf("error: last used buffer == nullptr\n");
													repl = util_replace_into(repl, "{:dev_ptr_input:}", buffer_swap_ptrs[swap]);
												} else {
													std::stringstream input_buffer_offset_ptr;
													input_buffer_offset_ptr << nn_n_from->last_used_buffer_str << " + " << dim_offset_in;
													repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
												}
											} else {
												std::stringstream input_buffer_offset_ptr;
												input_buffer_offset_ptr << buffer_swap_ptrs[swap] << " + " << dim_offset_in;
												repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
											}

											if (nn_n_to->last_used_buffer_str == nullptr) {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << buffer_swap_ptrs[!swap] << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
												util_chararray_from_const(buffer_swap_ptrs[!swap].c_str(), &nn_n_to->last_used_buffer_str);
											} else {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << nn_n_to->last_used_buffer_str << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
											}

											std::stringstream dev_ptr_params_offset_ptr;
											dev_ptr_params_offset_ptr << dev_ptr_name.str();// << " + " << params_offset;
											repl = util_replace_into(repl, "{:dev_ptr_params:}", dev_ptr_params_offset_ptr.str());

											edges_cu_edges_functions << repl << "\n";
										}

										for (int l = 0; l < edges_hpp_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_hpp_edges_functions_in[l], "{:edge_id:}", edge_id_ss.str());
											edges_hpp_edges_functions << repl << "\n";
										}
									}
								}
							}

							edge_id_2_kernel_c.insert(std::pair<char *, unsigned int>(nn_e->id, kernels_c));
						} else if (nn_e_op->type == NN_E_OP_T_CONCAT) {
							unsigned int kernels_c = 0;

							unsigned int output_layer_width = nn_node_layer_width_get(nn_n_to);

							for (int d_from = 0; d_from < m_array_size(&nn_n_from->dimensions); d_from++) {
								const char *assoc_id_from = nn_n_from->dimensions.data[d_from].assoc_id_out;
								for (int d_to = 0; d_to < m_array_size(&nn_n_to->dimensions); d_to++) {
									const char *assoc_id_to = nn_n_to->dimensions.data[d_to].assoc_id_in;
									if (strcmp(assoc_id_from, assoc_id_to) == 0) {
										int dim_offset_in = 0;
										for (int d_from_ = 0; d_from_ < d_from; d_from_++) {
											dim_offset_in += nn_n_from->dimensions.data[d_from_].rows * nn_n_from->dimensions.data[d_from_].cols * nn_n_from->dimensions.data[d_from_].channels;
										}
										int dim_offset_out = 0;
										for (int d_to_ = 0; d_to_ < d_to; d_to_++) {
											dim_offset_out += nn_n_to->dimensions.data[d_to_].rows * nn_n_to->dimensions.data[d_to_].cols * nn_n_to->dimensions.data[d_to_].channels;
										}
										unsigned int output_nodes_count = nn_n_to->dimensions.data[d_to].rows * nn_n_to->dimensions.data[d_to].cols * nn_n_to->dimensions.data[d_to].channels;

										std::stringstream kernel_content;
										for (int l = 0; l < edges_cu_edges_functions_concat_in.size(); l++) {
											std::string repl = util_replace_into(edges_cu_edges_functions_concat_in[l], "{:output_nodes_count:}", util_string_from_uint(output_layer_width));
											kernel_content << repl << "\n";
										}

										std::stringstream edge_id_ss;
										edge_id_ss << nn_e->id << "_" << kernels_c;
										kernels_c++;

										for (int l = 0; l < edges_cu_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_cu_edges_functions_in[l], "{:layer_output_width:}", util_string_from_uint(output_layer_width));
											repl = util_replace_into(repl, "{:batch_size:}", util_string_from_uint(batch_size));
											repl = util_replace_into(repl, "{:kernel_content:}", kernel_content.str());
											repl = util_replace_into(repl, "{:launch_cuda_stream:}", cuda_stream_name.str()); //d
											repl = util_replace_into(repl, "{:edge_id:}", edge_id_ss.str());
											if (sg > 0 && ssi == first_node_idx) {
												//assert(nn_n_from->last_used_buffer_str != nullptr);
												if (nn_n_from->last_used_buffer_str == nullptr) {
													printf("error: last used buffer == nullptr\n");
													repl = util_replace_into(repl, "{:dev_ptr_input:}", buffer_swap_ptrs[swap]);
												} else {
													std::stringstream input_buffer_offset_ptr;
													input_buffer_offset_ptr << nn_n_from->last_used_buffer_str << " + " << dim_offset_in;
													repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
												}
											} else {
												std::stringstream input_buffer_offset_ptr;
												input_buffer_offset_ptr << buffer_swap_ptrs[swap] << " + " << dim_offset_in;
												repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
											}

											if (nn_n_to->last_used_buffer_str == nullptr) {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << buffer_swap_ptrs[!swap] << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
												util_chararray_from_const(buffer_swap_ptrs[!swap].c_str(), &nn_n_to->last_used_buffer_str);
											} else {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << nn_n_to->last_used_buffer_str << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
											}

											std::string nullptr_str("nullptr");
											repl = util_replace_into(repl, "{:dev_ptr_params:}", nullptr_str);
											edges_cu_edges_functions << repl << "\n";
										}

										for (int l = 0; l < edges_hpp_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_hpp_edges_functions_in[l], "{:edge_id:}", edge_id_ss.str());
											edges_hpp_edges_functions << repl << "\n";
										}
									}
								}
							}

							edge_id_2_kernel_c.insert(std::pair<char *, unsigned int>(nn_e->id, kernels_c));

						} else if (nn_e_op->type == NN_E_OP_T_RELU) {
							unsigned int kernels_c = 0;

							unsigned int output_layer_width = nn_node_layer_width_get(nn_n_to);

							for (int d_from = 0; d_from < m_array_size(&nn_n_from->dimensions); d_from++) {
								const char *assoc_id_from = nn_n_from->dimensions.data[d_from].assoc_id_out;
								for (int d_to = 0; d_to < m_array_size(&nn_n_to->dimensions); d_to++) {
									const char *assoc_id_to = nn_n_to->dimensions.data[d_to].assoc_id_in;
									if (strcmp(assoc_id_from, assoc_id_to) == 0) {

										int dim_offset_in = 0;
										for (int d_from_ = 0; d_from_ < d_from; d_from_++) {
											dim_offset_in += nn_n_from->dimensions.data[d_from_].rows * nn_n_from->dimensions.data[d_from_].cols * nn_n_from->dimensions.data[d_from_].channels;
										}
										int dim_offset_out = 0;
										for (int d_to_ = 0; d_to_ < d_to; d_to_++) {
											dim_offset_out += nn_n_to->dimensions.data[d_to_].rows * nn_n_to->dimensions.data[d_to_].cols * nn_n_to->dimensions.data[d_to_].channels;
										}
										unsigned int output_nodes_count = nn_n_to->dimensions.data[d_to].rows * nn_n_to->dimensions.data[d_to].cols * nn_n_to->dimensions.data[d_to].channels;

										std::stringstream kernel_content;
										for (int l = 0; l < edges_cu_edges_functions_relu_in.size(); l++) {
											std::string repl = util_replace_into(edges_cu_edges_functions_relu_in[l], "{:output_nodes_count:}", util_string_from_uint(output_layer_width));
											kernel_content << repl << "\n";
										}

										std::stringstream edge_id_ss;
										edge_id_ss << nn_e->id << "_" << kernels_c;
										kernels_c++;

										for (int l = 0; l < edges_cu_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_cu_edges_functions_in[l], "{:layer_output_width:}", util_string_from_uint(output_layer_width));
											repl = util_replace_into(repl, "{:batch_size:}", util_string_from_uint(batch_size));
											repl = util_replace_into(repl, "{:kernel_content:}", kernel_content.str());
											repl = util_replace_into(repl, "{:launch_cuda_stream:}", cuda_stream_name.str()); //d
											repl = util_replace_into(repl, "{:edge_id:}", edge_id_ss.str());
											if (sg > 0 && ssi == first_node_idx) {
												//assert(nn_n_from->last_used_buffer_str != nullptr);
												if (nn_n_from->last_used_buffer_str == nullptr) {
													printf("error: last used buffer == nullptr\n");
													repl = util_replace_into(repl, "{:dev_ptr_input:}", buffer_swap_ptrs[swap]);
												} else {
													std::stringstream input_buffer_offset_ptr;
													input_buffer_offset_ptr << nn_n_from->last_used_buffer_str << " + " << dim_offset_in;
													repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
												}
											} else {
												std::stringstream input_buffer_offset_ptr;
												input_buffer_offset_ptr << buffer_swap_ptrs[swap] << " + " << dim_offset_in;
												repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
											}

											if (nn_n_to->last_used_buffer_str == nullptr) {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << buffer_swap_ptrs[!swap] << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
												util_chararray_from_const(buffer_swap_ptrs[!swap].c_str(), &nn_n_to->last_used_buffer_str);
											} else {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << nn_n_to->last_used_buffer_str << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
											}

											std::string nullptr_str("nullptr");
											repl = util_replace_into(repl, "{:dev_ptr_params:}", nullptr_str);
											edges_cu_edges_functions << repl << "\n";
										}

										for (int l = 0; l < edges_hpp_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_hpp_edges_functions_in[l], "{:edge_id:}", edge_id_ss.str());
											edges_hpp_edges_functions << repl << "\n";
										}
									}
								}
							}

							edge_id_2_kernel_c.insert(std::pair<char *, unsigned int>(nn_e->id, kernels_c));

						} else if (nn_e_op->type == NN_E_OP_T_SOFTMAX) {
							unsigned int kernels_c = 0;

							unsigned int output_layer_width = nn_node_layer_width_get(nn_n_to);

							for (int d_from = 0; d_from < m_array_size(&nn_n_from->dimensions); d_from++) {
								const char *assoc_id_from = nn_n_from->dimensions.data[d_from].assoc_id_out;
								for (int d_to = 0; d_to < m_array_size(&nn_n_to->dimensions); d_to++) {
									const char *assoc_id_to = nn_n_to->dimensions.data[d_to].assoc_id_in;
									if (strcmp(assoc_id_from, assoc_id_to) == 0) {

										int dim_offset_in = 0;
										for (int d_from_ = 0; d_from_ < d_from; d_from_++) {
											dim_offset_in += nn_n_from->dimensions.data[d_from_].rows * nn_n_from->dimensions.data[d_from_].cols * nn_n_from->dimensions.data[d_from_].channels;
										}
										int dim_offset_out = 0;
										for (int d_to_ = 0; d_to_ < d_to; d_to_++) {
											dim_offset_out += nn_n_to->dimensions.data[d_to_].rows * nn_n_to->dimensions.data[d_to_].cols * nn_n_to->dimensions.data[d_to_].channels;
										}
										unsigned int output_nodes_count = nn_n_to->dimensions.data[d_to].rows * nn_n_to->dimensions.data[d_to].cols * nn_n_to->dimensions.data[d_to].channels;

										std::stringstream kernel_content;
										for (int l = 0; l < edges_cu_edges_functions_softmax_in.size(); l++) {
											std::string repl = util_replace_into(edges_cu_edges_functions_softmax_in[l], "{:output_nodes_count:}", util_string_from_uint(output_layer_width));
											kernel_content << repl << "\n";
										}

										std::stringstream edge_id_ss;
										edge_id_ss << nn_e->id << "_" << kernels_c;
										kernels_c++;

										for (int l = 0; l < edges_cu_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_cu_edges_functions_in[l], "{:layer_output_width:}", util_string_from_uint(output_layer_width));
											repl = util_replace_into(repl, "{:batch_size:}", util_string_from_uint(batch_size));
											repl = util_replace_into(repl, "{:kernel_content:}", kernel_content.str());
											repl = util_replace_into(repl, "{:launch_cuda_stream:}", cuda_stream_name.str()); //d
											repl = util_replace_into(repl, "{:edge_id:}", edge_id_ss.str());
											if (sg > 0 && ssi == first_node_idx) {
												//assert(nn_n_from->last_used_buffer_str != nullptr);
												if (nn_n_from->last_used_buffer_str == nullptr) {
													printf("error: last used buffer == nullptr\n");
													repl = util_replace_into(repl, "{:dev_ptr_input:}", buffer_swap_ptrs[swap]);
												} else {
													std::stringstream input_buffer_offset_ptr;
													input_buffer_offset_ptr << nn_n_from->last_used_buffer_str << " + " << dim_offset_in;
													repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
												}
											} else {
												std::stringstream input_buffer_offset_ptr;
												input_buffer_offset_ptr << buffer_swap_ptrs[swap] << " + " << dim_offset_in;
												repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
											}

											if (nn_n_to->last_used_buffer_str == nullptr) {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << buffer_swap_ptrs[!swap] << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
												util_chararray_from_const(buffer_swap_ptrs[!swap].c_str(), &nn_n_to->last_used_buffer_str);
											} else {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << nn_n_to->last_used_buffer_str << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
											}

											std::string nullptr_str("nullptr");
											repl = util_replace_into(repl, "{:dev_ptr_params:}", nullptr_str);
											edges_cu_edges_functions << repl << "\n";
										}

										for (int l = 0; l < edges_hpp_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_hpp_edges_functions_in[l], "{:edge_id:}", edge_id_ss.str());
											edges_hpp_edges_functions << repl << "\n";
										}
									}
								}
							}

							edge_id_2_kernel_c.insert(std::pair<char *, unsigned int>(nn_e->id, kernels_c));
						} else if (nn_e_op->type == NN_E_OP_T_FULLYCONNECTED) {
							struct nn_edge_operation_fullyconnected *fullyconnected = (struct nn_edge_operation_fullyconnected *) nn_e_op->operation_params;

							unsigned int kernels_c = 0;

							unsigned int output_layer_width = nn_node_layer_width_get(nn_n_to);


							unsigned int input_nodes_count = 0;
							unsigned int input_dims_c = m_array_size(&nn_n_from->dimensions);
							for (int d = 0; d < input_dims_c; d++) {
								input_nodes_count += nn_n_from->dimensions.data[d].rows * nn_n_from->dimensions.data[d].cols * nn_n_from->dimensions.data[d].channels;
							}

							for (int l = 0; l < buffer_alloc_cpp_edges_params_ptrs_in.size(); l++) {
								buffer_alloc_cpp_edges_params_ptrs << util_replace_into(buffer_alloc_cpp_edges_params_ptrs_in[l], "{:edge_id:}", edge_id_str) << "\n";
							}
							for (int l = 0; l < buffer_alloc_hpp_edges_params_ptrs_in.size(); l++) {
								buffer_alloc_hpp_edges_params_ptrs << util_replace_into(buffer_alloc_hpp_edges_params_ptrs_in[l], "{:edge_id:}", edge_id_str) << "\n";
							}
							std::stringstream dev_ptr_name;
							dev_ptr_name << edge_id_str << "_dev_ptr";
							for (int l = 0; l < buffer_alloc_content_in.size(); l++) {
								std::string repl = util_replace_into(buffer_alloc_content_in[l], "{:dev_ptr:}", dev_ptr_name.str());
								repl = util_replace_into(repl, "{:size:}", util_string_from_uint(output_layer_width));

								buffer_alloc_content << repl << "\n";
							}
							std::stringstream params_filename_ss;
							params_filename_ss << "\"params/" << edge_id_str << ".params\"";
							for (int l = 0; l < buffer_fill_content_in.size(); l++) {
								std::string repl = util_replace_into(buffer_fill_content_in[l], "{:params_size:}", util_string_from_uint(output_layer_width));
								repl = util_replace_into(repl, "{:params_filename:}", params_filename_ss.str());
								repl = util_replace_into(repl, "{:params_dev_ptr:}", dev_ptr_name.str());

								buffer_fill_content << repl << "\n";
							}


							for (int d_from = 0; d_from < m_array_size(&nn_n_from->dimensions); d_from++) {
								const char *assoc_id_from = nn_n_from->dimensions.data[d_from].assoc_id_out;
								for (int d_to = 0; d_to < m_array_size(&nn_n_to->dimensions); d_to++) {
									const char *assoc_id_to = nn_n_to->dimensions.data[d_to].assoc_id_in;
									if (strcmp(assoc_id_from, assoc_id_to) == 0) {

										int dim_offset_in = 0;
										for (int d_from_ = 0; d_from_ < d_from; d_from_++) {
											dim_offset_in += nn_n_from->dimensions.data[d_from_].rows * nn_n_from->dimensions.data[d_from_].cols * nn_n_from->dimensions.data[d_from_].channels;
										}
										int dim_offset_out = 0;
										for (int d_to_ = 0; d_to_ < d_to; d_to_++) {
											dim_offset_out += nn_n_to->dimensions.data[d_to_].rows * nn_n_to->dimensions.data[d_to_].cols * nn_n_to->dimensions.data[d_to_].channels;
										}
																																					   //biases
																		  //all out_dims are same (rows, 1, 1) -> param_offset = rows * (dim_offset_in + d_from)
										unsigned int output_nodes_count = nn_n_to->dimensions.data[d_to].rows * nn_n_to->dimensions.data[d_to].cols * nn_n_to->dimensions.data[d_to].channels;
										unsigned int params_offset = output_nodes_count * (dim_offset_in + d_from);

										std::stringstream kernel_content;
										for (int l = 0; l < edges_cu_edges_functions_fullyconnected_in.size(); l++) {
											std::string repl = util_replace_into(edges_cu_edges_functions_fullyconnected_in[l], "{:output_nodes_count:}", util_string_from_uint(output_nodes_count));
											repl = util_replace_into(repl, "{:input_nodes_count:}", util_string_from_uint(input_nodes_count));
											kernel_content << repl << "\n";
										}

										std::stringstream edge_id_ss;
										edge_id_ss << nn_e->id << "_" << kernels_c;
										kernels_c++;

										for (int l = 0; l < edges_cu_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_cu_edges_functions_in[l], "{:layer_output_width:}", util_string_from_uint(output_layer_width));
											repl = util_replace_into(repl, "{:batch_size:}", util_string_from_uint(batch_size));
											repl = util_replace_into(repl, "{:kernel_content:}", kernel_content.str());
											repl = util_replace_into(repl, "{:launch_cuda_stream:}", cuda_stream_name.str());
											repl = util_replace_into(repl, "{:edge_id:}", edge_id_ss.str());
											if (sg > 0 && ssi == first_node_idx) {
												//assert(nn_n_from->last_used_buffer_str != nullptr);
												if (nn_n_from->last_used_buffer_str == nullptr) {
													printf("error: last used buffer == nullptr\n");
													repl = util_replace_into(repl, "{:dev_ptr_input:}", buffer_swap_ptrs[swap]);
												} else {
													std::stringstream input_buffer_offset_ptr;
													input_buffer_offset_ptr << nn_n_from->last_used_buffer_str << " + " << dim_offset_in;
													repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
												}
											} else {
												std::stringstream input_buffer_offset_ptr;
												input_buffer_offset_ptr << buffer_swap_ptrs[swap] << " + " << dim_offset_in;
												repl = util_replace_into(repl, "{:dev_ptr_input:}", input_buffer_offset_ptr.str());
											}

											if (nn_n_to->last_used_buffer_str == nullptr) {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << buffer_swap_ptrs[!swap] << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
												util_chararray_from_const(buffer_swap_ptrs[!swap].c_str(), &nn_n_to->last_used_buffer_str);
											} else {
												std::stringstream output_buffer_offset_ptr;
												output_buffer_offset_ptr << nn_n_to->last_used_buffer_str << " + " << dim_offset_out;
												repl = util_replace_into(repl, "{:dev_ptr_output:}", output_buffer_offset_ptr.str());
											}

											std::stringstream dev_ptr_params_offset_ptr;
											dev_ptr_params_offset_ptr << dev_ptr_name.str() << " + " << params_offset;
											repl = util_replace_into(repl, "{:dev_ptr_params:}", dev_ptr_params_offset_ptr.str());

											edges_cu_edges_functions << repl << "\n";
										}

										for (int l = 0; l < edges_hpp_edges_functions_in.size(); l++) {
											std::string repl = util_replace_into(edges_hpp_edges_functions_in[l], "{:edge_id:}", edge_id_ss.str());
											edges_hpp_edges_functions << repl << "\n";
										}
									}
								}
							}

							edge_id_2_kernel_c.insert(std::pair<char *, unsigned int>(nn_e->id, kernels_c));
						}
					}
				}
				swap = !swap;
			}
		}
	}

	std::vector<std::string> edges_cu_in = util_file_read("src.in/edges.cu.in");
	for (int l = 0; l < edges_cu_in.size(); l++) {
		edges_cu_in[l] = util_replace_into(edges_cu_in[l], "{:edges_functions:}", edges_cu_edges_functions.str());
	}
	std::stringstream edges_cu_out;
	edges_cu_out << "src.out/edges.cu";
	util_file_write(edges_cu_out.str().c_str(), edges_cu_in);

	std::vector<std::string> edges_hpp_in = util_file_read("src.in/edges.hpp.in");
	for (int l = 0; l < edges_hpp_in.size(); l++) {
		edges_hpp_in[l] = util_replace_into(edges_hpp_in[l], "{:edges_functions:}", edges_hpp_edges_functions.str());
	}
	std::stringstream edges_hpp_out;
	edges_hpp_out << "src.out/edges.hpp";
	util_file_write(edges_hpp_out.str().c_str(), edges_hpp_in);


	//allocate buffers
	std::vector<std::string> buffer_alloc_hpp = util_file_read("src.in/buffer_alloc.hpp.in");
	std::vector<std::string> buffer_alloc_cpp = util_file_read("src.in/buffer_alloc.cpp.in");

	std::stringstream dev_ptrs_hpp_content;
	std::stringstream dev_ptrs_cpp_content;


		for (int ms = 0; ms < max_elements_per_group; ms++) {
			for (int i = 0; i < 2; i++) {
				std::stringstream ss_dev_ptr;
				ss_dev_ptr << "dev_ptr_" << ms << "_" << i;

				std::vector<std::string> dev_ptrs_hpp = util_file_read("src.in/buffer_alloc.hpp.dev_ptrs.in");
				for (int l = 0; l < dev_ptrs_hpp.size(); l++) {
					dev_ptrs_hpp_content << util_replace_into(dev_ptrs_hpp[l], "{:dev_ptr:}", ss_dev_ptr.str()) << "\n";
				}
				std::vector<std::string> dev_ptrs_cpp = util_file_read("src.in/buffer_alloc.cpp.dev_ptrs.in");
				for (int l = 0; l < dev_ptrs_cpp.size(); l++) {
					dev_ptrs_cpp_content << util_replace_into(dev_ptrs_cpp[l], "{:dev_ptr:}", ss_dev_ptr.str())<< "\n";
				}

				std::vector<std::string> buffer_alloc = util_file_read("src.in/buffer_alloc.cpp.buffer_alloc.in");
				for (int l = 0; l < buffer_alloc.size(); l++) {
					std::stringstream ss_value;
					ss_value << max_buffer_sizes.data[2 * ms + i];

					std::string replaced = util_replace_into(buffer_alloc[l], "{:dev_ptr:}", ss_dev_ptr.str());
					replaced = util_replace_into(replaced, "{:size:}", ss_value.str());
					buffer_alloc_content << replaced << "\n";
				}
			}
		}

		//printf("-- buffer_alloc.hpp --\n");
		for (int l = 0; l < buffer_alloc_hpp.size(); l++) {
			buffer_alloc_hpp[l] = util_replace_into(buffer_alloc_hpp[l], "{:dev_ptrs:}", dev_ptrs_hpp_content.str());
			buffer_alloc_hpp[l] = util_replace_into(buffer_alloc_hpp[l], "{:edges_params_ptrs:}", buffer_alloc_hpp_edges_params_ptrs.str());
			//printf("%s\n", buffer_alloc_hpp[l].c_str());
		}
		std::stringstream buffer_alloc_hpp_out;
		buffer_alloc_hpp_out << "src.out/buffer_alloc.hpp";
		util_file_write(buffer_alloc_hpp_out.str().c_str(), buffer_alloc_hpp);

		//printf("-- buffer_alloc.cpp --\n");
		for (int l = 0; l < buffer_alloc_cpp.size(); l++) {
			buffer_alloc_cpp[l] = util_replace_into(buffer_alloc_cpp[l], "{:dev_ptrs:}", dev_ptrs_cpp_content.str());
			buffer_alloc_cpp[l] = util_replace_into(buffer_alloc_cpp[l], "{:buffer_alloc_content:}", buffer_alloc_content.str());
			buffer_alloc_cpp[l] = util_replace_into(buffer_alloc_cpp[l], "{:buffer_fill_content:}", buffer_fill_content.str());
			buffer_alloc_cpp[l] = util_replace_into(buffer_alloc_cpp[l], "{:edges_params_ptrs:}", buffer_alloc_cpp_edges_params_ptrs.str());
			//printf("%s\n", buffer_alloc_cpp[l].c_str());
		}
		//printf("\n");
		std::stringstream buffer_alloc_cpp_out;
		buffer_alloc_cpp_out << "src.out/buffer_alloc.cpp";
		util_file_write(buffer_alloc_cpp_out.str().c_str(), buffer_alloc_cpp);


	sg_pos = 1;
	int max_group_elements = 0;
	for (int sg = 0; sg < segment_groups_c; sg++) {
		int segment_group_elements_c = segments_groups.data[sg_pos];	sg_pos++;
		if (segment_group_elements_c > max_group_elements) max_group_elements = segment_group_elements_c;
		int *segment_start_idx = (int *) malloc(segment_group_elements_c * sizeof(int));
		int *segment_ssi_cur_offset = (int *) malloc(segment_group_elements_c * sizeof(int));
		bool *ssi_term = (bool *) malloc(segment_group_elements_c * sizeof(bool));
		for (int seg = 0; seg < segment_group_elements_c; seg++) {
			segment_start_idx[seg] = segments_groups.data[sg_pos]; 		sg_pos++;
			segment_ssi_cur_offset[seg] = 0;
			ssi_term[seg] = false;
		}
		bool ssi_not_all_term = true;
		while (ssi_not_all_term) {
			//std::cout << "not all term " << sg << "\n";
			ssi_not_all_term = false;
			for (int seg = 0; seg < segment_group_elements_c; seg++) {
				if (ssi_term[seg]) continue;
				int first_node_idx = segments_starts.data[segment_start_idx[seg]] + 1 + segment_ssi_cur_offset[seg];
				int ssi = first_node_idx;

				struct nn_node *nn_n_from 	= &nn_g->nodes.data[nn_g->segments.data[ssi    ]];
				struct nn_node *nn_n_to 	= &nn_g->nodes.data[nn_g->segments.data[ssi + 1]];
				for (int e = 0; e < edges_c; e++) {
					struct nn_edge *nn_e = &nn_g->edges.data[e];
					if (strcmp(nn_e->id_from, nn_n_from->id) == 0 && strcmp(nn_e->id_to, nn_n_to->id) == 0) {
						std::string edge_id_str(nn_e->id);
						std::map<char *, unsigned int>::iterator kernels_c_it = edge_id_2_kernel_c.find(nn_e->id);
						if (kernels_c_it != edge_id_2_kernel_c.end()) {
							int edges_kernel_c = kernels_c_it->second;
							for (int kc = 0; kc < edges_kernel_c; kc++) {
								net_execution << "\t\t\tlaunch_" << edge_id_str << "_" << kc <<"_kernel();" << "//" << sg << ", " << seg << ", " << ssi << "\n";
								std::cout << "\t\t\tlaunch_" << edge_id_str << "_" << kc <<"_kernel();" << "//" << sg << ", " << seg << ", " << ssi << "\n";
							}
							//net_execution << "\t\t\tlaunch_" << edge_id_str << "_kernel();" << "//" << sg << ", " << seg << ", " << ssi << "\n";

						} else {
							printf("error: kernel_c not found\n");
						}

						//tmp global sync
						if ((sg > 0 && segment_ssi_cur_offset[seg] == 0) || nn_n_to->type == NN_N_T_OUTPUT) {
							net_execution << "\t\t\tcudaDeviceSynchronize();" << "\n";
						}
						break;
					}
				}
				segment_ssi_cur_offset[seg]++;
				if (ssi + 2 < segments_c && nn_g->segments.data[ssi+2] >= 0) {
					ssi_not_all_term = true;
				} else {
					ssi_term[seg] = true;
				}
			}
		}
		free(segment_start_idx);
		free(segment_ssi_cur_offset);
		free(ssi_term);
	}

	for (int l = 0; l < main_cpp_in.size(); l++) {
		main_cpp_in[l] = util_replace_into(main_cpp_in[l], "{:net_execution:}", net_execution.str());
		main_cpp_in[l] = util_replace_into(main_cpp_in[l], "{:cuda_stream_count:}", util_string_from_uint(max_group_elements));
	}
	std::stringstream main_cpp_out;
	main_cpp_out << "src.out/main.cpp";
	util_file_write(main_cpp_out.str().c_str(), main_cpp_in);

	sg_pos = 1;
	printf("segment_groups_c %d\n", segment_groups_c);
	for (int sg = 0; sg < segment_groups_c; sg++) {
		int segment_group_elements_c = segments_groups.data[sg_pos];	sg_pos++;
		printf("\tsegment_group %d, c %d\n", sg, segment_group_elements_c);
		for (int seg = 0; seg < segment_group_elements_c; seg++) {
			int segment_start_idx = segments_groups.data[sg_pos];		sg_pos++;
			printf("\t\telement %d", segment_start_idx);
			int value0 = max_buffer_sizes.data[2 * seg + 0];
			int value1 = max_buffer_sizes.data[2 * seg + 1];
			printf(" req0 req1 %d %d ", value0, value1);
		}
		printf("\n");
	}

	/*

	int *schedule = (int *) malloc(segments_starts_c * nn_s->graph_branches_parallel_max * sizeof(int));
	memset(schedule, 0, segments_starts_c * nn_s->graph_branches_parallel_max * sizeof(int));

	int nodes_c = m_array_size(&nn_g->nodes);
	if (nn_s->target == NN_S_T_INFERENCE) {
		bool buffer_swapping = false;
		int max_buffer_size = 0;
		int max_batch_size 	= 0;

		for (int n = 0; n < nodes_c; n++) {
			struct nn_node *nn_n = &nn_g->nodes.data[n];
			if (nn_n->output_mem_req > max_buffer_size) max_buffer_size = nn_n->output_mem_req;
		}

		max_batch_size = nn_s->total_memory_device_max_buffer/nn_g->output_mem_req;
		if (max_batch_size < nn_s->input_batch_min) {
			buffer_swapping = true;
			max_batch_size = nn_s->total_memory_device_max_buffer/(2 * max_buffer_size);
			if (max_batch_size < nn_s->input_batch_min) {
				printf("error (nn_scheduler): not enough device memory for input_batch_min %d. max_batch: %d", nn_s->input_batch_min, (nn_s->total_memory_device_max_buffer/(2 * max_buffer_size)));
				return;
			}
		} else {

		}
		if (max_batch_size > nn_s->input_batch_max) max_batch_size = nn_s->input_batch_max;

		bool ops_streaming = false;
		if (nn_s->total_memory_device_max_ops < nn_g->ops_mem_req) {
			ops_streaming = true;
		}

		if (!ops_streaming && !buffer_swapping) {

		}
	}

	*/


}
