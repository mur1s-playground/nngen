/*
 * nn_scheduler.hpp
 *
 *  Created on: Jun 27, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_SCHEDULER_HPP_
#define SRC_NN_SCHEDULER_HPP_

#include "nn_graph.hpp"

enum nn_scheduler_target {
	NN_S_T_INFERENCE,
	NN_S_T_TRAINING
};

enum nn_execution_mode {
	NN_E_M_BUFFER_SWAP,
	NN_E_M_TRAIN
};

struct nn_scheduler {
	struct nn_graph 			*nn_g;

	enum nn_scheduler_target 	target;

	unsigned int				total_memory_host_max;

	unsigned int				total_memory_device_max_buffer;
	unsigned int				total_memory_device_max_ops;

	unsigned int				graph_branches_parallel_max;

	unsigned int				input_batch_min;
	unsigned int				input_batch_max;
};

void nn_scheduler_init(struct nn_scheduler *nn_s, struct nn_graph *nn_g, enum nn_scheduler_target target, unsigned int total_memory_host_max, unsigned int total_memory_device_max_buffer, unsigned int total_memory_device_max_ops, unsigned int graph_branches_parallel_max, unsigned int input_batch_min, unsigned int input_batch_max);
void nn_scheduler_schedule(struct nn_scheduler *nn_s);

#endif /* SRC_NN_SCHEDULER_HPP_ */
