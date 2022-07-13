/*
 * input_stream.hpp
 *
 *  Created on: Jul 13, 2022
 *      Author: mur1
 */

#ifndef SRC_OUT_INPUT_STREAM_HPP_
#define SRC_OUT_INPUT_STREAM_HPP_

#include "thread.h"
#include "network.h"
#include "mutex.h"

enum input_stream_buffer_state {
	IS_B_S_FREE,
	IS_B_S_WAITING_FOR_INPUT,
	IS_B_S_INPUT_READY,
	IS_B_S_WAITING_FOR_OUTPUT,
	IS_B_S_OUTPUT_READY,
};

struct input_stream_process_args {
	struct input_stream *is;
	struct Network 	client;

	int thread_id;

};

struct input_stream {
	struct Network 						nw;

	struct ThreadPool 					tp;

	bool 								running;

	unsigned int						batch_input_size;
	unsigned int						batch_output_size;

	unsigned int						buffer_slot_size;
	int									buffer_slot_count;
	struct mutex 						buffer_lock;
	char 								*buffer;
	enum input_stream_buffer_state 		*buffer_state;
};

void input_stream_init(struct input_stream *is);

void *input_stream_process(void *args);
void *input_stream_loop(void *args);


#endif /* SRC_OUT_INPUT_STREAM_HPP_ */
