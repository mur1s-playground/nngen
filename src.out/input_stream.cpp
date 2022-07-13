/*
 * input_stream.cpp
 *
 *  Created on: Jul 13, 2022
 *      Author: mur1
 */

#include "input_stream.hpp"
#include "util.hpp"

void input_stream_init(struct input_stream *is) {
	is->running = true;

	is->buffer_slot_count = 5;
	is->batch_input_size = 31360000;
	is->batch_output_size = 400000;
	is->buffer_slot_size = (is->batch_input_size >= is->batch_output_size) * is->batch_input_size + (is->batch_input_size < is->batch_output_size) * is->batch_output_size;
	is->buffer = (char *) malloc(is->buffer_slot_count * is->buffer_slot_size);
	mutex_init(&is->buffer_lock);
	is->buffer_state = (enum input_stream_buffer_state *) malloc(is->buffer_slot_count * sizeof(enum input_stream_buffer_state));
	for (int i = 0; i < is->buffer_slot_count; i++) {
		is->buffer_state[i] = IS_B_S_FREE;
	}

	network_init(&is->nw);
	network_tcp_socket_create(&is->nw, "::", 2, 6666);
	network_tcp_socket_server_bind(&is->nw);
	network_tcp_socket_server_listen(&is->nw);

	thread_pool_init(&is->tp, 10);
	thread_create(&is->tp, (void *) &input_stream_loop, (void *) is);
}

void *input_stream_process(void *args) {
	struct input_stream_process_args *isp = (struct input_stream_process_args *) args;

	//WAITING FOR FREE BUFFER SLOT
	int buffer_id = -1;
	while (buffer_id < 0) {
		mutex_wait_for(&isp->is->buffer_lock);
		for (int b = 0; b < isp->is->buffer_slot_count; b++) {
			if (isp->is->buffer_state[b] == IS_B_S_FREE) {
				isp->is->buffer_state[b] = IS_B_S_WAITING_FOR_INPUT;
				buffer_id = b;
				break;
			}
		}
		mutex_release(&isp->is->buffer_lock);
		if (buffer_id < 0) {
			util_sleep(33);
		}
	}

	//READING INPUT INTO BUFFER
	char *buffer = &isp->is->buffer[buffer_id * isp->is->buffer_slot_size];
	unsigned int buffer_read_total = 0;

	while (buffer_read_total < isp->is->batch_input_size) {
		unsigned int buffer_read = 0;
		isp->client.read(&isp->client, (void *) &buffer[buffer_read_total], isp->is->batch_input_size - buffer_read_total, nullptr, &buffer_read);
		if (buffer_read >= 0) {
			buffer_read_total += buffer_read;
		}
	}

	mutex_wait_for(&isp->is->buffer_lock);
	isp->is->buffer_state[buffer_id] = IS_B_S_INPUT_READY;
	mutex_release(&isp->is->buffer_lock);

	//WAIT FOR OUTPUT
	while (true) {
		util_sleep(33);
		mutex_wait_for(&isp->is->buffer_lock);
		bool output_ready = isp->is->buffer_state[buffer_id] == IS_B_S_OUTPUT_READY;
		mutex_release(&isp->is->buffer_lock);
		if (output_ready) break;
	}

	//SENDING OUTPUT
	isp->client.send(&isp->client, (void *) &buffer, isp->is->batch_output_size);

	mutex_wait_for(&isp->is->buffer_lock);
	isp->is->buffer_state[buffer_id] == IS_B_S_FREE;
	mutex_release(&isp->is->buffer_lock);

	thread_terminated(&isp->is->tp, isp->thread_id);
	free(isp);

	return nullptr;
}

void *input_stream_loop(void *args) {
	struct input_stream *is = (struct input_stream *) args;
	while (is->running) {
		struct input_stream_process_args *isp = (struct input_stream_process_args *) malloc(sizeof(struct input_stream_process_args));
		isp->is = is;
		network_init(&isp->client);
		network_tcp_socket_server_accept(&is->nw, &isp->client);

		while (true) {
			int thread_id = thread_create(&is->tp, (void *) &input_stream_process, (void *) isp);

			if (thread_id < is->tp.size) {
				isp->thread_id = thread_id;
				break;
			} else {
				util_sleep(33);
			}
		}
	}
	return nullptr;
}

