/*
 * main.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "random.hpp"
#include "nn_graph.hpp"
#include "nn_id_factory.hpp"
#include "nn_graph_ui.hpp"
#include "util.hpp"
#include "thread.hpp"
#include "shell.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include <string.h>

struct ThreadPool main_thread_pool;

void init_static() {
	SDL_Init(SDL_INIT_VIDEO);
	TTF_Init();

	random_init();
	thread_pool_init(&main_thread_pool, 5);
	nn_graphs_init();
	nn_graph_uis_init();
}

int main() {
	init_static();

	SDL_Event event;

	thread_create(&main_thread_pool, (void *) &shell_loop, nullptr);

	/*
	struct nn_graph nn_g;
	nn_graph_init(&nn_g, "nns/mnist_c", "mnist_c_");

	struct nn_graph_ui nn_g_ui;
	nn_graph_ui_init(&nn_g_ui, &nn_g);

	//nn_id_factory_dump(&nn_g.nn_id_f);
	nn_graph_dump(&nn_g);

	nn_graph_ui_recalculate_graph_position(&nn_g_ui);
	//nn_graph_ui_dump_graph(&nn_g_ui);
	*/

	while(true) {
		while (SDL_PollEvent(&event) != 0) {
			unsigned int window_id = event.window.windowID;
			nn_graph_ui_event_process(window_id, event);
		}

		nn_graph_uis_render();

		mutex_wait_for(&shell_cmd_lock);
		for (int sc = 0; sc < shell_cmd_queue.size(); sc++) {
			std::vector<std::string> args = shell_cmd_queue[sc];
			if (strstr(args[0].c_str(), "nn_load") != nullptr) {
				int id = std::stoi(args[1].c_str());
			    const char *directory 	= args[2].c_str();
			    const char *prefix		= args[3].c_str();
			    struct nn_graph *nn_g = (struct nn_graph *) malloc(sizeof(struct nn_graph));
			    nn_graph_init(nn_g, directory, prefix);
			} else if (strstr(args[0].c_str(), "nn_window") != nullptr) {
                int id = std::stoi(args[1].c_str());
                struct nn_graph *nn_g = nn_graph_get(id);
                struct nn_graph_ui *nn_g_ui = (struct nn_graph_ui *) malloc(sizeof(struct nn_graph_ui));

                nn_graph_ui_init(nn_g_ui, nn_g);
                nn_graph_ui_recalculate_graph_position(nn_g_ui);
			} else if (strstr(args[0].c_str(), "nn_dump") != nullptr) {
				int id = std::stoi(args[1].c_str());
				struct nn_graph *nn_g = nn_graph_get(id);
				nn_graph_recalculate_dimensions(nn_g);
				nn_graph_dump(nn_g);
			}
		}
		shell_cmd_queue.clear();
		mutex_release(&shell_cmd_lock);

		util_sleep(33);
	}

	return 0;
}

