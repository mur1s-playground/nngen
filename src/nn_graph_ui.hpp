/*
 * nn_graph_ui.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_GRAPH_UI_HPP_
#define SRC_NN_GRAPH_UI_HPP_

#include "ui_window.hpp"
#include "ui_window_popup.hpp"
#include "nn_graph.hpp"

#include <SDL2/SDL.h>


struct nn_graph_ui {
	struct nn_graph 		*nn_g;
	bool					draw;

	struct ui_window 		ui_w;

	bool 					popup_open;
	struct ui_window_popup 	ui_popup;
	bool					popup_draw;
};

void nn_graph_uis_init();
void nn_graph_uis_render();

void nn_graph_ui_recalculate_graph_position(struct nn_graph_ui *nn_g_ui);
void nn_graph_ui_dump_graph(struct nn_graph_ui *nn_g_ui);

void nn_graph_ui_init(struct nn_graph_ui *nn_g_ui, struct nn_graph *nn_g);
void nn_graph_ui_event_process(unsigned int window_id, SDL_Event event);


#endif /* SRC_NN_GRAPH_UI_HPP_ */
