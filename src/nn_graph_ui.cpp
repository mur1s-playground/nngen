/*
 * nn_graph_ui.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "nn_graph_ui.hpp"
#include "nn_node.hpp"

#include <sstream>

#include "m_array.hpp"

struct m_array<struct nn_graph_ui *> nn_graph_uis;

void nn_graph_uis_init() {
	m_array_init(&nn_graph_uis);
}

void nn_graph_ui_draw(struct nn_graph_ui *nn_g_ui);

void nn_graph_uis_render() {
	int size = m_array_size(&nn_graph_uis);

	for (int w = 0; w < size; w++) {
		nn_graph_ui_draw(nn_graph_uis.data[w]);
	}
}

void nn_graph_ui_init(struct nn_graph_ui *nn_g_ui, struct nn_graph *nn_g) {
	nn_g_ui->nn_g = nn_g;
	nn_g_ui->draw = true;
	nn_g_ui->popup_draw = false;

	std::stringstream window_name;
	window_name << nn_g->directory << "/" << nn_g->prefix;

	ui_window_init(&nn_g_ui->ui_w, window_name.str().c_str(), 0, 0, 640, 480, false);

	m_array_push_back(&nn_graph_uis, nn_g_ui);
}

void nn_graph_ui_event_process(unsigned int window_id, SDL_Event event) {
	int size = m_array_size(&nn_graph_uis);
	bool is_popup = false;
	struct nn_graph_ui *nn_g_ui = nullptr;

	for (int w = 0; w < size; w++) {
		struct nn_graph_ui *nn_g_ui_tmp = nn_graph_uis.data[w];
		if (nn_g_ui_tmp->ui_w.window_id == window_id) {
			nn_g_ui = nn_g_ui_tmp;
			break;
		} else if (nn_g_ui_tmp->ui_popup.ui_w_popup.window_id == window_id) {
			nn_g_ui = nn_g_ui_tmp;
			is_popup = true;
			break;
		}
	}

	if (nn_g_ui != nullptr) {
		switch (event.type) {
			case SDL_WINDOWEVENT:
				if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
					return;
		        }
		        break;
		    case SDL_QUIT:
		    	break;
		    case SDL_KEYDOWN:
		        if (event.key.keysym.sym == SDLK_PLUS) {

		        } else if (event.key.keysym.sym == SDLK_MINUS) {

		        } else if (event.key.keysym.sym == SDLK_n) {

		        } else if (event.key.keysym.sym == SDLK_RIGHT) {

		        } else if (event.key.keysym.sym == SDLK_LEFT) {

		        }
		        break;
		    case SDL_MOUSEMOTION:
		    	if (is_popup) {
		    		nn_g_ui->popup_draw = true;
		    		nn_g_ui->ui_popup.ui_w_popup.mouse_position[0] = event.motion.x;
		    		nn_g_ui->ui_popup.ui_w_popup.mouse_position[1] = event.motion.y;
		    	} else {
		    		nn_g_ui->draw = true;
		    		nn_g_ui->ui_w.mouse_position[0] = event.motion.x;
		    		nn_g_ui->ui_w.mouse_position[1] = event.motion.y;
		    	}
		    	/*
		        if (event.button.button == SDL_BUTTON(SDL_BUTTON_RIGHT)) {
		        	//nn_window_position_move(nn_w, -event.motion.xrel, -event.motion.yrel);
		        }*/
		        break;
		    case SDL_MOUSEBUTTONDOWN:
		    	if (event.button.button == 3) {
		    		if (nn_g_ui->popup_open) {
		    			    //popup_close
		    		}
		    		ui_window_popup_init(&nn_g_ui->ui_popup, &nn_g_ui->ui_w);
		    	} else if (event.button.button == 1) {
		    		int x = event.motion.x;
		            int y = event.motion.y;
		    	}
		        break;
		    case SDL_MOUSEWHEEL:
		        //nn_window_zoom(nn_w, event.wheel.y * -0.1f);
		        break;
		}
	}
}

void nn_graph_ui_draw(struct nn_graph_ui *nn_g_ui) {
	if (nn_g_ui->draw) {
		ui_window_render(&nn_g_ui->ui_w);

		int row_height = 40;
		int node_width = 200;

		int margin = 10;

		struct nn_graph *nn_g = nn_g_ui->nn_g;
		int edges_c = m_array_size(&nn_g->edges);
		for (int e = 0; e < edges_c; e++) {
			struct nn_edge *nn_e = &nn_g->edges.data[e];

			struct nn_node *nn_from 	= nn_graph_node_get_by_id(nn_g, nn_e->id_from);
			int node_position_x_from 	= nn_from->ui_grid_position[0] * (node_width + margin) + node_width/2;
			int node_position_y_from 	= nn_from->ui_grid_position[1] * (row_height + margin) + row_height/2;
			struct nn_node *nn_to 		= nn_graph_node_get_by_id(nn_g, nn_e->id_to);
			int node_position_x_to	 	= nn_to->ui_grid_position[0] * (node_width + margin) + node_width/2;
			int node_position_y_to	 	= nn_to->ui_grid_position[1] * (row_height + margin) + row_height/2;

			SDL_Color edge_color = {255, 0, 0};

			ui_windor_render_line(&nn_g_ui->ui_w, node_position_x_from, node_position_y_from, node_position_x_to, node_position_y_to, edge_color);
		}

		int nodes_c = m_array_size(&nn_g->nodes);
		for (int n = 0; n < nodes_c; n++) {
			struct nn_node *nn_n = &nn_g->nodes.data[n];
			int node_position_x = nn_n->ui_grid_position[0] * (node_width + margin);
			int node_position_y = nn_n->ui_grid_position[1] * (row_height + margin);
			SDL_Color font_color = {255, 255, 255};
			ui_window_render_text(&nn_g_ui->ui_w, node_width, row_height, nn_n->id, node_position_x, node_position_y, font_color);
		}

		nn_g_ui->draw = false;
	}
	if (nn_g_ui->popup_draw) {
		ui_window_render(&nn_g_ui->ui_popup.ui_w_popup);
		SDL_Color font_color = {255, 255, 255};
		ui_window_render_text(&nn_g_ui->ui_popup.ui_w_popup, 50, 50, "test", 0, 0, font_color);
		nn_g_ui->draw = false;
	}
}

void nn_graph_ui_recalculate_graph_position_traverse(struct nn_graph *nn_g, struct nn_node *nn_from);

void nn_graph_ui_recalculate_graph_position(struct nn_graph_ui *nn_g_ui) {
	struct nn_graph *nn_g = nn_g_ui->nn_g;

	int nodes_c = m_array_size(&nn_g->nodes);

	int input_c = 0;
	for (int n = 0; n < nodes_c; n++) {
		struct nn_node *nn_n = &nn_g->nodes.data[n];
		if (nn_n->type == NN_N_T_INPUT) {
			nn_n->ui_grid_position[0] = input_c;
			nn_n->ui_grid_position[1] = 0;
			nn_graph_ui_recalculate_graph_position_traverse(nn_g, nn_n);
			input_c++;
		}
	}
}

void nn_graph_ui_recalculate_graph_position_traverse(struct nn_graph *nn_g, struct nn_node *nn_from) {
	int edges_c = m_array_size(&nn_g->edges);

	int outgoing_edges_c = 0;
	for (int e = 0; e < edges_c; e++) {
		struct nn_edge *nn_e = &nn_g->edges.data[e];
		if (strcmp(nn_e->id_from, nn_from->id) == 0) {
			struct nn_node *nn_to = nn_graph_node_get_by_id(nn_g, nn_e->id_to);
			if (nn_from->ui_grid_position[0] + outgoing_edges_c > nn_to->ui_grid_position[0]) {
				nn_to->ui_grid_position[0] = nn_from->ui_grid_position[0] + outgoing_edges_c;
			}
			if (nn_from->ui_grid_position[1] + 1 > nn_to->ui_grid_position[1]) {
				nn_to->ui_grid_position[1] = nn_from->ui_grid_position[1] + 1;
			}
			nn_graph_ui_recalculate_graph_position_traverse(nn_g, nn_to);
			outgoing_edges_c++;
		}
	}
}

void nn_graph_ui_dump_graph(struct nn_graph_ui *nn_g_ui) {
	struct nn_graph *nn_g = nn_g_ui->nn_g;

	int nodes_c = m_array_size(&nn_g->nodes);

	int min_x = INT_MAX;
	int max_x = -1;
	int min_y = INT_MAX;
	int max_y = -1;

	for (int n = 0; n < nodes_c; n++) {
		struct nn_node *nn_n = &nn_g->nodes.data[n];
		if (nn_n->ui_grid_position[0] < min_x) min_x = nn_n->ui_grid_position[0];
		if (nn_n->ui_grid_position[0] > max_x) max_x = nn_n->ui_grid_position[0];
		if (nn_n->ui_grid_position[1] < min_y) min_y = nn_n->ui_grid_position[1];
		if (nn_n->ui_grid_position[1] > max_y) max_y = nn_n->ui_grid_position[1];
	}
	printf("x(%d %d), y(%d %d)\n", min_x, max_x, min_y, max_y);

	for (int y = min_y; y <= max_y; y++) {
		for (int x = min_x; x <= max_x; x++) {
			for (int n = 0; n < nodes_c; n++) {
				struct nn_node *nn_n = &nn_g->nodes.data[n];
				if (nn_n->ui_grid_position[0] == x && nn_n->ui_grid_position[1] == y) {
					printf("%s", nn_n->id);
				}
			}
			printf("\t");
		}
		printf("\n");
	}
}
