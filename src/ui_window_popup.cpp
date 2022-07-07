/*
 * ui_window_popup.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */
#include "ui_window_popup.hpp"

void ui_window_popup_init(struct ui_window_popup *ui_w_p, struct ui_window *ui_w) {
	ui_w_p->ui_w_parent = ui_w;

	int parent_x = 0;
	int parent_y = 0;
	SDL_GetWindowPosition(ui_w_p->ui_w_parent->sdl_window, &parent_x, &parent_y);
	ui_window_init(&ui_w_p->ui_w_popup, "popup", parent_x + ui_w_p->ui_w_parent->mouse_position[0], parent_y + ui_w_p->ui_w_parent->mouse_position[1], 300, 300, true);
	SDL_RaiseWindow(ui_w_p->ui_w_popup.sdl_window);
}

void ui_window_popup_item_add(struct ui_window_popup *ui_w_p, struct ui_window_popup_item *ui_w_p_i) {

}

