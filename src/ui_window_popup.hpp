/*
 * ui_window_popup.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_UI_WINDOW_POPUP_MENU_HPP_
#define SRC_UI_WINDOW_POPUP_MENU_HPP_

#include "m_array.hpp"
#include "ui_window.hpp"
#include "ui_window_popup_item.hpp"

struct ui_window_popup {
	struct ui_window ui_w_popup;
	struct ui_window *ui_w_parent;

	struct m_array<struct ui_window_popup_item *>	items;
};

void ui_window_popup_init(struct ui_window_popup *ui_w_p, struct ui_window *ui_w);
void ui_window_popup_item_add(struct ui_window_popup *ui_w_p, struct ui_window_popup_item *ui_w_p_i);



#endif /* SRC_UI_WINDOW_POPUP_MENU_HPP_ */
