/*
 * ui_window_popup_item.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_UI_WINDOW_POPUP_ITEM_HPP_
#define SRC_UI_WINDOW_POPUP_ITEM_HPP_

#include "m_array.hpp"

struct ui_window_popup_item {
	char *name;
	void *callback;

	//struct m_array<struct ui_window_popup_item *>	*sub;
};

void ui_window_popup_item_new(struct ui_window_popup_item **item, const char *name, void *callback);

#endif /* SRC_UI_WINDOW_POPUP_ITEM_HPP_ */
