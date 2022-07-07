/*
 * ui_window_popup_item.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "ui_window_popup_item.hpp"

#include "util.hpp"

void ui_window_popup_item_new(struct ui_window_popup_item **item, const char *name, void *callback) {
	*item = (struct ui_window_popup_item *) malloc(sizeof(struct ui_window_popup_item));
	util_chararray_from_const(name, &((*item)->name));
	(*item)->callback = callback;
}


