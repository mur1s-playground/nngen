/*
 * ui_window.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_UI_WINDOW_HPP_
#define SRC_UI_WINDOW_HPP_

#include <SDL2/SDL.h>

#include "matrix.hpp"

struct ui_window {
	unsigned int window_id;
    char *name;

    unsigned int position_x;
    unsigned int position_y;
    unsigned int width;
    unsigned int height;

    SDL_Window *sdl_window;
    SDL_Renderer *sdl_renderer;
    SDL_Texture *sdl_texture;

    struct matrix<unsigned char> fb;

    int mouse_position[2];
};

void ui_window_init(struct ui_window *ui_w, const char *name, unsigned int position_x, unsigned int position_y, unsigned int width, unsigned int height, bool borderless);
void ui_window_destroy(struct ui_window *ui_w);
void ui_window_render(struct ui_window *ui_w);
void ui_window_render_text(struct ui_window *ui_w, int w_target, int h_target, const char *text, int pos_x, int pos_y, SDL_Color color);
void ui_windor_render_line(struct ui_window *ui_w, int node_position_x_from, int node_position_y_from, int node_position_x_to, int node_position_y_to, SDL_Color color);
unsigned int ui_window_id_get(struct ui_window *ui_w);



#endif /* SRC_UI_WINDOW_HPP_ */
