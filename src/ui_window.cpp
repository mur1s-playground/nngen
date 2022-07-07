/*
 * ui_window.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

/*
 * ui_window.c
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "ui_window.hpp"

#include <time.h>
#include <stdlib.h>

#include <SDL2/SDL_ttf.h>

#include "util.hpp"

void ui_window_init(struct ui_window *ui_w, const char *name, unsigned int position_x, unsigned int position_y, unsigned int width, unsigned int height, bool borderless) {
    util_chararray_from_const(name, &ui_w->name);
    ui_w->position_x = position_x;
    ui_w->position_y = position_y;
    ui_w->width = width;
    ui_w->height = height;

    int flags = 0;
    if (borderless) {
    	flags = SDL_WINDOW_BORDERLESS;
    }

    ui_w->sdl_window = SDL_CreateWindow(ui_w->name, position_x, position_y, width, height, flags);
    ui_w->sdl_renderer = SDL_CreateRenderer(ui_w->sdl_window, -1, 0);
    ui_w->sdl_texture = SDL_CreateTexture(ui_w->sdl_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height);

    ui_w->window_id = ui_window_id_get(ui_w);

    matrix_init(&ui_w->fb, height, width * 4);
}

void ui_window_destroy(struct ui_window *ui_w) {
        SDL_DestroyWindow(ui_w->sdl_window);
}

void ui_window_render(struct ui_window *ui_w) {
    SDL_UpdateTexture(ui_w->sdl_texture, NULL, ui_w->fb.m, ui_w->width * 4);
    SDL_RenderClear(ui_w->sdl_renderer);
    SDL_RenderCopy(ui_w->sdl_renderer, ui_w->sdl_texture, NULL, NULL);
    SDL_RenderPresent(ui_w->sdl_renderer);
}

void ui_window_render_text(struct ui_window *ui_w, int w_target, int h_target, const char *text, int pos_x, int pos_y, SDL_Color color) {
		TTF_Font* Sans = nullptr;

		int w_t = 0, h_t = 0;
		int font_size = 2;
		do {
			if (Sans != nullptr) TTF_CloseFont(Sans);
			font_size++;
			Sans = TTF_OpenFont("./font.ttf", font_size);
			if (!Sans) {
				printf("error (ui_window): open font failed\n");
				if (font_size == 2) {
					exit(1);
				} else {
					break;
				}
			}
			TTF_SizeText(Sans, text, &w_t, &h_t);
		} while (w_t < w_target && h_t < h_target);
		font_size--;
		TTF_CloseFont(Sans);
		Sans = TTF_OpenFont("./font.ttf", font_size);

		SDL_Surface* surfaceMessage = TTF_RenderText_Solid(Sans, text, color);
		SDL_Texture* Message = SDL_CreateTextureFromSurface(ui_w->sdl_renderer, surfaceMessage);

		SDL_Rect Message_rect;
		Message_rect.x = pos_x + ((w_target - w_t) / 2);
		Message_rect.y = pos_y + ((h_target - h_t) / 2);
		Message_rect.w = w_t;
		Message_rect.h = h_t;

		SDL_RenderCopy(ui_w->sdl_renderer, Message, NULL, &Message_rect);
		SDL_RenderPresent(ui_w->sdl_renderer);

		SDL_FreeSurface(surfaceMessage);
		SDL_DestroyTexture(Message);
}

void ui_windor_render_line(struct ui_window *ui_w, int node_position_x_from, int node_position_y_from, int node_position_x_to, int node_position_y_to, SDL_Color color) {
	SDL_SetRenderDrawColor(ui_w->sdl_renderer, color.r, color.g, color.b, SDL_ALPHA_OPAQUE);
	SDL_RenderDrawLine(ui_w->sdl_renderer, node_position_x_from, node_position_y_from, node_position_x_to, node_position_y_to);
	SDL_RenderPresent(ui_w->sdl_renderer);
}

unsigned int ui_window_id_get(struct ui_window *ui_w) {
        return SDL_GetWindowID(ui_w->sdl_window);
}




