/*
 * externaliser.hpp
 *
 *  Created on: Jun 27, 2022
 *      Author: mur1
 */

#ifndef SRC_EXTERNALISER_HPP_
#define SRC_EXTERNALISER_HPP_

#include "m_array.hpp"

struct externaliser {
	char *out_ext;
	unsigned int position;
	unsigned int alloc_c;
};

void externaliser_init(struct externaliser *e, unsigned int preallocate);
void externaliser_destroy(struct externaliser *e);

void externaliser_int_add(struct externaliser *e, int to_add);
void externaliser_raw_add(struct externaliser *e, void *raw, int size_to_add);
void externaliser_str_add(struct externaliser *e, char *str, unsigned int len);

template<typename T>
void externaliser_m_array_add(struct externaliser *e, m_array<T> *m_arr) {
	int size = m_array_size(m_arr);
	externaliser_int_add(e, size);
	externaliser_raw_add(e, m_arr->data, size * sizeof(T));
}

void externaliser_2device(struct externaliser *e, void **device_ptr);

#endif /* SRC_EXTERNALISER_HPP_ */
