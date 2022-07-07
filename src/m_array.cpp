/*
 * m_array.cpp
 *
 *  Created on: Jun 26, 2022
 *      Author: mur1
 */

#include "m_array.hpp"

int m_array_contains(struct m_array<int> *m_arr, int d) {
	for (int i = 0; i < m_arr->data_c; i++) {
		if (d == m_arr->data[i]) return i;
	}
	return -1;
}

int m_array_contains(struct m_array<char *> *m_arr, char *d) {
	for (int i = 0; i < m_arr->data_c; i++) {
		if (strcmp(d, m_arr->data[i]) == 0) return i;

	}
	return -1;
}


