/*
 * m_array.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_M_ARRAY_HPP_
#define SRC_M_ARRAY_HPP_

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>
#include <cstring>

template<typename T>
struct m_array {
	T *data;
	unsigned int data_c;
	unsigned int alloc_c;
};

template<typename T>
void m_array_init(struct m_array<T> *m_arr) {
	m_arr->data 		= nullptr;
	m_arr->data_c 		= 0;
	m_arr->alloc_c 		= 0;
}

template<typename T>
void m_array_destroy(struct m_array<T> *m_arr) {
	if (m_arr->alloc_c > 0) {
		free(m_arr->data);
		m_array_init(m_arr);
	}
}

template<typename T>
void m_array_2device(struct m_array<T> *m_arr, void **device_ptr) {
	unsigned int size_req = m_array_size(m_arr);
	cudaMalloc(device_ptr, size_req * sizeof(T));
	cudaMemcpy(*device_ptr, m_arr->data, size_req, cudaMemcpyHostToDevice);
}

template<typename T>
void m_array_push_back(struct m_array<T> *m_arr, T d) {
	if (m_arr->data_c == m_arr->alloc_c) {
		if (m_arr->alloc_c == 0) {
			m_arr->data = (T *) malloc(sizeof(T));
			m_arr->alloc_c++;
		} else {
			m_arr->data = (T *) realloc(m_arr->data, (m_arr->alloc_c + 1) * sizeof(T));
			m_arr->alloc_c++;
		}
	}
	memcpy(&m_arr->data[m_arr->data_c], &d, sizeof(T));
	m_arr->data_c++;
}

int m_array_contains(struct m_array<int> *m_arr, int d);
int m_array_contains(struct m_array<char *> *m_arr, char *d);

/*
template<typename T>
int m_array_contains(struct m_array<T> *m_arr, T d) {
	for (int i = 0; i < m_arr->data_c; i++) {
		if (typeid(T) == typeid(char *)) {
			if (strcmp(d, m_arr->data[i]) == 0) return i;
		} else if (typeid(T) == typeid(int)) {
			if (d == m_arr->data[i]) return i;
		}
	}
	return -1;
}
*/

template<typename T>
unsigned int m_array_size(struct m_array<T> *m_arr) {
	return m_arr->data_c;
}

template<typename T>
void m_array_dump(struct m_array<T> *m_arr) {
	printf("m_array_dump %d\n", m_arr->data_c);
	for (int i = 0; i < m_arr->data_c; i++) {
		if (typeid(T) == typeid(char *) || typeid(T) == typeid(unsigned char *)) {
			printf("%d: %s\n", i, m_arr->data[i]);
		}
	}
	printf("m_array_dump end\n");
}

#endif /* SRC_M_ARRAY_HPP_ */
