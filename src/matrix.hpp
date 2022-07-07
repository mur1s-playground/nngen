/*
 * matrix.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_MATRIX_HPP_
#define SRC_MATRIX_HPP_

#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <typeinfo>
#include <cmath>

template<typename T>
struct matrix {
        int cols;
        int rows;

        T *m;
        T *m_dev;

        cudaStream_t c_stream;
};

template<typename T>
__host__ void matrix_init(struct matrix<T> *m, int rows, int cols) {
        m->cols = cols;
        m->rows = rows;

        m->m = (T *) malloc(cols * rows * sizeof(T));

        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void **) &m->m_dev, cols * rows * sizeof(T));

        if (err != cudaSuccess) {
                printf("cuda malloc error %d %d\n", rows, cols);
        }

        m->c_stream = nullptr;
}

template<typename T>
__host__ void matrix_init_block(struct matrix<T> **m, int rows, int cols, int count) {
        *m = (struct matrix<T> *) malloc(count * sizeof(struct matrix<T>));

        T *ptr = (T *) malloc(count * cols * rows * sizeof(T));
        T *dev_ptr = nullptr;

        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void **) &dev_ptr, count * cols * rows * sizeof(T));

        if (err != cudaSuccess) {
                printf("cuda malloc error\n");
        }

        for (int c = 0; c < count; c++) {
                (*m)[c].cols = cols;
                (*m)[c].rows = rows;
                (*m)[c].m = &ptr[c * cols * rows];
                (*m)[c].m_dev = dev_ptr + (c * cols * rows);
                (*m)[c].c_stream = nullptr;
        }
}



#endif /* SRC_MATRIX_HPP_ */
