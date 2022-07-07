/*
 * random.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_RANDOM_HPP_
#define SRC_RANDOM_HPP_

#include <stdlib.h>
#include <time.h>
#include <cstdlib>

void random_init();

template<typename T>
void random_get(T range_from, T range_to, size_t length, T* out) {
        size_t numbers = 0;
        while (numbers < length) {
                float value = range_from + (rand()/(float)RAND_MAX) * ((float) range_to - (float) range_from);
                out[numbers] = (T) value;
                numbers++;
        }
}



#endif /* SRC_RANDOM_HPP_ */
