/*
 * util.hpp
 *
 *  Created on: Jul 5, 2022
 *      Author: mur1
 */

#ifndef SRC_OUT_UTIL_HPP_
#define SRC_OUT_UTIL_HPP_

#include <cstddef>

void util_file_read(const char *name, size_t from_bytes, size_t length, char *out);
void util_file_write(const char *name, size_t length, char *data);
void util_sleep(const unsigned int milliseconds);


#endif /* SRC_OUT_UTIL_HPP_ */
