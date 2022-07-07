/*
 * util.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_UTIL_HPP_
#define SRC_UTIL_HPP_

#include <vector>
#include <string>

void util_chararray_from_const(const char *str, char **out);
void util_sleep(const unsigned int milliseconds);
std::vector<std::string> util_file_read(const char *file);
void util_file_write(const char *file, std::vector<std::string> content);
std::vector<std::string> util_split(const std::string& str, const std::string& separator);
void util_size_format_print(unsigned int size);
std::string util_replace_into(std::string line, std::string placeholder, std::string value);
std::string util_string_from_uint(unsigned int nr);


#endif /* SRC_UTIL_HPP_ */
