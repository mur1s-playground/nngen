/*
 * util.hpp
 *
 *  Created on: Jul 5, 2022
 *      Author: mur1
 */

#ifndef SRC_OUT_UTIL_HPP_
#define SRC_OUT_UTIL_HPP_

#include <cstddef>
#include <string>
#include <vector>

void util_file_read(const char *name, size_t from_bytes, size_t length, char *out);
void util_file_write(const char *name, size_t length, char *data);
void util_sleep(const unsigned int milliseconds);

std::string& util_ltrim(std::string& str, const std::string& chars);
std::string& util_rtrim(std::string& str, const std::string& chars);
std::string& util_trim(std::string& str, const std::string& chars);
std::vector<std::string>  util_split(const std::string& str, const std::string& separator);
void util_ipv6_address_to_normalform(char* pbar);
char* util_issue_command(const char* cmd);

#endif /* SRC_OUT_UTIL_HPP_ */
