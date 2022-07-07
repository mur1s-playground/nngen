/*
 * util.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "util.hpp"

#include <stdlib.h>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <sstream>

void util_chararray_from_const(const char *str, char **out) {
        char *o = (char *)malloc(strlen(str) + 1);
        memcpy(o, str, strlen(str));
        o[strlen(str)] = '\0';

        *out = o;
}

void util_sleep(const unsigned int milliseconds) {
#ifdef _WIN32
        Sleep(milliseconds);
#else
        usleep(milliseconds * 1000);
#endif
}

std::vector<std::string> util_file_read(const char *file) {
    std::vector<std::string> result = std::vector<std::string>();

    std::ifstream t;
    t.open(file);
    while(t){
        std::string line;
        std::getline(t, line);
        result.push_back(line);
        if (t.eof()) break;
    }
    t.close();

    return result;
}

void util_file_write(const char *file, std::vector<std::string> content) {
	std::ofstream t;
	t.open(file, std::ios::trunc);
	for (int c = 0; c < content.size(); c++) {
		t << content[c] << "\n";
	}
	t.flush();
	t.close();
}

std::vector<std::string> util_split(const std::string& str, const std::string& separator) {
	std::vector<std::string> result;
	int start = 0;
	int end = str.find_first_of(separator, start);
	while (end != std::string::npos) {
		result.push_back(str.substr(start, end - start));
	    start = end + 1;
	    end = str.find_first_of(separator, start);
	}
	result.push_back(str.substr(start));
	return result;

}

void util_size_format_print(unsigned int size) {
	if (size < 1000) {
		printf("%dB", size);
	} else if (size < 1000000) {
		printf("%.1fkB", (float)size/1000.0f);
	} else if (size < 1000000000) {
		printf("%.1fMB", (float)size/1000000.0f);
	} else {
		printf("%.1fGB", (float)size/1000000000.0f);
	}
}

std::string util_replace_into(std::string line, std::string placeholder, std::string value) {
	std::string result(line.c_str());

	int pos = result.find(placeholder);
	while (pos != std::string::npos) {
		result.replace(pos, placeholder.length(), value);
		pos = result.find(placeholder);
	}

	return result;
}

std::string util_string_from_uint(unsigned int nr) {
	std::stringstream out_ss;
	out_ss << nr;
	std::string result(out_ss.str().c_str());
	return result;
}
