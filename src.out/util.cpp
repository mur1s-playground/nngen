/*
 * util.cpp
 *
 *  Created on: Jul 5, 2022
 *      Author: mur1
 */

#include "util.hpp"

#include <fstream>
#include <unistd.h>

void util_file_read(const char *name, size_t from_bytes, size_t length, char *out) {
        std::ifstream                in_file;
//      ifstream::pos_type      size;

        in_file.open(name, std::ios::in|std::ios::binary|std::ios::ate);

//      size = in_file.tellg();
        in_file.seekg(from_bytes, std::ios::beg);

        in_file.read(out, length);
}

void util_file_write(const char *name, size_t length, char *data) {
        std::ofstream                out_file;
        out_file.open(name, std::ios::out|std::ios::binary);

        out_file.write(data, length);

        out_file.close();
}

void util_sleep(const unsigned int milliseconds) {
#ifdef _WIN32
        Sleep(milliseconds);
#else
        usleep(milliseconds * 1000);
#endif
}
