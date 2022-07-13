/*
 * util.cpp
 *
 *  Created on: Jul 5, 2022
 *      Author: mur1
 */

#include "util.hpp"

#include <fstream>
#include <unistd.h>
#include <cstring>
#include <sstream>

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

std::string& util_ltrim(std::string& str, const std::string& chars) {
        str.erase(0, str.find_first_not_of(chars));
        return str;
}

std::string& util_rtrim(std::string& str, const std::string& chars) {
        str.erase(str.find_last_not_of(chars) + 1);
        return str;
}

std::string& util_trim(std::string& str, const std::string& chars) {
        return util_ltrim(util_rtrim(str, chars), chars);
}

std::vector<std::string>  util_split(const std::string& str, const std::string& separator) {
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

void util_ipv6_address_to_normalform(char* pbar) {
        char part_2[46];
        //memset(&part_2, 'x', 46);
        part_2[45] = '\0';

        int part_2_pos = 44;

        char* end = strchr(pbar, '\0');
        end--;
        int segment_ct = 0;
        int segs = 0;

        bool dbl = false;
        while (end >= pbar) {
                if (*end == ':') {
                        if (*(end - 1) == ':') {
                                dbl = true;
                        }
                        else if (segment_ct > 0) {
                                for (int s = segment_ct; s < 4; s++) {
                                        part_2[part_2_pos] = '0';
                                        part_2_pos--;
                                }
                        }
                        part_2[part_2_pos] = *end;
                        part_2_pos--;
                        segment_ct = 0;
                        segs++;
                }
                else {
                        part_2[part_2_pos] = *end;
                        part_2_pos--;
                        segment_ct = (segment_ct + 1) % 4;
                }
                end--;
                if (dbl) break;
        }
        //printf("part_2: %s\n", part_2);

        part_2_pos++;
        if (dbl) {
                char part_1[46];
                //memset(&part_1, 'x', 46);
                int part_1_pos = 44;
                part_1[45] = '\0';
                while (end >= pbar) {
                        if (*end == ':') {
                                if (segment_ct > 0) {
                                        for (int s = segment_ct; s < 4; s++) {
                                                part_1[part_1_pos] = '0';
                                                part_1_pos--;
                                        }
                                }
                                part_1[part_1_pos] = *end;
                                part_1_pos--;
                                segment_ct = 0;
                                segs++;
                        }
                        else {
                                part_1[part_1_pos] = *end;
                                part_1_pos--;
                                segment_ct = (segment_ct + 1) % 4;
                        }
                        end--;
                }
                //printf("part_1: %s\n", part_1);
                part_1_pos++;

                int total_pos = 0;
                memcpy(pbar, &part_1[part_1_pos], 45 - part_1_pos);
                total_pos = 45 - part_1_pos;
                for (int s = segs; s < 8; s++) {
                        pbar[total_pos] = '0';
                        total_pos++;
                        pbar[total_pos] = '0';
                        total_pos++;
                        pbar[total_pos] = '0';
                        total_pos++;
                        pbar[total_pos] = '0';
                        total_pos++;
                        if (s < 7) {
                                pbar[total_pos] = ':';
                                total_pos++;
                        }
                }
                memcpy(&pbar[total_pos], &part_2[part_2_pos], 45 - part_2_pos);
                total_pos += (45 - part_2_pos);
                pbar[total_pos] = '\0';
        }
        else {
                memcpy(pbar, &part_2[part_2_pos], 45 - part_2_pos);
                pbar[45 - part_2_pos] = '\0';
        }
}

char* util_issue_command(const char* cmd) {
        char* result;
        char buffer[256];
        int i;
        std::stringstream ss;
#ifdef _WIN32
        FILE* pipe = _popen(cmd, "r");
        if (pipe != nullptr) {
                while (fgets(buffer, sizeof buffer, pipe) != NULL) {
                        ss << buffer;
                }
                _pclose(pipe);
        }
#else
        FILE* pipe;
        pipe = (FILE*)popen(cmd, "r");
        if (!pipe) return NULL;
        while (!feof(pipe)) {
                if (fgets(buffer, 256, pipe) != NULL) {
                        for (i = 0; i < 256; i++) {
                                if (buffer[i] == '\0') break;
                                ss << buffer[i];
                        }
                }
        }
        pclose(pipe);
#endif
        result = (char*)malloc(ss.str().length() + 1);
        memcpy(result, ss.str().data(), ss.str().length());
        result[ss.str().length()] = '\0';
        return result;
}
