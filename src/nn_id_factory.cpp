/*
 * nn_id_factory.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#include "nn_id_factory.hpp"

#include "util.hpp"
#include "random.hpp"

#include <stdio.h>
#include <stdlib.h>

void nn_id_factory_init(struct nn_id_factory *nn_id_f) {
	m_array_init(&nn_id_f->ids);
}

int nn_id_factory_id_add(struct nn_id_factory *nn_id_f, const char *id) {
	char *id_ = nullptr;
	util_chararray_from_const(id, &id_);
	if (m_array_contains(&nn_id_f->ids, id_) == -1) {
		m_array_push_back(&nn_id_f->ids, id_);
		return m_array_size(&nn_id_f->ids) - 1;
	}
	free(id_);
	printf("error (nn_id_factory): id %s already exists\n", id);
	return -1;
}

int nn_id_factory_nid_get(struct nn_id_factory *nn_id_f, const char *id) {
	return m_array_contains(&nn_id_f->ids, (char *) id);
}

void nn_id_factory_id_get(struct nn_id_factory *nn_id_f, char **id_out) {
	bool found_new = false;
	while (!found_new) {
		char id_new[17];
		id_new[16] = '\0';
		random_get((char)97, (char)122, (char)16, (char *) id_new);
		int add_id = nn_id_factory_id_add(nn_id_f, (const char *)id_new);
		if (add_id > -1) {
			found_new = true;
			*id_out = nn_id_f->ids.data[add_id];
		}
	}
}

void nn_id_factory_dump(struct nn_id_factory *nn_id_f) {
	m_array_dump(&nn_id_f->ids);
}

