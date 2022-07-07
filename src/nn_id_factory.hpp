/*
 * nn_id_factory.hpp
 *
 *  Created on: Jun 24, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_ID_FACTORY_HPP_
#define SRC_NN_ID_FACTORY_HPP_

#include "m_array.hpp"

struct nn_id_factory {
	struct m_array<char *>	ids;
};

void 	nn_id_factory_init(struct nn_id_factory *nn_id_f);
int 	nn_id_factory_id_add(struct nn_id_factory *nn_id_f, const char *id);
void 	nn_id_factory_id_get(struct nn_id_factory *nn_id_f, char **id_out);
void 	nn_id_factory_dump(struct nn_id_factory *nn_id_f);
int 	nn_id_factory_nid_get(struct nn_id_factory *nn_id_f, const char *id);

#endif /* SRC_NN_ID_FACTORY_HPP_ */
