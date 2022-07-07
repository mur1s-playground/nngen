/*
 * nn_dimension.hpp
 *
 *  Created on: Jun 26, 2022
 *      Author: mur1
 */

#ifndef SRC_NN_DIMENSION_HPP_
#define SRC_NN_DIMENSION_HPP_

struct nn_dimension {
	unsigned int cols;
	unsigned int rows;
	unsigned int channels;
	char		 *assoc_id_in;
	char		 *assoc_id_out;
	//unsigned int t;
};



#endif /* SRC_NN_DIMENSION_HPP_ */
