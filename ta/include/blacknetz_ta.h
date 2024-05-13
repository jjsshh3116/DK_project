#ifndef TA_DARKNETP_H
#define TA_DARKNETP_H

#include "darknet_TA.h"
#include "network_TA.h"

extern float *netta_truth;
extern int debug_summary_com;
extern int debug_summary_pass;

/*
 * This UUID is generated with uuidgen
 * the ITU-T UUID generator at http://www.itu.int/ITU-T/asn1/uuid.html
 */
#define TA_BLACKNETZ_UUID \
	{ 0xec9f2f40, 0x4a46, 0x49bf, \
		{ 0x99, 0xf8, 0xaf, 0xe2, 0x49, 0xe1, 0xed, 0x07} }

/* The function IDs implemented in this TA */
#define MAKE_NETWORK_CMD 1
#define WORKSPACE_NETWORK_CMD 2
#define MAKE_CONV_CMD 3
#define MAKE_MAX_CMD 4
#define MAKE_DROP_CMD 5
#define MAKE_CONNECTED_CMD 6
#define MAKE_SOFTMAX_CMD 7
#define MAKE_COST_CMD 8
#define FORWARD_CMD 9
#define BACKWARD_CMD 10
#define BACKWARD_ADD_CMD 11
#define UPDATE_CMD 12
#define NET_TRUTH_CMD 13
#define CALC_LOSS_CMD 14
#define TRANS_WEI_CMD 15
#define OUTPUT_RETURN_CMD 16
#define SAVE_WEI_CMD 17

#define FORWARD_BACK_CMD 18
#define BACKWARD_BACK_CMD 19
#define BACKWARD_BACK_ADD_CMD 20

#define MAKE_AVG_CMD 21

#define BLACK_FORWARD_CMD 22

void summary_array(char *print_name, float *arr, int n);

#endif /*TA_DARKNETP_H*/
