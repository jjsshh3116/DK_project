#ifndef MAIN_CA_H
#define MAIN_CA_H

#include <err.h>
#include <stdio.h>
#include <string.h>

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>
#include "darknet.h"

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

#define TA_BLACKNETZ_UUID \
	{ 0xec9f2f40, 0x4a46, 0x49bf, \
		{ 0x99, 0xf8, 0xaf, 0xe2, 0x49, 0xe1, 0xed, 0x07} }
		
extern TEEC_Context ctx;
extern TEEC_Session sess;

extern float *net_input_back;
extern float *net_delta_back;
extern float *net_output_back;
extern char state;

void debug_plot(char *filename, int num, float *tobeplot, int length);

void make_network_CA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches);

void update_net_agrv_CA_allocateSM(int workspace_size, float *workspace);

void update_net_agrv_CA(int cond, int workspace_size, float *workspace);

void make_convolutional_layer_CA(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot, int index);

void make_maxpool_layer_CA(int batch, int h, int w, int c, int size, int stride, int padding, int index);

void make_avgpool_layer_CA(int batch, int h, int w, int c);

void make_dropout_layer_CA(int batch, int inputs, float probability, int w, int h, int c, float *net_prev_output, float *net_prev_delta);

void make_connected_layer_CA(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

void make_softmax_layer_CA(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss);

void make_cost_layer_CA(int batch, int inputs, COST_TYPE cost_type, float scale, float ratio, float noobject_scale, float thresh);

void forward_network_CA(float *net_input, int net_inputs, int net_batch, int net_train, int net_index);

void black_forward_network_CA(float *c, black_pixels *black_in_TEE, layer l, int net_index);

void forward_network_back_CA(float *l_output, int net_inputs, int net_batch, int net_index);

void backward_network_CA_addidion(float *l_output, float *l_delta, int net_inputs, int net_batch);

void backward_network_CA(float *net_input, int l_inputs, int batch, int net_train);

void backward_network_back_CA_addidion(float *l_output, float *l_delta, int net_inputs, int net_batch);

void backward_network_back_CA(float *net_input, int l_inputs, int net_batch, float *net_delta);

void update_network_CA(update_args a);

void net_truth_CA(float *net_truth, int net_truths, int net_batch);

void calc_network_loss_CA(int n, int batch);

void net_output_return_CA(int net_outputs, int net_batch);

void transfer_weights_CA(float *vec, int length, int layer_i, char type, int additional);

void save_weights_CA(float *vec, int length, int layer_i, char type);

void summary_array(char *print_name, float *arr, int n);

#endif
