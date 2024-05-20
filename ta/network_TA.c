#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "darknet_TA.h"
#include "blas_TA.h"
#include "network_TA.h"
#include "math_TA.h"
#include "activations_TA.h"
#include "convolutional_layer_TA.h"

#include "blacknetz_ta.h"
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

network_TA netta;
int roundnum = 0;
float err_sum = 0;
float avg_loss = -1;

float *ta_net_input;
float *ta_net_delta;
float *ta_net_output;

void make_network_TA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches)
{
    netta.n = n;

    //netta.seen = calloc(1, sizeof(size_t));
    netta.seen = calloc(1, sizeof(uint64_t));
    netta.layers = calloc(netta.n, sizeof(layer_TA));
    netta.t    = calloc(1, sizeof(int));
    netta.cost = calloc(1, sizeof(float));

    netta.learning_rate = learning_rate;
    netta.momentum = momentum;
    netta.decay = decay;
    netta.time_steps = time_steps;
    netta.notruth = notruth;
    netta.batch = batch;
    netta.subdivisions = subdivisions;
    netta.random = random;
    netta.adam = adam;
    netta.B1 = B1;
    netta.B2 = B2;
    netta.eps = eps;
    netta.h = h;
    netta.w = w;
    netta.c = c;
    netta.inputs = inputs;
    netta.max_crop = max_crop;
    netta.min_crop = min_crop;
    netta.max_ratio = max_ratio;
    netta.min_ratio = min_ratio;
    netta.center = center;
    netta.clip = clip;
    netta.angle = angle;
    netta.aspect = aspect;
    netta.saturation = saturation;
    netta.exposure = exposure;
    netta.hue = hue;
    netta.burn_in = burn_in;
    netta.power = power;
    netta.max_batches = max_batches;
    netta.workspace_size = 0;

    //netta.truth = net->truth; ////// ing network.c train_network
}

void forward_network_TA()
{
    if(roundnum == 0){
        // ta_net_input malloc so not destroy before addition backward
        ta_net_input = malloc(sizeof(float) * netta.layers[0].inputs * netta.layers[0].batch);
        ta_net_delta = malloc(sizeof(float) * netta.layers[0].inputs * netta.layers[0].batch);

        if(netta.workspace_size){
            printf("workspace_size=%.2lfMB\n", (double)netta.workspace_size / 1048576.0);
            netta.workspace = calloc(1, netta.workspace_size);
        }
    }

    roundnum++;

    layer_TA l = netta.layers[netta.index];

    if(l.delta){
        fill_cpu_TA(l.outputs * l.batch, 0, l.delta, 1);
    }


    l.forward_TA(l, netta);

    netta.input = l.output;

    ta_net_output = malloc(sizeof(float)*l.outputs);

    


    if(debug_summary_pass == 1){
        summary_array("forward_network / l.output", l.output, l.outputs*netta.batch);
    }


    calc_network_cost_TA();
}

void black_forward_network_TA(float *c, int c_size, black_pixels_TA *black_in_TEE, int black_size)
{
    if(roundnum == 0){
        // ta_net_input malloc so not destroy before addition backward
        ta_net_input = malloc(sizeof(float) * netta.layers[0].inputs * netta.layers[0].batch);
        ta_net_delta = malloc(sizeof(float) * netta.layers[0].inputs * netta.layers[0].batch);

        if(netta.workspace_size){
            printf("workspace_size=%.2lfMB\n", (double)netta.workspace_size / 1048576.0);
            netta.workspace = calloc(1, netta.workspace_size);
        }
    }

    roundnum++;

    layer_TA l = netta.layers[netta.index];
    l.black_in_TEE = black_in_TEE;
    l.black_size = black_size;

    for(int z = 0; z < l.black_size; z++){
        int c_index = l.black_in_TEE[z].C_index;
        float A_PART = l.black_in_TEE[z].weight;
        float b = l.black_in_TEE[z].B;

        c[c_index] += A_PART * b;
    }

    l.output = c;
    // for(int z = 0; z < c_size; z++){
    //     IMSG("%d After C otuput[%d]: %d \n", l.index, z, (int)(l.output[z] * 1000000.0));
    // }

    add_bias_TA(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
   
    activate_array_TA(l.output, l.outputs*l.batch, l.activation);
    // for(int z = 0; z < c_size; z++){
    //     IMSG("%d After activate_layer otuput[%d]: %d \n", l.index, z, (int)(l.output[z] * 1000000.0));
    // }

    // IMSG("black_farward_network_TA l.output memory address: %d\n", l.output);
    // IMSG("black_farward_network_TA netta.layers[netta.index].output memory address: %d\n", netta.layers[netta.index].output);
    // IMSG("black_farward_network_TA c memory address: %d\n", c);
    // netta.input = l.output;
    //netta.layers[netta.index].output = l.output;

    ta_net_output = malloc(sizeof(float)*l.outputs);


   calc_network_cost_TA();
}


void update_network_TA(update_args_TA a)
{
    int i;
    for(i = 0; i < netta.n; ++i){
        layer_TA l = netta.layers[i];
        if(l.update_TA){
            l.update_TA(l, a);
        }
    }
}


void calc_network_cost_TA()
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < netta.n; ++i){
        if(netta.layers[i].cost){
            sum += netta.layers[i].cost[0];
            ++count;
        }
    }
    *netta.cost = sum/count;
    err_sum += *netta.cost;
}


void calc_network_loss_TA(int n, int batch)
{
    float loss = (float)err_sum/(n*batch);

    if(avg_loss == -1) avg_loss = loss;
    avg_loss = avg_loss*.9 + loss*.1;

    char loss_char[20];
    char avg_loss_char[20];
    ftoa(loss, loss_char, 5);
    ftoa(avg_loss, avg_loss_char, 5);
    IMSG("loss = %s, avg loss = %s from the TA\n",loss_char, avg_loss_char);
    err_sum = 0;
}



//void backward_network_TA(float *ca_net_input, float *ca_net_delta)
void backward_network_TA(float *ca_net_input)
{
    int i;

    for(i = netta.n-1; i >= 0; --i){
        layer_TA l = netta.layers[i];

        if(l.stopbackward) break;
        if(i == 0){
            for(int z=0; z<l.inputs*l.batch; z++){
             // note: both ca_net_input and ca_net_delta are pointer
                ta_net_input[z] = ca_net_input[z];
                //ta_net_delta[z] = ca_net_delta[z]; zeros removing
                ta_net_delta[z] = 0.0f;
            }

            netta.input = ta_net_input;
            netta.delta = ta_net_delta;
        }else{
            layer_TA prev = netta.layers[i-1];
            netta.input = prev.output;
            netta.delta = prev.delta;
        }

        netta.index = i;
        l.backward_TA(l, netta);

        // when the first layer in TEE is a Dropout layer
        if((l.type == DROPOUT_TA) && (i == 0)){
            for(int z=0; z<l.inputs*l.batch; z++){
                ta_net_input[z] = l.output[z];
                ta_net_delta[z] = l.delta[z];
            }
            //netta.input = l.output;
            //netta.delta = l.delta;
        }
    }
}
