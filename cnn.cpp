#include <random>

#include "cnn.h"


CNN::CNN(int fltr_sz, int max_pool_sz, int n_fltrs, int strd, int num_nodes, double learning_rate){
    filter_size = fltr_sz;
    max_pool_size = max_pool_sz;
    n_filters = n_fltrs;
    stride = strd;
    n_nodes = num_nodes;
    lr = learning_rate;

    init_normal_distribution();
    init_filters();
    init_biases();
    init_weights();
}

void CNN::init_normal_distribution(){
    gen = mt19937(rd());
    normal = normal_distribution<double>(0, 1); // (mean, std)
}

void CNN::init_filters(){
    filters = vector<matrix>(n_filters, matrix(filter_size, vector<double>(filter_size)));
    for (int i = 0; i < filters[0].size(); i++){
        for (int j = 0; j < filter_size; j++){
            for (int k = 0; k < filter_size; k++){
                filters[i][j][k] = normal(gen); // sample from the normal distribution
            }   
        }
    }
}

void CNN::init_biases(){
    bias = vector<double>(n_filters);
    for (int i = 0; i < n_filters; i++){
        bias[i] = normal(gen);
    }
}

void CNN::init_weights(){
    // init the matrix of weights for the fully connected layer
}