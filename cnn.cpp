#include <random>
#include <numeric>
#include <iostream>
#include <vector>
#include <functional>

#include "cnn.h"

int matrix_inner_product(matrix a, matrix b){
    int result = 0;
    for(int i=0; i<a.size();i++){
        result += std::inner_product(a[i], b[i]);
    }
    return result;
}

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

matrix CNN::convolution(matrix sample, matrix filter){
    // output size = [(W−K+2P)/S]+1
    int output_size = 1 + (sample.size() - filter_size)/stride;
    matrix output = matrix(output_size, vector<double>(output_size));

    for(int i =0; i<output_size; i++){
        for(int j=0; j<output_size;j++){
            matrix tmp = matrix(filter_size, vector<double>(filter_size));
            for(int k=0;k<output_size;k++){
            vector<double>::const_iterator first = sample[i+k].begin() + j;
            vector<double>::const_iterator last = sample[i+k].begin() + j +  filter_size;
            vector<double> newVec(first, last);
            tmp[k] = newVec;
            }
            output[i][j] = matrix_inner_product(filter, tmp);
        }
    }
    return output;
}