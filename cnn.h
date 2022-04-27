#include <random>
#include <numeric>
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <math.h>
#include "utils.h"

using namespace std;
using matrix = vector<vector<double>>;

#define max(a,b) ((a)>(b)?(a):(b))


class CNN {
    private:
        // random_device rd;
        // mt19937 gen;
        // normal_distribution<double> normal;

        int filter_size;
        int max_pool_size;
        int n_filters;
        int stride;
        int n_nodes;
        double lr;

        matrix image;
        vector<matrix> image_conv; // sample after convolution.
        vector<matrix> image_pool; // sample after max_pool
        
        vector<matrix> filters;
        vector<double> bias;
        matrix weights;

        vector<double> out;

        void init_normal_distribution();

        void init_filters();
        void init_biases();
        void init_weights();

        void load_image(matrix sample);
        
        void fwd_pass(); // for the fully connected layer
        vector<double> softmax(vector<double> out);

        void fwd_prop();
        void back_prop(); // update all the weights

    public:
        CNN(int fltr_sz, int max_pool_sz, int n_fltrs, int strd, int num_nodes, double learning_rate);
        void train(matrix sample); // call this for every sample
        matrix convolution(matrix sample, matrix filter); // for a single image
        void relu(matrix &sample);
        matrix max_pool(matrix sample); // for a single image

        // move the following functions to private after testing
        vector<double> softmax_backprop(matrix in, vector<double> in_flat, vector<double> out, vector<double> out_softmax, int label);
};


int matrix_inner_product(matrix a, matrix b){
    double result = 0.0;
    for(int i=0; i<a.size();i++){
        result += inner_product(a[i].begin(), a[i].end(), b[i].begin(), 0);   
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

    // init_normal_distribution();
    // init_filters();
    // init_biases();
    // init_weights();
}

// void CNN::init_normal_distribution(){
//     gen = mt19937(rd());
//     normal = normal_distribution<double>(0, 1); // (mean, std)
// }

// void CNN::init_filters(){
//     filters = vector<matrix>(n_filters, matrix(filter_size, vector<double>(filter_size)));
//     for (int i = 0; i < filters[0].size(); i++){
//         for (int j = 0; j < filter_size; j++){
//             for (int k = 0; k < filter_size; k++){
//                 filters[i][j][k] = normal(gen); // sample from the normal distribution
//             }   
//         }
//     }
// }

// void CNN::init_biases(){
//     bias = vector<double>(n_filters);
//     for (int i = 0; i < n_filters; i++){
//         bias[i] = normal(gen);
//     }
// }

// void CNN::init_weights(){
//     // init the matrix of weights for the fully connected layer
// }

matrix CNN::convolution(matrix sample, matrix filter){
    // output size = [(W−K+2P)/S]+1
    int output_size =  (1 + sample.size() - filter_size)/stride;
    matrix output = matrix(output_size, vector<double>(output_size));
    for(int i =0; i<output_size; i++){
        for(int j=0; j<output_size;j++){
            matrix tmp = matrix(filter_size, vector<double>(filter_size));
            for(int k=0;k<filter_size;k++){
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


void CNN::relu(matrix &sample){
    
    for_each(
        sample.begin(), 
        sample.end(),
        [](vector<double> &tmp){
            replace_if(tmp.begin(), tmp.end(), [](double &i){return i<0.0;}, 0.0);
            });
}



matrix CNN::max_pool(matrix sample){
    int s = sample.size()/max_pool_size;
    matrix out( s, vector<double>(s, 0));
    for (size_t i = 0; i < s; i++)
    {
        for (size_t j = 0; j < s; j++)
        {
            for (size_t k = 0; k < max_pool_size; k++)
            {
                for (size_t l = 0; l < max_pool_size; l++)
                {
                    float val = sample[i*max_pool_size + k][j*max_pool_size + l];
                    out[i][j] = max(out[i][j], val);      
                }   
            }
        }
    }

    return out; 

}

vector<double> CNN::softmax_backprop(matrix in, vector<double> in_flat, vector<double> out, vector<double> out_softmax, int label){
    vector<double> d_L_d_out = vector<double>(out.size(), 0);
    vector<double> t_exp = vector<double>(out.size());
    vector<double> d_out_d_t = vector<double>(out.size());
    vector<double> d_t_d_w = vector<double>(in_flat.size());
    vector<double> d_L_d_inputs = vector<double>(in_flat.size());
    matrix d_L_d_w = matrix(in_flat.size(), vector<double>(n_nodes));
    float d_t_d_b = 1;
    float sum;

    d_L_d_out[label] = -1 / out[label];

    for (int i = 0; i < out.size(); i++)
        t_exp[i] = exp(out[i]);
    
    for_each(t_exp.begin(), t_exp.end(), [&] (float n) {
        sum += n;
    });

    // d_out_d_t = -t_exp[label] * t_exp / (S ** 2)
    for (int i = 0; i < out.size(); i++)
        d_out_d_t[i] = -t_exp[label] * t_exp[i] / pow(sum, 2);
    
    d_out_d_t[label] = t_exp[label] * (sum - t_exp[label]) / pow(sum, 2);

    
    


    return gradient;
}