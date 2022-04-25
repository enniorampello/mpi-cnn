#include <random>
#include "cnn.h"
#define max(a,b) ((a)>(b)?(a):(b))


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
    for(int i =0; i<CNN::image.size(); i++){

    }
}

void CNN::relu(matrix &sample){
    for (int i = 0; i < sample.size(); i++)
    {
        for (int j = 0; j < sample[i].size(); j++)
        {
            if(sample[i][j] < 0.0){
                sample[i][j] = 0.0;
            }
        }
    }
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

//testing the convolution function
// void main(){

//     matrix test; 
//     matrix filter;
//     test[0][0] = 1;
//     test[0][1] = 1;
//     test[0][2] = 1;

//     test[1][0] = 1;
//     test[1][1] = 1;
//     test[1][2] = 1;

//     test[2][0] = 1;
//     test[2][1] = 1;
//     test[2][2] = 1;

//     filter[0][0] = -1;
//     filter[0][1] = -1;
//     filter[0][2] = -1;

//     filter[1][0] = -1;
//     filter[1][1] = -1;
//     filter[1][2] = -1;

//     filter[2][0] = -1;
//     filter[2][1] = -1;
//     filter[2][2] = -1;

//     matrix ans = convolution(test, filter);
    

// }