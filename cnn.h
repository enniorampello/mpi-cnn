#include <random>
#include <numeric>
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
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
        int num_classes;
        double lr;

        matrix image;
        vector<matrix> image_conv; // sample after convolution.
        vector<matrix> image_pool; // sample after max_pool
        
        vector<matrix> filters;
        vector<double> bias;
        matrix weights;

        matrix conv_output; // convolutional block output
        vector<double> penultimate_output; // flatened output
        vector<double> softmax_inp;
        vector<double> out;

        void init_normal_distribution();

        void init_filters();
        void init_biases();
        void init_weights();

        void load_image(matrix sample);
        
        matrix convolution(matrix sample, matrix filter); // for a single image
        void relu(matrix &sample);

        void flatten(matrix tmp);
        void fully_connected(); // for the fully connected layer
        void softmax();

        void back_prop(); // update all the weights

    public:
        CNN(int fltr_sz, int max_pool_sz, int n_fltrs, int strd, int num_nodes, double learning_rate);
        void train(matrix sample); // call this for every sample
        
        matrix max_pool(matrix sample); // for a single image
        void fwd_prop(matrix input_img);
        
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
    num_classes = 10;

    // init_normal_distribution();
    // init_filters();
    // init_biases();
    // init_weights();
}

// void CNN::init_normal_distribution(){
//     gen = mt19937(rd());
//     normal = normal_distribution<double>(0, 1); // (mean, std)
// }

void CNN::init_filters(){
    filters.push_back(matrix(filter_size, vector<double>(filter_size)));
    filters[0][0][0] = 1;
    filters[0][0][1] = -1;

    filters[0][1][0] = -1;
    filters[0][1][1] = -1;
}


//     bias = vector<double>(n_filters);
//     for (int i = 0; i < n_filters; i++){
//         bias[i] = normal(gen);
//     }
// }

void CNN::init_weights(){
//     // init the matrix of weights for the fully connected layer
//
    weights = matrix(penultimate_output.size(), vector<double>(num_classes));
    for(int c=0;c<num_classes;c++){
        for(int i=0;i<penultimate_output.size();i++){
                weights[i][c] = 0.01+0.1*c + i/10.0; 
            }
    }
    
    
}

void CNN::init_biases(){
//     // init the matrix of weights for the fully connected layer
//

    for(int i=0;i<num_classes;i++){
        bias.push_back(0.15 * i/10.0) ; 
    }
    

}


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


void CNN::flatten(matrix tmp){
    
    for(auto && v : tmp){
        penultimate_output.insert(penultimate_output.end(), v.begin(), v.end());
    }

}

void CNN::softmax(){
    double tot_sum=0.0;
    vector<double> tmp = softmax_inp;
    for(int i=0; i<softmax_inp.size(); i++){
        tmp[i] = exp(tmp[i]);
        tot_sum += tmp[i];
    }
    for(int i=0;i<tmp.size(); i++) out.push_back(tmp[i]/tot_sum);
    
}

void CNN::fully_connected(){
    // Temporary fix to make the penultimate_output as vector of vectors
    matrix pen = matrix(1, vector<double>(penultimate_output.size()));
    pen[0] = penultimate_output;
    matrix tmp = multiply(pen, weights);
    for(int i=0;i<tmp.size();i++){
        vector<double> c = vector_addition(tmp[i], bias);
        tmp[i] = c;
    }
    print_matrix(tmp);
    // Again the whole matrix thingy is converted back to whatever
    softmax_inp = tmp[0];
    softmax();

}

void CNN::fwd_prop(matrix input_img){
    // Convolution layer

    init_filters();
    conv_output = convolution(input_img, filters[0]);
    relu(conv_output);

    // TODO: Fix max pool layer
    // matrix tmp1 = max_pool(tmp);

    // flattening the output from max pool
    flatten(conv_output);
    init_weights();
    init_biases();

    fully_connected();
    print_vector(out);
}

