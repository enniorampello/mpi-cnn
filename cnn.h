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

normal_distribution<double> normal = normal_distribution<double>(0, 1); 

class CNN {
    private:
        int filter_size;
        int max_pool_size;
        int n_filters;
        int stride;
        int n_nodes;
        int num_classes;
        double lr;
        mt19937 gen;

        matrix image;
        int label;
        vector<matrix> image_conv; // sample after convolution.
        vector<matrix> image_pool; // sample after max_pool
        
        vector<matrix> filters;
        vector<double> bias;
        matrix weights;

        matrix conv_output; // convolutional block output
        vector<double> conv_out_flat; // flatened output
        vector<double> softmax_inp;
        vector<double> out;

        void init_filters();
        void init_biases();
        void init_weights();

        void load_image(matrix sample);
        
        matrix convolution(matrix sample, matrix filter); // for a single image
        void relu(matrix &sample);

        void flatten(matrix tmp); // TODO 
        void fully_connected(); // for the fully connected layer
        void softmax();

    public:
        normal_distribution<double> normal;

        CNN(int fltr_sz, int max_pool_sz, int n_fltrs, int strd, int num_nodes, double learning_rate, mt19937 gen);
        void train(matrix sample); // call this for every sample
        
        matrix max_pool(matrix sample); // for a single image

        void fwd_prop(matrix input_img);
        void back_prop(int label); // update all the weights

        matrix convolution_backprop(matrix last_input, matrix d_L_d_out);
        matrix max_pool_backprop(matrix last_input, matrix d_L_d_out);
        matrix softmax_backprop();

        double cross_entropy_loss();
};


CNN::CNN(int fltr_sz, int max_pool_sz, int n_fltrs, int strd, int num_nodes, double learning_rate, mt19937 gen){
    filter_size = fltr_sz;
    max_pool_size = max_pool_sz;
    n_filters = n_fltrs;
    stride = strd;
    n_nodes = num_nodes;
    lr = learning_rate;
    num_classes = 10;
    gen = gen;

    normal = normal_distribution<double>(0, 1); // (mean, std)
    init_filters();
    init_biases();
    init_weights();
}

double CNN::cross_entropy_loss(){
    vector<int> label_vec(10, 0);
    float loss = 0;
    label_vec[label] = 1;

    for(auto i = 0; i < 10; i++)
        loss -= label_vec[i]*log(out[i]);
    return loss;
}

void CNN::init_filters(){
    filters = vector<matrix>(n_filters, matrix(filter_size, vector<double>(filter_size)));
    for (int i = 0; i < filters.size(); i++){

        for (int j = 0; j < filter_size; j++){
            for (int k = 0; k < filter_size; k++){
                filters[i][j][k] = normal(gen); // sample from the normal distribution
            }   
        }
    }
}

void CNN::init_biases(){
    bias = vector<double>(num_classes);
    for (int i = 0; i < n_filters; i++){
        bias[i] = normal(gen);
    }
}

void CNN::init_weights(){
    int size = 169;
    weights = matrix(size, vector<double>(num_classes));
    for(int c=0;c<num_classes;c++){
        for(int i=0;i<size;i++){
                weights[i][c] = normal(gen); 
            }
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
    conv_out_flat.clear();
    for(auto && v : tmp){
        conv_out_flat.insert(conv_out_flat.end(), v.begin(), v.end());
    }
}

void CNN::softmax(){
    double tot_sum=0.0;
    vector<double> tmp = softmax_inp;
    for(int i=0; i<softmax_inp.size(); i++){
        tmp[i] = exp(tmp[i]);
        tot_sum += tmp[i];
    }
    out.clear();
    for(int i=0;i<tmp.size(); i++) out.push_back(tmp[i]/tot_sum);
}

void CNN::fully_connected(){
    // Temporary fix to make the conv_out_flat as vector of vectors
    matrix pen = matrix(1, vector<double>(conv_out_flat.size()));
    pen[0] = conv_out_flat;
    matrix tmp = multiply(pen, weights);
    for(int i=0;i<tmp.size();i++){
        vector<double> c = vector_addition(tmp[i], bias);
        tmp[i] = c;
    }
    // Again the whole matrix thingy is converted back to whatever
    softmax_inp = tmp[0];
    softmax();
}

void CNN::fwd_prop(matrix input_img){
    // Convolution layer
    conv_output = convolution(input_img, filters[0]);
    
    relu(conv_output);
    flatten(max_pool(conv_output));

    fully_connected();
}

void CNN::back_prop(int lbl){
    label = lbl;
    matrix d_L_d_out_maxpool = softmax_backprop();
    // matrix d_L_d_out_conv = max_pool_backprop(d_L_d_out_maxpool);
    // convolution_backprop(d_L_d_out_conv);
}

matrix CNN::softmax_backprop(){
    vector<double> d_L_d_out = vector<double>(out.size(), 0);
    vector<double> t_exp = vector<double>(softmax_inp.size());
    vector<double> d_out_d_t = vector<double>(out.size());
    vector<double> d_L_d_t = vector<double>(out.size());
    vector<double> d_L_d_b = vector<double>(out.size());
    vector<double> d_t_d_w = vector<double>(conv_out_flat.size());
    vector<double> d_L_d_inputs_flat = vector<double>(conv_out_flat.size());
    matrix d_L_d_inputs = matrix(conv_output.size(), vector<double>(conv_output[0].size()));
    matrix d_L_d_w = matrix(conv_out_flat.size(), vector<double>(out.size()));
    float d_t_d_b = 1;
    float sum;

    d_L_d_out[label] = -1 / out[label];

    for (auto i = 0; i < softmax_inp.size(); i++)
        t_exp[i] = exp(softmax_inp[i]);
    
    for_each(t_exp.begin(), t_exp.end(), [&] (float n) {
        sum += n;
    });

    for (auto i = 0; i < out.size(); i++)
        d_out_d_t[i] = -t_exp[label] * t_exp[i] / pow(sum, 2);
    d_out_d_t[label] = t_exp[label] * (sum - t_exp[label]) / pow(sum, 2);

    for (auto i = 0; i < conv_out_flat.size(); i++)
        d_t_d_w[i] = conv_out_flat[i];
    
    for (auto i = 0; i < out.size(); i++)
        d_L_d_t[i] = d_L_d_out[label] * d_out_d_t[i];

    // d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
    d_L_d_w = multiply(d_t_d_w, d_L_d_t);
    d_L_d_b = d_L_d_t;
    d_L_d_inputs_flat = multiply(d_L_d_t, weights);

    for (auto i = 0; i < weights.size(); i++)
        for (auto j = 0; j < weights[0].size(); j++)
            weights[i][j] -= lr * d_L_d_w[i][j];
    for (auto i = 0; i < out.size(); i++)
        bias[i] -= lr * d_L_d_b[i];
    
    int idx = 0;
    for (auto i = 0; i < d_L_d_inputs.size(); i++)
        for (auto j = 0; j < d_L_d_inputs[0].size(); j++){
            d_L_d_inputs[i][j] = d_L_d_inputs_flat[idx];
            idx++;
        }
    
    print_matrix(weights);

    return d_L_d_inputs;
}

// last_input is probably conv_output 
// consider dropping the parameter and writing: matrix last_input = conv_output \e
matrix CNN::max_pool_backprop(matrix last_input, matrix d_L_d_out){
    //TODO: ask danny if the "last input" is the "conv input" or something else
    matrix d_L_d_input = matrix(last_input.size(), vector<double>(last_input[0].size())); 

    int s = last_input.size()/max_pool_size;

    int max_val = -1;
    int max_k = 0;
    int max_l = 0;

    matrix out( s, vector<double>(s, 0));
    for (size_t i = 0; i < s; i++)
    {
        for (size_t j = 0; j < s; j++)
        {
            for (size_t k = 0; k < max_pool_size; k++)
            {
                for (size_t l = 0; l < max_pool_size; l++)
                {
                    float val = last_input[i*max_pool_size + k][j*max_pool_size + l];
                    if(max_val < val){
                        max_val = val;
                        max_k = k;
                        max_l = l;
                    }       
                }   
            }
            d_L_d_input[i*max_pool_size + max_k][i*max_pool_size + max_l] = d_L_d_out[i][j];
        }
    }
    return d_L_d_input; 
}

// last_input should be the sample itself no? maybe can drop the parameter \e
matrix CNN::convolution_backprop(matrix last_input, matrix d_L_d_out){
    //assuming filters are only 1
    matrix d_L_d_filters = matrix(filter_size, vector<double>(filter_size)); 

    int output_size =  (1 + last_input.size() - filter_size)/stride;
    // matrix output = matrix(output_size, vector<double>(output_size));

    
    for(int i =0; i<output_size; i++){
        for(int j=0; j<output_size;j++){

            matrix tmp = matrix(filter_size, vector<double>(filter_size));
            
            for(int k=0;k<filter_size;k++){
            vector<double>::const_iterator first = last_input[i+k].begin() + j;
            vector<double>::const_iterator last = last_input[i+k].begin() + j +  filter_size;
            vector<double> newVec(first, last);
            tmp[k] = newVec;
            }


            d_L_d_filters = sum_matrices(d_L_d_filters, multiply_scalar_matrix(d_L_d_out[i][j], tmp));
        }
    }

    filters[0] = sum_matrices(filters[0], multiply_scalar_matrix(-1*lr, d_L_d_filters));
    return filters[0];
}