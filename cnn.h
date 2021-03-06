#include <numeric>
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <random>
#include <math.h>
#include "utils.h"

using namespace std;
using matrix = vector<vector<double>>;

#define EPS 0.0001
#define MAX_GRAD 15
#define max(a,b) ((a)>(b)?(a):(b))

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

        vector<matrix> conv_output; // convolutional block output
        vector<double> conv_out_flat; // flatened output
        vector<double> softmax_inp;
        vector<double> out_prob;

        void init_filters();
        void init_biases();
        void init_weights();

        void load_image(matrix sample);
        
        vector<matrix> convolution(matrix img); // for a single image
        static void relu(matrix &sample);

        void flatten(vector<matrix> tmp); // TODO 
        void fully_connected(); // for the fully connected layer
        void softmax();

        

    public:
        normal_distribution<double> normal;

        vector<matrix> filters;
        vector<double> bias;
        matrix weights;

        CNN(int fltr_sz, int max_pool_sz, int n_fltrs, int strd, int num_nodes, double learning_rate, mt19937 gen);
        void train(matrix sample); // call this for every sample
        
        vector<matrix> max_pool(vector<matrix> sample); // for a single image

        void fwd_prop(matrix input_img);
        void back_prop(int label); // update all the weights

        vector<matrix> convolution_backprop(vector<matrix> d_L_d_out);
        vector<matrix> max_pool_backprop(vector<matrix> d_L_d_out);
        vector<matrix> softmax_backprop();

        double cross_entropy_loss();
        void print_img();
        int check_label(int label); //checking if labels and predicted softmax match
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
    out_prob = vector<double>(10);
}

double CNN::cross_entropy_loss(){
    vector<int> label_vec(10, 0);
    double loss = 0;
    label_vec[label] = 1;

    for(auto i = 0; i < 10; i++)
        loss -= label_vec[i]*log(out_prob[i]+EPS);
    
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
    for (int i = 0; i < num_classes; i++){
        bias[i] = normal(gen);
    }
}

void CNN::init_weights(){
    int size = 169*n_filters;
    weights = matrix(size, vector<double>(num_classes));
    for(int c=0;c<num_classes;c++){
        for(int i=0;i<size;i++){
                weights[i][c] = normal(gen);
            }
    }
}

int CNN::check_label(int label){
    int maxElementIndex = max_element(out_prob.begin(),out_prob.end()) - out_prob.begin();
    // int maxElement = *max_element(out_prob.begin(), out_prob.end());

    if(label == maxElementIndex)
        return 1;
    else
        return 0;
}

vector<matrix> CNN::convolution(matrix img){

    int output_size =  (1 + img.size() - filter_size)/stride;
    // matrix output = matrix(output_size, vector<double>(output_size));

    vector<matrix> output = vector<matrix>(n_filters, matrix(output_size, vector<double>(output_size)));

    for(int filter_no=0; filter_no<n_filters;filter_no++){
        
        for(auto i =0; i<output_size; i++){
        for(auto j=0; j<output_size;j++){
            matrix tmp = matrix(filter_size, vector<double>(filter_size));
            for(int k=0;k<filter_size;k++){
            vector<double>::const_iterator first = img[i+k].begin() + j;
            vector<double>::const_iterator last = img[i+k].begin() + j +  filter_size;
            vector<double> newVec(first, last);
            tmp[k] = newVec;
            }
            output[filter_no][i][j] = matrix_inner_product(filters[filter_no], tmp);           
        }
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

vector<matrix> CNN::max_pool(vector<matrix> sample){
    int output_size = sample[0].size()/max_pool_size;
    vector<matrix> out = vector<matrix>(n_filters, matrix(output_size, vector<double>(output_size, 0.0)));
    for(int filter_num=0;filter_num<n_filters;filter_num++){
        for (size_t i = 0; i < output_size; i++)
            {
                for (size_t j = 0; j < output_size; j++)
                {
                    for (size_t k = 0; k < max_pool_size; k++)
                    {
                        for (size_t l = 0; l < max_pool_size; l++)
                        {
                            float val = sample[filter_num][i*max_pool_size + k][j*max_pool_size + l];
                            out[filter_num][i][j] = max(out[filter_num][i][j], val);      
                        }   
                    }
                }
            }
    }
    return out; 

}

void CNN::flatten(vector<matrix> tmp){
    conv_out_flat.clear();
    for(int filter_num =0;filter_num<n_filters;filter_num++)
        for(auto && v : tmp[filter_num]){
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
    for(int i=0;i<tmp.size(); i++) out_prob[i] = tmp[i]/tot_sum;
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
    image = input_img;
    // Convolution layer
    conv_output = convolution(input_img);
   
    for_each(conv_output.begin(), conv_output.end(), relu);
    // relu(conv_output);
    flatten(max_pool(conv_output));
    fully_connected();
}

void CNN::back_prop(int lbl){
    label = lbl;
    vector<matrix> d_L_d_out_maxpool = softmax_backprop();
    vector<matrix> d_L_d_out_conv = max_pool_backprop(d_L_d_out_maxpool);
    convolution_backprop(d_L_d_out_conv);
}

vector<matrix> CNN::softmax_backprop(){
    // Declaration of variables to store the values 
    vector<double> d_L_d_out = vector<double>(out_prob.size(), 0);
    vector<double> t_exp = vector<double>(softmax_inp.size());
    vector<double> d_out_d_t = vector<double>(out_prob.size());
    vector<double> d_L_d_t = vector<double>(out_prob.size());
    vector<double> d_L_d_b = vector<double>(out_prob.size());
    vector<double> d_t_d_w = vector<double>(conv_out_flat.size());
    vector<double> d_L_d_inputs_flat = vector<double>(conv_out_flat.size());
    vector<matrix> d_L_d_inputs = vector<matrix>(conv_output.size(), matrix(conv_output[0].size(), vector<double>(conv_output[0][0].size())));
    matrix d_L_d_w = matrix(conv_out_flat.size(), vector<double>(out_prob.size()));

    float d_t_d_b = 1;
    float sum;

    d_L_d_out[label] = -1 / out_prob[label];

    for (auto i = 0; i < softmax_inp.size(); i++)
        t_exp[i] = exp(softmax_inp[i]);
    
    for (auto i = 0; i < t_exp.size(); i++)
        sum += t_exp[i];

    for (auto i = 0; i < out_prob.size(); i++)
        d_out_d_t[i] = -t_exp[label] * t_exp[i] / pow(sum, 2);
    d_out_d_t[label] = t_exp[label] * (sum - t_exp[label]) / pow(sum, 2);

    for (auto i = 0; i < conv_out_flat.size(); i++)
        d_t_d_w[i] = conv_out_flat[i];
    
    for (auto i = 0; i < out_prob.size(); i++)
        d_L_d_t[i] = d_L_d_out[label] * d_out_d_t[i];

    // d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
    d_L_d_w = multiply(d_t_d_w, d_L_d_t);
    d_L_d_b = d_L_d_t;    
    d_L_d_inputs_flat = multiply(d_L_d_t, weights);

    for (auto i = 0; i < weights.size(); i++)
        for (auto j = 0; j < weights[0].size(); j++){
            if (d_L_d_w[i][i] > MAX_GRAD)
                d_L_d_w[i][i] = MAX_GRAD;
            else if (d_L_d_w[i][i] < -MAX_GRAD)
                d_L_d_w[i][i] = -MAX_GRAD;
               
            weights[i][j] -= lr * d_L_d_w[i][j];
        }
    for (auto i = 0; i < out_prob.size(); i++){
        if (d_L_d_b[i] > MAX_GRAD)
            d_L_d_b[i] = MAX_GRAD;
        else if (d_L_d_b[i] < -MAX_GRAD)
            d_L_d_b[i] = -MAX_GRAD;
            
        bias[i] -= lr * d_L_d_b[i];
    }
    
    int idx = 0;
    for(int filter_num=0;filter_num<n_filters;filter_num++)
        for (auto i = 0; i < d_L_d_inputs[0].size(); i++)
            for (auto j = 0; j < d_L_d_inputs[0][0].size(); j++){
                if (d_L_d_inputs_flat[idx] < -MAX_GRAD)
                    d_L_d_inputs_flat[idx] = -MAX_GRAD;
                else if (d_L_d_inputs_flat[idx] > MAX_GRAD)
                    d_L_d_inputs_flat[idx] = MAX_GRAD;
                d_L_d_inputs[filter_num][i][j] = d_L_d_inputs_flat[idx];
                idx++;
            }
    
    // print_matrix(weights);

    return d_L_d_inputs;
}

// last_input is probably conv_output 
// consider dropping the parameter and writing: matrix last_input = conv_output \e
vector<matrix> CNN::max_pool_backprop(vector<matrix> d_L_d_out){

    vector<matrix> d_L_d_input = vector<matrix>(n_filters,matrix(conv_output[0].size(), vector<double>(conv_output[0][0].size()))); 

    int s = conv_output[0].size()/max_pool_size;

    int max_val = -1;
    int max_k = 0;
    int max_l = 0;

    matrix out( s, vector<double>(s, 0));
    for(int filter_num = 0;filter_num<n_filters;filter_num++){
    for (size_t i = 0; i < s; i++)
    {
        for (size_t j = 0; j < s; j++)
        {
            for (size_t k = 0; k < max_pool_size; k++)
            {
                for (size_t l = 0; l < max_pool_size; l++)
                {
                    float val = conv_output[filter_num][i*max_pool_size + k][j*max_pool_size + l];
                    if(max_val < val){
                        max_val = val;
                        max_k = k;
                        max_l = l;
                    }       
                }   
            }
            d_L_d_input[filter_num][i*max_pool_size + max_k][i*max_pool_size + max_l] = d_L_d_out[filter_num][i][j];
            if (d_L_d_input[filter_num][i*max_pool_size + max_k][i*max_pool_size + max_l] > MAX_GRAD)
                d_L_d_input[filter_num][i*max_pool_size + max_k][i*max_pool_size + max_l] = MAX_GRAD;
            else if (d_L_d_input[filter_num][i*max_pool_size + max_k][i*max_pool_size + max_l] < -MAX_GRAD)
                d_L_d_input[filter_num][i*max_pool_size + max_k][i*max_pool_size + max_l] = -MAX_GRAD;
            
        }
    }
    }
    return d_L_d_input; 
}

// last_input should be the sample itself no? maybe can drop the parameter \e
vector<matrix> CNN::convolution_backprop(vector<matrix> d_L_d_out){
    vector<matrix> d_L_d_filters = vector<matrix>(n_filters,matrix(filter_size, vector<double>(filter_size))); 

    int output_size =  (1 + image.size() - filter_size)/stride;
    // matrix output = matrix(output_size, vector<double>(output_size));

    for(int filter_num=0;filter_num<n_filters;filter_num++){
        for(int i =0; i<output_size; i++){
        for(int j=0; j<output_size;j++){

            matrix tmp = matrix(filter_size, vector<double>(filter_size));
            
            for(int k=0;k<filter_size;k++){
            vector<double>::const_iterator first = image[i+k].begin() + j;
            vector<double>::const_iterator last = image[i+k].begin() + j +  filter_size;
            vector<double> newVec(first, last);
            tmp[k] = newVec;
            }

            d_L_d_filters[filter_num] = sum_matrices(d_L_d_filters[filter_num], multiply_scalar_matrix(d_L_d_out[filter_num][i][j], tmp));
            
        }
    }
    filters[filter_num] = sum_matrices(filters[filter_num], multiply_scalar_matrix(-1*lr, d_L_d_filters[filter_num]));
    }
    for (auto n = 0; n < filters.size(); n++){
        for(auto i = 0; i < filters[0].size(); i++){
            for(auto j = 0; j < filters[0].size(); j++){
                if (filters[n][i][j] > MAX_GRAD)
                    filters[n][i][j] = MAX_GRAD;
                else if (filters[n][i][j] < -MAX_GRAD)
                    filters[n][i][j] = -MAX_GRAD;
            }
        }
    }
    return filters;
}

void CNN::print_img(){
    print_matrix(image);
}