#include <vector>
#include <iostream>
#include <random>
#include "cnn.h"
#include "data-reading/data-reading.h"

using namespace std;
using matrix = vector<vector<double>>;


int main(){
    int filter_sze = 3;
    int max_pool_sze = 2;

    random_device rd;
    mt19937 gen;

    gen = mt19937(rd());
    
    CNN model = CNN(filter_sze, max_pool_sze, 1, 1, 16, 0.001, gen);
    matrix test = matrix(4, vector<double>(4)); 
    matrix filter = matrix(filter_sze, vector<double>(filter_sze));

    vector<matrix> images;
    vector<int> labels(10);

    matrix image;
    int label;

    read_mnist_data("data-reading/train-images.idx3-ubyte", images, 10);
    read_mnist_labels("data-reading/train-labels.idx1-ubyte", labels, 10);

    for (auto i = 0; i < 10; i++){
        image = images[i];
        label = labels[i];

        model.fwd_prop(image);
        model.back_prop(label);
    }

    return 0;
}