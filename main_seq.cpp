#include <vector>
#include <iostream>
#include "cnn.h"
#include "data-reading/data-reading.h"

#define NUM_EPOCHS 20
using namespace std;
using matrix = vector<vector<double>>;


int main(){
    int filter_sze = 3;
    int max_pool_sze = 2;
    
    random_device rd;
    mt19937 gen;

    gen = mt19937(rd());
    
    CNN model = CNN(filter_sze, max_pool_sze, 2, 1, 16, 0.01, gen);
    matrix test = matrix(4, vector<double>(4)); 
    matrix filter = matrix(filter_sze, vector<double>(filter_sze));

    vector<matrix> images;
    vector<int> labels;

    matrix image;
    int label;

    read_mnist_data("data-reading/train-images.idx3-ubyte", images, 500);
    read_mnist_labels("data-reading/train-labels.idx1-ubyte", labels, 500);

    for (auto epoch = 0; epoch < NUM_EPOCHS; epoch++){
        double loss = 0;
        double acc = 0;
        cout<<"running epoch "<<epoch+1<<endl;
        for (auto i = 0; i < images.size(); i++){
            image = images[i];
            label = labels[i];
            // print_matrix(subtract_matrices(images[i], images[i+1]));
            
            model.fwd_prop(image);            
            model.back_prop(label);

            loss += model.cross_entropy_loss();
        }   
        loss /= images.size();

        // computing accuracy
        for (auto i = 0; i < images.size();i++){
            image = images[i];
            label = labels[i];

            model.fwd_prop(image);
            int tmp = model.check_label(label);
            acc += tmp;
        }
        acc /= images.size();
        cout<<"Epoch: "<<epoch+1<<", Loss: "<<loss<<", Accuracy: "<<acc<<endl;
    }
    return 0;
}