#include <vector>
#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include "cnn.h"
#include "data-reading/data-reading.h"

#define NUM_PROCESSORS 2 // THIS NUMBER MUST MATCH THE ONE GIVEN IN THE COMMAND LINE
#define NUM_EPOCHS 20
#define NUM_IMAGES 1000
#define NUM_FILTERS 2
#define NUM_CLASSES 10
#define FILTER_SIZE 3
#define CONV_MAT_SIZE 169

using namespace std;
using matrix = vector<vector<double>>;

void distribute_data(vector<matrix> &images, vector<int> &labels, int rank);
void average_weights(double *weights, int weight_size, int rank);

void flatten_vector(vector<double> &vec, double *flat_vec, int dim1);
void unflatten_vector(vector<double> &vec, double *flat_vec, int dim1);

void flatten_matrix(matrix &mat, double *flat_mat, int dim1, int dim2);
void unflatten_matrix(matrix &mat, double *flat_mat, int dim1, int dim2);

void flatten_vector_of_matrices(vector<matrix> &vec, double *flat_vec, int dim1, int dim2, int dim3);
void unflatten_vector_of_matrices(vector<matrix> &vec, double *flat_vec, int dim1, int dim2, int dim3);

int main(int argc, char *argv[]){
    MPI_Status status;
    MPI_Request request;
    MPI_Datatype mpi_matrix;
    int rc, P, p;
    int max_pool_sze = 2;
    random_device rd;
    mt19937 gen;
    vector<matrix> images;
    vector<int> labels;
    matrix image;
    int label;

    int filters_size = NUM_FILTERS*FILTER_SIZE*FILTER_SIZE;
    int weights_size = CONV_MAT_SIZE*NUM_FILTERS*NUM_CLASSES;
    int biases_size = NUM_CLASSES;

    double *filters = (double *) malloc(filters_size*sizeof(double));
    double *weights = (double *) malloc(weights_size*sizeof(double));
    double *biases = (double *) malloc(biases_size*sizeof(double));

    rc = MPI_Init(&argc, &argv);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &p);
    rc = MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (p == 0){
        printf("number of processes: %d\n", P);
        printf("number of images: %d\n", NUM_IMAGES);
        read_mnist_data("data-reading/train-images.idx3-ubyte", images, NUM_IMAGES);
        read_mnist_labels("data-reading/train-labels.idx1-ubyte", labels, NUM_IMAGES);
    }
    distribute_data(images, labels, p);

    printf("\tI am process %d\n", p);

    gen = mt19937(rd());
    CNN model = CNN(FILTER_SIZE, max_pool_sze, NUM_FILTERS, 1, 16, 0.02, gen);

    for (auto epoch = 0; epoch < NUM_EPOCHS; epoch++){
        double loss = 0;
        double acc = 0;
        cout<<"running epoch "<<epoch+1<<endl;
        for (auto i = 0; i < images.size(); i++){
            image = images[i];
            label = labels[i];

            model.fwd_prop(image);
            model.back_prop(label);

            loss += model.cross_entropy_loss();
            // cout<<model.cross_entropy_loss()<<" ";
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
        cout<<"Process: "<<p<<", Epoch: "<<epoch+1<<", Loss: "<<loss<<", Accuracy: "<<acc<<endl;
        
        flatten_vector_of_matrices(model.filters, filters, NUM_FILTERS, FILTER_SIZE, FILTER_SIZE);
        flatten_matrix(model.weights, weights, CONV_MAT_SIZE*NUM_FILTERS, NUM_CLASSES);
        flatten_vector(model.bias, biases, NUM_CLASSES);

        average_weights(filters, filters_size, p);
        average_weights(weights, weights_size, p);
        average_weights(biases, biases_size, p);

        unflatten_vector_of_matrices(model.filters, filters, NUM_FILTERS, FILTER_SIZE, FILTER_SIZE);
        unflatten_matrix(model.weights, weights, CONV_MAT_SIZE*NUM_FILTERS, NUM_CLASSES);
        unflatten_vector(model.bias, biases, NUM_CLASSES);
    }
    
    MPI_Finalize();
    return 0;
}

void distribute_data(vector<matrix> &images, vector<int> &labels, int rank){
    double send_images[NUM_IMAGES*28*28], recv_images[NUM_IMAGES/NUM_PROCESSORS*28*28];
    int send_labels[NUM_IMAGES], recv_labels[NUM_IMAGES/NUM_PROCESSORS];

    if (rank == 0){
        int idx = 0;
        for (auto n = 0; n < NUM_IMAGES; n++){
            for (auto i = 0; i < 28; i++)
                for (auto j = 0; j < 28; j++){
                    send_images[idx] = images[n][i][j];
                    idx++;
                }
            send_labels[n] = labels[n];
        }
    }
    MPI_Scatter(
        send_images, 
        NUM_IMAGES/NUM_PROCESSORS*28*28, 
        MPI_DOUBLE, 
        recv_images, 
        NUM_IMAGES/NUM_PROCESSORS*28*28,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );
    MPI_Scatter(
        send_labels, 
        NUM_IMAGES/NUM_PROCESSORS, 
        MPI_INT, 
        recv_labels, 
        NUM_IMAGES/NUM_PROCESSORS,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );
    
    images.clear();
    labels.clear();

    images = vector<matrix>(NUM_IMAGES/NUM_PROCESSORS, matrix(28, vector<double>(28)));
    labels = vector<int>(NUM_IMAGES/NUM_PROCESSORS);

    int idx = 0;
    for (auto n = 0; n < NUM_IMAGES/NUM_PROCESSORS; n++){
        for (auto i = 0; i < 28; i++)
            for (auto j = 0; j < 28; j++){
                // if (recv_images[idx] < 0.0001) recv_images[idx] = 0;
                images[n][i][j] = recv_images[idx];
                idx++;
            }
        labels[n] = recv_labels[n];
    }
}

void average_weights(double *weights, int weight_size, int rank){
    // int weight_size = CONV_MAT_SIZE*NUM_FILTERS*NUM_CLASSES;
    double all_weights[NUM_PROCESSORS*weight_size];
    double weight_avg[weight_size];

    MPI_Gather(weights, weight_size, MPI_DOUBLE, 
               all_weights, weight_size, MPI_DOUBLE, 
               0, MPI_COMM_WORLD);
    
    if (rank == 0){
        for (auto i = 0; i < weight_size; i++)
            weight_avg[i] = 0;

        auto idx = 0;
        for (auto p = 0; p < NUM_PROCESSORS; p++){
            for (auto i = 0; i < weight_size; i++){
                weight_avg[i] += all_weights[p*weight_size+i];
            }
        }
        for (auto i = 0; i < weight_size; i++)
            weight_avg[i] /= NUM_PROCESSORS;
    }

    MPI_Scatter(weight_avg, weight_size, MPI_DOUBLE,
                weights, weight_size, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
}

void flatten_vector(vector<double> &vec, double *flat_vec, int dim1){
    for (auto i = 0; i < dim1; i++)
        flat_vec[i] = vec[i];
}

void unflatten_vector(vector<double> &vec, double *flat_vec, int dim1){
    for (auto i = 0; i < dim1; i++)
        vec[i] = flat_vec[i];
}

void flatten_matrix(matrix &mat, double *flat_mat, int dim1, int dim2){
    int idx = 0;
    for (auto i = 0; i < dim1; i++)
        for (auto j = 0; j < dim2; j++){
            flat_mat[idx] = mat[i][j];
            idx++;
        }
}

void unflatten_matrix(matrix &mat, double *flat_mat, int dim1, int dim2){
    int idx = 0;
    for (auto i = 0; i < dim1; i++)
        for (auto j = 0; j < dim2; j++){
            mat[i][j] = flat_mat[idx];
            idx++;
        }
}

void flatten_vector_of_matrices(vector<matrix> &vec, double *flat_vec, int dim1, int dim2, int dim3){
    int idx = 0;
    for (auto n = 0; n < dim1; n++)
        for (auto i = 0; i < dim2; i++)
            for (auto j = 0; j < dim3; j++){
                flat_vec[idx] = vec[n][i][j];
                idx++;
            }
}

void unflatten_vector_of_matrices(vector<matrix> &vec, double *flat_vec, int dim1, int dim2, int dim3){
    int idx = 0;
    for (auto n = 0; n < dim1; n++)
        for (auto i = 0; i < dim2; i++)
            for (auto j = 0; j < dim3; j++){
                vec[n][i][j] = flat_vec[idx];
                idx++;
            }
}