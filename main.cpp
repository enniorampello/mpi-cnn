#include <vector>
#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include "cnn.h"
#include "data-reading/data-reading.h"

#define NUM_PROCESSORS 4 // THIS NUMBER MUST MATCH THE ONE GIVEN IN THE COMMAND LINE
#define NUM_EPOCHS 20
#define NUM_IMAGES 1000

using namespace std;
using matrix = vector<vector<double>>;

void distribute_data(vector<matrix> &images, vector<int> &labels, int rank);

int main(int argc, char *argv[]){
    MPI_Status status;
    MPI_Request request;
    MPI_Datatype mpi_matrix;
    int rc, P, p;
    int filter_sze = 3;
    int max_pool_sze = 2;
    random_device rd;
    mt19937 gen;
    vector<matrix> images;
    vector<int> labels;
    matrix image;
    int label;

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
    CNN model = CNN(filter_sze, max_pool_sze, 1, 1, 16, 0.005, gen);

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