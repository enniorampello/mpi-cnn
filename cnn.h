#include <vector>
#include <random>
using namespace std;
using matrix = vector<vector<double>>;

class CNN {
    private:
        random_device rd;
        mt19937 gen;
        normal_distribution<double> normal;

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

        matrix convolution(matrix sample, matrix filter); // for a single image
        void relu(matrix &sample);
        matrix max_pool(matrix sample); // for a single image
        void fwd_pass(); // for the fully connected layer
        vector<double> softmax(vector<double> out);

        void fwd_prop();
        void back_prop(); // update all the weights

    public:
        CNN(int fltr_sz, int max_pool_sz, int n_fltrs, int strd, int num_nodes, double learning_rate);
        void train(matrix sample); // call this for every sample
};
