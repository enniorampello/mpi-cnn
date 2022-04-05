#include<string>
#include<vector>


using namespace std;
using matrix = vector<vector<double>>;
//declaring functions 
void read_mnist_data(string path, vector<matrix> &vec);

void read_mnist_labels(string path, vector<vector <double>> &labels);