#include<iostream>
#include<vector>

using namespace std;
using matrix = vector<vector<double>>;

void print_matrix(matrix sample){
    for (int i = 0; i < sample.size(); i++){
        for (int j = 0; j < sample[i].size(); j++){
            cout << sample[i][j] << " ";
        }
        cout<<endl;
    }
}

void print_vector(vector<double> sample){
    for (int i = 0; i < sample.size(); i++){
            cout << sample[i] <<endl;
    }
}