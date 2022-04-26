#include <vector>
#include <iostream>
#include "cnn.h"

using namespace std;
using matrix = vector<vector<double>>;

template <class T>
std::vector <std::vector<T> > multiply(std::vector <std::vector<T> > &a, std::vector <std::vector<T> > &b)
{
    const int n = a.size();     // a rows
    const int m = a[0].size();  // a cols
    const int p = b[0].size();  // b cols
    
    std::vector <std::vector<T> > c(n, std::vector<T>(p, 0));
    for (auto j = 0; j < p; ++j){
        for (auto k = 0; k < m; ++k){
            for (auto i = 0; i < n; ++i){
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c; 
}

int main(){
    int filter_sze = 2;
    int max_pool_sze = 2;
    CNN model = CNN(filter_sze, max_pool_sze, 1, 1, 0, 0.0);
    matrix test = matrix(4, vector<double>(4)); 
    matrix filter = matrix(filter_sze, vector<double>(filter_sze));
    
    model.softmax_backprop(vector<double>(4, 0), 2);
    
    return 0;
}