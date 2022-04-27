#include <vector>
#include <iostream>
#include "cnn.h"

using namespace std;
using matrix = vector<vector<double>>;

int main(){
    int filter_sze = 2;
    int max_pool_sze = 2;
    CNN model = CNN(filter_sze, max_pool_sze, 1, 1, 0, 0.0);
    matrix test = matrix(4, vector<double>(4)); 
    matrix filter = matrix(filter_sze, vector<double>(filter_sze));
    

    
    return 0;
}