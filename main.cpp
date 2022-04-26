#include <vector>
#include <iostream>
#include "cnn.h"

using namespace std;
using matrix = vector<vector<double>>;


int main(){
    // matrix a(2, vector<double>(2));
    // matrix b(1, vector<double>(2));
    // matrix c(2, vector<double>(2, 0));

    // a[0][0] = 1.0;
    // a[0][1] = 0.0;
    // a[1][0] = 1.0;
    // a[1][1] = 0.0;

    // b[0][0] = 2.0;
    // b[0][1] = 3.0;

    // c = multiply(b, a);

    // for (vector<double> i: c){
    //     for (double j: i){
    //         cout << j << ' ';
    //     }
    //     cout << endl;
    // }
    int filter_sze = 2;
    int max_pool_sze = 2;
    CNN model = CNN(filter_sze, max_pool_sze, 1, 1, 0, 0.0);
    matrix test = matrix(4, vector<double>(4)); 
    matrix filter = matrix(filter_sze, vector<double>(filter_sze));
    
    test[0][0] = 10;
    test[0][1] = 1;
    test[0][2] = 1;
    test[0][3] = 0.4;

    test[1][0] = 1;
    test[1][1] = 1;
    test[1][2] = 1;
    test[1][3] = 2;

    test[2][0] = 1;
    test[2][1] = 1;
    test[2][2] = 1;
    test[1][3] = 2;

    test[3][0] = 1;
    test[3][1] = 0.5;
    test[3][2] = 2.0;
    test[3][3] = 0.8;

    
    
    // matrix ans = model.convolution(test, filter);
    // print_matrix(ans);
    // model.relu(ans);
    // print_matrix(ans);
    // cout<<endl;
    // print_matrix(test);
    // matrix ans1 = model.max_pool(test);
    // print_matrix(ans1);

    model.fwd_prop(test);

    return 0;
}