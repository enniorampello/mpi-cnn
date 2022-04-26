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

    filter[0][0] = 1;
    filter[0][1] = -1;

    filter[1][0] = -1;
    filter[1][1] = -1;
    
    matrix ans = model.convolution(test, filter);
    print_matrix(ans);
    model.relu(ans);
    print_matrix(ans);
    cout<<endl;
    print_matrix(test);
    matrix ans1 = model.max_pool(test);
    print_matrix(ans1);

    return 0;
}