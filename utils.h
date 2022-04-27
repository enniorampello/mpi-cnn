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
            cout << sample[i] <<" ";
    }
    cout<<endl;
}

matrix multiply(matrix &a, matrix &b)
{
    const int n = a.size();     // a rows
    const int m = a[0].size();  // a cols
    const int p = b[0].size();  // b cols
    matrix c(n, vector<double>(p, 0));

    for (auto j = 0; j < p; ++j){
        for (auto k = 0; k < m; ++k){
            for (auto i = 0; i < n; ++i){
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c; 
}

vector<double> multiply(vector<double> &a, matrix &b)
{
    vector<double> c;
    const int n = a.size();     // a length
    const int m = b.size();  // b rows
    const int p = b[0].size();  // b cols
    
    if (n == m){
        c = vector<double>(p, 0);
        for (auto i = 0; i < p; i++)
            for (auto j = 0; j < n; j++)
                c[i] += a[j]*b[i][j];
    }
    else if (n == p){
        c = vector<double>(m, 0);
        for (auto i = 0; i < m; i++)
            for (auto j = 0; j < p; j++)
                c[i] += a[j]*b[i][j];
    }
    else 
        cout << "Matrix multiplication error: sizes do not match.";

    return c; 
}

// the result will be a matrix with dimensions (a.size(), b.size()), so the order of the arguments MATTERS!
// if a.size() == b.size() and you want a scalar as output, just use std::inner_product
matrix multiply(vector<double> &a, vector<double> &b){
    int n = a.size();
    int m = b.size();
    matrix c(n, vector<double>(m, 0));
    
    for (auto i = 0; i < n; i++)
        for (auto j = 0; j < m; j++)
            c[i][j] = a[i]*b[j];
    return c;
}

// the purpose of the flag here is to be able to overload the function to return a double
double multiply(matrix a, matrix b, int flag){
    double result = 0.0;
    for(int i=0; i<a.size();i++){
        result += inner_product(a[i].begin(), a[i].end(), b[i].begin(), 0);   
    }
    return result;

}

template <class T>
 vector<T> vector_addition(vector<T> &a, vector<T>&b)
 {   
     vector<T> c;
     transform(a.begin(), a.end(), b.begin(), back_inserter(c), plus<double>());
     return c;
 }

 template <class T>
 vector<vector<T>> matrix_addition(vector <vector<T> > &a, vector <vector<T> > &b)
 {
     vector<vector<T>> c(a.size(), vector<T>(a[0].size(), 0));
     for(int i=0; i<c.size();i++)
         c[i] = vector_addition(a[i], b[i]);
    return c;
}