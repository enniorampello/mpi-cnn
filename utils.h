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

template <class T>
vector <vector<T> > multiply(vector <vector<T> > &a, vector <vector<T> > &b)
{
    const int n = a.size();     // a rows
    const int m = a[0].size();  // a cols
    const int p = b[0].size();  // b cols
    
    vector <vector<T> > c(n, vector<T>(p, 0));
    for (auto j = 0; j < p; ++j){
        for (auto k = 0; k < m; ++k){
            for (auto i = 0; i < n; ++i){
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c; 
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
    for(int i=0; i<c.size();i++){
        c[i] = vector_addition(a[i], b[i]);
    }

    return c;

}