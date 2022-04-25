#include <vector>
#include <iostream>

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
    matrix a(2, vector<double>(2));
    matrix b(1, vector<double>(2));
    matrix c(2, vector<double>(2, 0));

    a[0][0] = 1.0;
    a[0][1] = 0.0;
    a[1][0] = 1.0;
    a[1][1] = 0.0;

    b[0][0] = 2.0;
    b[0][1] = 3.0;

    c = multiply(b, a);

    for (vector<double> i: c){
        for (double j: i){
            cout << j << ' ';
        }
        cout << endl;
    }

    return 0;
}