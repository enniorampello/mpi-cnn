#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using matrix = vector<vector<double>>;


int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


void read_mnist_data(string path, vector<matrix> &vec, int n_images)
{
    cout<<"reading data from "<<path<<endl;
    ifstream file(path, ios::binary);

    if (file.is_open())
    {
        cout<<"reading successful\n";
        int magic_number=0;
        int number_of_images = 0;
        int n_rows=0;
        int n_cols=0;

        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        cout<<number_of_images<<endl;

        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);

        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        cout<<n_rows<<" "<<n_cols<<endl;

        if (n_images == 0)
            n_images = number_of_images;

        for(int i = 0; i < n_images; ++i)
        {
            matrix img(n_rows, vector<double>(n_cols));
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    img[r].push_back(temp / 255.0);

                }
            }
            vec.push_back(img);
        }
    }
    else{
        cout<<"reading failed\n";
    }
}

void read_mnist_labels(string path, vector<int> &labels, int n_labels){
    cout<<"reading labels from "<<path<<endl;
    ifstream file(path, ios::binary);

    if (file.is_open())
    {
        cout<<"reading successful\n";
        int magic_number=0;
        int number_of_labels=0;

        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        file.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);
        cout<<number_of_labels<<endl;

        if (n_labels == 0)
            n_labels = number_of_labels;
        
        for(int i = 0; i < n_labels; ++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp, 1);

            // one hot encoding
            //vector<double> one_hot(10, 0.0);
            //one_hot[temp] = 1.0;
            //labels.push_back(one_hot);

            // no encoding
            labels.push_back(temp);
        }
    }
    else{
        cout<<"reading failed\n";
    }
}

// int main(){

//     //----------testing mnist data reading-----------
//     // vector<matrix> vec;
//     // read_mnist_data("train-images.idx3-ubyte", vec);
//     // for (int i = 0; i < vec[0].size(); i++){
//     //     for (int j = 0; j < vec[0][i].size(); j++){
//     //         cout << vec[0][i][j] << " ";
//     //     }
//     //     cout<<endl;
//     // }

//     //----------testing mnist label reading-----------
//     // vector<vector<double>> labels;
//     // read_mnist_labels("train-labels.idx1-ubyte", labels);
//     // for (int i = 0; i < labels.size(); i++){
//     //     for (int j = 0; j < labels[i].size(); j++){
//     //         cout << labels[i][j] << " ";
//     //     }
//     //     cout<<endl;
//     // }

//     return 0;

// }