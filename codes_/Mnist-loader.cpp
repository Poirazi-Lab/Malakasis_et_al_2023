#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <string>


using namespace std;


int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, vector<vector<double> > &vec)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i)
        {
            vector<double> tp;
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back((double)temp);
                }
            }
            vec.push_back(tp);
        }
    }
}




void read_Mnist_Label(string filename, vector<double> &vec)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec[i]= (double)temp;
        }
    }
}


void print_vec(std::vector<double> const &input)
{
    for (int i = 0; i < input.size(); i++) {
        std::cout << input[i] << ' ';
    }
}
 
void vis_mnist(std::vector<double> const &input)
{
    for (int i = 0; i < input.size(); i++) {
        if (input.at(i)==0)
        {
            cout << "." << ' ';
        }
        else 
        {
            cout << "*" << ' ';
        }
        if (i%28==0)
        {
            cout<<endl;
        }
    }
}
 


int main()
{

//***********************TRAIN************************

    //read MNIST train images
    string train_img = "./MNIST/train-images-idx3-ubyte";
    int number_of_images = 60000;
    //int image_size = 28 * 28;


    
    vector<vector<double> > train_imgvec;
    read_Mnist(train_img, train_imgvec);


    //read MNIST train labels into double vector
    string train_labels = "./MNIST/train-labels-idx1-ubyte";

    vector<double> train_labelsvec(number_of_images);
    read_Mnist_Label(train_labels, train_labelsvec);



//***********************TEST************************


    //read MNIST test images
    string test_img = "./MNIST/t10k-images-idx3-ubyte";
    number_of_images = 10000;


    vector<vector<double> > test_imgvec;
    read_Mnist(test_img, test_imgvec);

    //read MNIST test labels into double vector
    string test_labels = "./MNIST/t10k-labels-idx1-ubyte";

    vector<double> test_labelsvec(number_of_images);
    read_Mnist_Label(test_labels, test_labelsvec);


    //Loading Verifications
    cout<<train_imgvec.size()<<endl;
    cout<<train_imgvec[0].size()<<endl;
    cout<<train_labelsvec[0]<<" "<<train_labelsvec[1]<<endl;
    cout<<test_imgvec.size()<<endl;
    cout<<test_imgvec[0].size()<<endl;
    cout<<test_labelsvec[0]<<" "<<test_labelsvec[1]<<endl;
    print_vec(train_imgvec[0]);
    vis_mnist(train_imgvec[0]);
    return 0;
}