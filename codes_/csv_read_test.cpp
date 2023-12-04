#include <string>
#include <fstream>
#include <vector>
#include <utility> 
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
#include <unistd.h>
#include <iostream>

using namespace std;


template <typename T> static void PrintVector( vector<T>&  ar, ostream& outfile) 
{
    for (typename vector<T>::iterator it = ar.begin(); it != ar.end(); it++)
    {
        outfile << *it << ' ';
    }
    outfile << std::endl;
}

vector<vector<double>> read_csv(string filename){

    vector<vector<double>> result;
    vector<double> temp;

    // Create an input filestream
    std::ifstream myFile(filename);

    // Helper vars
    string line;
    string cell;
    double val;


    // Read data, line by line
    while(getline(myFile, line))
    {
        stringstream lineStream(line);
        while(getline(lineStream,cell, ','))
        {
            stringstream ss(cell);
            ss>>val;
            temp.push_back(val);

        }
        result.push_back(temp);
        temp.clear();      
    }

    // Close file
    myFile.close();

    return result;
}



int main()
{

    vector<vector<double>> labs = read_csv("MNIST_centroid_labels");


    cout<<labs.size()<<endl;

    for (int i=0; i < labs.size(); i++)
    {
        cout<<labs[i].size()<<endl;
    }


    return 0;
}