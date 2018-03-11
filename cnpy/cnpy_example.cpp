/// g++ -o cnpy_example cnpy_example.cpp cnpy.cpp -lz 

#include"cnpy.h"
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>

const unsigned int Nx = 16;
const unsigned int Ny = 8;

int main()
{
  //create random data
  double* data = new double[Nx*Ny];
  for(unsigned int i = 0; i < Nx*Ny; i++) data[i] = (double) (rand())/RAND_MAX;
  
  //save it to file
  const unsigned int shape[] = {Ny,Nx};
  cnpy::npy_save("data.npy",data,shape,2,"w");
  
  //cleanup: note that we are responsible for deleting all loaded data
  delete[] data;
 
}
