### Matrix multiplication

Basic parallel matrix multiplication using CUDA.

Directories description:
- src : Matrix multiplication code
- test : Matrix generator

A simple plotter is also provided, you can made uggly graphics like this

![Matrix image](https://raw.githubusercontent.com/pin3da/HPC/master/matrix_mult/matrix_mul.png)

The blue line is the time obtained from the gpu, the green one is the time obtained from cpu.


The following image shows the time for small values.

![Matrix image](https://raw.githubusercontent.com/pin3da/HPC/master/matrix_mult/small_sizes.png)

### Compile and running

create a build directory

    mkdir build
    cd build

create cmake stuff

    cmake ..
    
compile & run

    make
    generator > input
    mult < input > data
    
Enjoy data and performace test :D

### Notes
There is an error with cmake, compile by yourself if is possible.
inside biuld/ perform:

    /usr/local/cuda-6.5/bin/nvcc ../src/mult.cu -o mult
