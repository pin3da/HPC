2D convolution.
===================

This code uses the [LodePNG library](http://lodev.org/lodepng/) to read/write the images.
It's very easy to use and has no dependencies :D.

Directories description:
- src    : code and lodepng library.
- images : Input images for tests.

### Compile and running.

create a build directory

    mkdir build
    cd build

create cmake stuff

    cmake ..

compile & run

    make
    task > times

### Images and filters.

The program is tested with a sobel kernel, you can generate cool images like these:

Original image:
![Original image](https://raw.githubusercontent.com/pin3da/HPC/master/convolution2D/images/cat2.png)

Sobel Filter X:
![Sobel Filter X](https://raw.githubusercontent.com/pin3da/HPC/master/convolution2D/images/sobel_x.png)

Sobel Filter Y:
![Sobel Filter Y](https://raw.githubusercontent.com/pin3da/HPC/master/convolution2D/images/sobel_y.png)


<!--### Performance graphics

Coming soon-->

