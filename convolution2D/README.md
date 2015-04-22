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
    [./constant_mem | ./global_mem | ./shared_mem] > times

### Test

After compile the code you can run all the codes several times using the script `run_test`

    cd test
    ./run_test

### Images and filters.

The program is tested with a sobel kernel, you can generate cool images like these:

Original image:
![Original image](https://raw.githubusercontent.com/pin3da/HPC/master/convolution2D/images/cat2.png)

Sobel Filter X:
![Sobel Filter X](https://raw.githubusercontent.com/pin3da/HPC/master/convolution2D/images/sobel_x.png)

Sobel Filter Y:
![Sobel Filter Y](https://raw.githubusercontent.com/pin3da/HPC/master/convolution2D/images/sobel_y.png)


### Performance graphics

Discalimer: I'm learning [d3](http://d3js.org/) while I do these graphs, they have some bugs. The file `test/index.html` can be used to
render the images.

Image 1. Showing all data, parallel code wins :D
![All data](https://raw.githubusercontent.com/pin3da/HPC/master/convolution2D/images/all_data.png)

Image 2. Showing parallel executions. In this case the data was modified to move left the last value (x coordinate).
![parallel data](https://raw.githubusercontent.com/pin3da/HPC/master/convolution2D/images/data.png)

Image 3. Showing parallel executions with small inputs.
![Small values](https://raw.githubusercontent.com/pin3da/HPC/master/convolution2D/images/small_values.png)

### Performance tables

This table contains all the averaged data for different sizes.

[Table](https://github.com/pin3da/HPC/blob/master/convolution2D/test/clean.tsv)

This table shows the improvement of each version compared to the sequential version.

[Speed up](https://github.com/pin3da/HPC/blob/master/convolution2D/test/speed-up.tsv)

This table shows the improvement of each version compared to the previous version.

[Speed up 2](https://github.com/pin3da/HPC/blob/master/convolution2D/test/speed-up2.tsv)

### Performance graphics II (bars).

Image 1. Comparing all implementations with an image of size 168744872
![Small values](https://raw.githubusercontent.com/pin3da/HPC/master/convolution2D/images/bar9.png)

Image 2. Comparing all implementations with an image of size 16084992
![Small values](https://raw.githubusercontent.com/pin3da/HPC/master/convolution2D/images/bar10.png)


### Conclusions tb; dr.

The parallel code is really faster than serial code, even for small sizes. But the improvements of tiled version over constant memory version are not significant in contrast with the codification work.


Special Thanks to [lvandeve](http://lodev.org/) for his great library.

