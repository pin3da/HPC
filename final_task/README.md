Parallel Number-theoretic transform
===================================

In this task we are trying to implement a Number-theoretic transform which is a generalization
of a [Fast Fourier Transform](http://en.wikipedia.org/wiki/Fast_Fourier_transform).

[Here](http://www.cs.cmu.edu/afs/cs/academic/class/15750-s01/www/notes/lect0424) you can find a good explanation
of the algorithm.

In general terms, the Fast Fourier Transform has a wide number of applications, particularly in the signal processing field, it is most commonly used as a spectrum analyzer across multiple disciplines (from audio engineering to space engineering), however it has also other important applications in computational science, for example, in data compression, particularly for lossy image and sound formats, and as a replacement for the traditional convolution algorithm. In mathematics, it has been also used to solve partial differential equations. 


## The problem.

Suppose that you need to compute the convolution between two polynomials and you are sure the result could not be greater
than (a * b * c).

The first way to attempt to solve this problem could be using a Fast Fourier transform, but this solution has a problem: uses floating point numbers and they are imprecise.

The second way (which we use) is using a Number-Theoretic transform (NTT) to compute the answer mod a, b, c and then using the [CRT](http://en.wikipedia.org/wiki/Chinese_remainder_theorem) reconstruct the answer mod (a * b * c).

## The algorithm
Here is the basic algorithm used in our implementation of the FFT:

![Algorithm](https://raw.githubusercontent.com/pin3da/HPC/master/final_task/doc/images/FFT_Iter.png) 

The Bit Reverse Copy is a simple algorithm that will help us get the items of the vector on the order we need them according to the following tree:

![BCR](https://raw.githubusercontent.com/pin3da/HPC/master/final_task/doc/images/BCR.png)

## Parallel solution.

In our solution we use [CUDA](http://en.wikipedia.org/wiki/CUDA) in order to compute one NTT (yes, one) in parallel (which graphically would look something like this):

![Parallel](https://raw.githubusercontent.com/pin3da/HPC/master/final_task/doc/images/Parallel.png)

As we need to compute several NTT with different modulos we use [ZMQ](http://zeromq.org/) to connect several
GPU's on a cluster [1].

The architecture is quite simple and is based in the [Parallel pipeline](http://zguide.zeromq.org/page:all#toc14) suggested by ZMQ.

![Ventilator](https://github.com/imatix/zguide/raw/master/images/fig5.png)

In this model, there is a ventilator who is in charge of assignate task to the workers. In this case each task
is just to compute the NTT with a different modulo.

After all those computations, there is a sink who is in charge of merge the results using the CRT.


[1]. This could be understood as several GPU's on the same PC or several GPU's on several PCs.

## The Implementation

We used, as said before CUDA 6 along with ZMQ, along with the CZMQ wrapper.

Right now we have implemented a parallelized version of the FFT, the Convolution using the FFT, and the Chinese Remainder Theorem (CRT for short). The first two are implemented in the component known as "worker" while the last one is implemented in what is known as the "sink", this allows each machine to compute a convolution (e.g. machine one computes convolution mod a, machine two computes concolution mod b, etc.) while the sink computes the CRT, given us the convolution a * b *c. these implementations are somewhat basic and use mostly global memory to access data, a little portion of the code uses constant memory. We also have a Serial version of the whole process for comparison purposes.

## Conclusions

* ZMQ presents itself as a very good alternative for working with parallel models like this one, the architecture itself is pretty simple but efficient, and can be effectively used in conjunction with CUDA without much hassle.
* The FFT has many different implementations, the one implemented here has a complexity of O(n*log(n)), and in our experience it runs much faster on a processor than on a GPU, this can probably be changed making a better use of the GPU (Using shared Memory for example).
* The CRT on the other hand reacts very well to being implemented on a parallel enviroment (CUDA) exposing decent acceleration in comparison with the serial version.
* The past two conclusions lead us to believe the best performance/work ratio implementation may be to use each worker to compute the serial version of the FFT (and convolution), and CUDA on the sink to compute the CRT which can be easily implemented in parallel with a decent performance boost.
* There is a library called cuFFT which provides different FFT implementations that are very well optimized, this could be another resource to use in the future.

### References

- [Wikipedia Number-Theoretic transform](http://en.wikipedia.org/wiki/Discrete_Fourier_transform_%28general%29#Number-theoretic_transform)
- [Wolfram Number-Theoretic transform](http://mathworld.wolfram.com/NumberTheoreticTransform.html)
- [NTT explanation](http://www.cs.cmu.edu/afs/cs/academic/class/15750-s01/www/notes/lect0424)
- [Chinese Remainder Theorem](http://en.wikipedia.org/wiki/Chinese_remainder_theorem)


______

Manuel Pineda - Carlos González.
