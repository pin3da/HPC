Parallel Number-theoretic transform
===================================

In this task we are trying to implement a Number-theoretic transform which is a generalization
of a [Fast Fourier Transform](http://en.wikipedia.org/wiki/Fast_Fourier_transform).

[Here](http://www.cs.cmu.edu/afs/cs/academic/class/15750-s01/www/notes/lect0424) you can find a good explanation
of the algorithm.


## The problem.

Suppose that you need to compute the convolution between two polynomials and you are sure the result could not be greater
than (a * b * c).

The first way to attempt to solve this problem could be using a Fast Fourier transform, but this solution has a problem: uses floating point numbers and they are imprecise.

The second way (which we use) is using a Number-Theoretic transform (NTT) to compute the answer mod a, b, c and then using the [CRT](http://en.wikipedia.org/wiki/Chinese_remainder_theorem) reconstruct the answer mod (a * b * c).

## Parallel solution.

In our solution we use [CUDA](http://en.wikipedia.org/wiki/CUDA) in order to compute one NTT (yes, one) in parallel.
As we need to compute several NTT with different modulos  we use [ZMQ](http://zeromq.org/) to connect several
GPU's on a cluster [1].

The architecture is quite simple and is based in the [Parallel pipeline](http://zguide.zeromq.org/page:all#toc14) suggested by ZMQ.

![Ventilator](https://github.com/imatix/zguide/raw/master/images/fig5.png)

In this model, there is a ventilator who is in charge of assignate task to the workers. In this case each task
is just to compute the NTT with a different modulo.

After all those computations, there is a sink who is in charge of merge the results using the CRT.


[1]. This could be understood as several GPU's on the same PC or several GPU's on several PCs.

### References

- [Wikipedia Number-Theoretic transform](http://en.wikipedia.org/wiki/Discrete_Fourier_transform_%28general%29#Number-theoretic_transform)
- [Wolfram Number-Theoretic transform](http://mathworld.wolfram.com/NumberTheoreticTransform.html)
- [NTT explanation](http://www.cs.cmu.edu/afs/cs/academic/class/15750-s01/www/notes/lect0424)
- [Chinese Remainder Theorem](http://en.wikipedia.org/wiki/Chinese_remainder_theorem)


______

Manuel Pineda - Carlos González.
