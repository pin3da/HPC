#! /usr/bin/python
import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
    return fromfile(fname, sep='\n')

x  = m_load('sizes')
y1 = m_load('parallel')
y2 = m_load('serial')


plt.title  ('Matrix Multiplication')
plt.xlabel ('Computational complexity (N * M * O)')
plt.ylabel ('Execution time (seconds)')

plt.plot(x, y1, 'rx', label='parallel code')
plt.plot(x, y2, 'bo', label='serial code')

plt.legend()

plt.hold()
plt.show()
