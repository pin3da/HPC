#! /usr/bin/python
import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
    return fromfile(fname, sep='\n')

x  = m_load('sizes')
y1 = m_load('parallel')
y2 = m_load('serial')
plt.plot(x, y1)
plt.plot(x, y2)

plt.hold()
plt.show()
