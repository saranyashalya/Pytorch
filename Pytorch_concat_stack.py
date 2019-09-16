# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:45:27 2019

@author: Saranya
"""

#concatenation is to add elements in same axis of tensor
# stacking is to add elements to new axis of tensor

## Pytorch framework
import torch

t = torch.tensor([1,2,3], dtype = torch.float32)
print(t.shape)

#using unsqueeze to create new axis
t1 = t.unsqueeze(dim=0)
print(t1.shape)

t2 = t.unsqueeze(dim=1)
print(t2.shape)


##concatenation operation

t1 = torch.tensor([1,1,1])
t2 = torch.tensor([2,2,2])
t3 = torch.tensor([3,3,3])

t = torch.cat((t1,t2,t3), dim =0)
print(t.shape)

t = torch.stack((t1,t2,t3), dim =0)
print(t.shape)

t = torch.cat(
        (t1.unsqueeze(dim=0),
         t2.unsqueeze(dim=0),
         t3.unsqueeze(dim=0)
         ), dim=0)

print(t.shape)

# stacking and concatenating on second axis
t = torch.stack((t1,t2,t3), dim = 1)
print(t.shape)

t = torch.cat((t1.unsqueeze(dim=1),
               t2.unsqueeze(dim=1),
               t3.unsqueeze(dim=1))
                , dim =1)
print(t.shape)



## Tensorflow framework

import tensorflow as tf

t1 = tf.constant([1,1,1])
t2 = tf.constant([2,2,2])
t3 = tf.constant([3,3,3])

tf.concat(
        (t1,t2,t3),
        axis =0)

t_stack = tf.stack(
        (t1,t2,t3),
        axis =0)
print(t_stack)

t_concat = tf.concat(
                (tf.expand_dims(t1,0),
                tf.expand_dims(t2,0),
                tf.expand_dims(t3,0)),
                axis = 0)
print(t_concat)

tf.stack((t1,t2,t3), axis = 1)

t_concat = tf.concat(
                (tf.expand_dims(t1,1),
                tf.expand_dims(t2,1),
                tf.expand_dims(t3,1)),
                axis = 1)
print(t_concat)

## Numpy

import numpy as np

t1= np.array([1,1,1])
t2 = np.array([2,2,2])
t3 = np.array([3,3,3])

np.concatenate(
        (t1,t2,t3), 
        axis = 0)

np.stack(
        (t1,t2,t3),
        axis =0)


np.concatenate(
        (np.expand_dims(t1, 0),
        np.expand_dims(t2,0),
        np.expand_dims(t3,0)),
        axis = 0)
        
        
np.stack(
        (t1,t2,t3),
        axis =1)

np.concatenate(
        (np.expand_dims(t1, 1),
        np.expand_dims(t2,1),
        np.expand_dims(t3,1)),
        axis = 1)
