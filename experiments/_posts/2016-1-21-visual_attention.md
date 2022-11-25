---
layout: post
title: Multimodal learning techniques - 2
use_math: true
category: posts
---

## Encoder-Decoder Network

It is a general framework using neural networks for mapping structured inputs to structured outputs. 

What do we mean by structured outputs?

When we are dealing with problems such as machine translation, we need to map input(*source language*) to an output which has its own structure(*translated language*). As described in [1], this task is not only concerned in capturing semantics on source language, but also deals with forming coherent sentence in the translated language. 

### Encoders 

The encoder reads the input data *x* and maps into a representation *c*. 
$$
c = f_{enc}(x)
$$
In case of input as image, we use a CNN as encoder and extract features vector or *cube* from convolutional layers.

### Decoders

Decoder generates output *y* conditioned on context *c* of the input.
$$
p(y \mid x) = f_{dec}(c)
$$
When are descibing image with natural language, we use RNN as a decoder.

As described in [1], RNN's we use for this task are conditional language models (model distribution over sentences given an additional context *c*).

$$
h\_{t} = \phi\_{\theta}(h\_{t-1},x,c) 
$$


### Issue with encoder-decoder framework

Encoder always compresses the input vector into a single fixed dimensional representation vector *c*.

Not all of the images contain the same amount of information, and hence to describe the varying amount of content in the image with a single fixed dimensional vector is not a good design.


## Attention mechanism 

To tackle above discussed issues, [1] introduces a concept of attention mechanism between encoder and decoder.

To incorporate structured representation of input, we want encoder to return a set of vectors describing the spatial or temporal component of input. 
We refer $c$ to as context set, which is composed of fixed size vectors. The number of vectors in each example may vary. 
$$
c = {c\_{1}, c\_{2}, \dots, c\_{M}}
$$

The attention model controls what information is seen by the decoder and hence the pipeline is composed of encoder, attention model and then decoder.

The attention model takes input from hidden state of decoder at previous time step $z\_{t-1}$ and score the context vector $c\_{i}$. This signifies which vector $c\_{i}$ is most relevant to focus on for next timestep.

$e\_{i}^{t}$ signifies the scores and $\alpha\_{i}^{t}$ signifies the attention weights given to each element of context vector $c$. 

Scores $e\_{i}^{t}$ are calculated as: 
$$e\_{i}^{t} = f\_{ATT}(z\_{t-1}, c\_{i}, {\alpha\_{j}^{t-1} })$$

Attention weights $\alpha$ are calculated by applying *softmax* to scores $e$. 


Using attention weights and previous context vector, we can calculate the new context vector. In soft attention model, we calculate the new context vector as:
$$
c^{t} = \sum_{i=1}^{M} \alpha\_{i}c\_{i}
$$


With the new context vector $c^{t}$, we calculate the new state of decoder, which is RNN in image captioning case. $h\_{t} = \phi\_{\theta}(h\_{t-1},x\_{t},c\_{t})$.


This design of pipeline solves the problem of limited information being encoded by the encoder. Now based on decoder's output at each time step, we calculate the weightage given to spatial or temporal part of input. Hence, each vector output by encoder now describes a particular region of the input. Attention mechanism learns to chose which information needs to focussed at a particular time step.


![Attention based model](https://heuritech.files.wordpress.com/2016/01/caption_attention1.png?w=470)

*Figure: Attention based model. This image is copied from blog post by Heuritech [4]*

### Notations for Recurrent neural network

State transition is a function: 
$$
h\_{t}^{l-1}, h\_{t-1}^{l} \rightarrow h\_{t}^{l} 
$$



### References

1. Describing Multimedia Content using Attention-based Encoder--Decoder Networks 
   Link : http://arxiv.org/abs/1507.01053

2. Recurrent Neural Network Regularization

3. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

4. [Heuritech blog](http://blog.heuritech.com/2015/12/01/learning-to-link-images-with-their-descriptions/)

5. Skip-Thought Vectors
