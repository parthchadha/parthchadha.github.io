---
layout: post
title: Multimodal learning techniques - 1
use_math: true
category: posts
---
## Multimodal distributed representations

###Goal: 
We want to mathematically model the similarity between image and the sentences that describe the image. We can embed both image and sentences into an embedding space, where this embedding space has the property that vectors that are nearby are visually or semantically related. By the use of joint image-text embedding space, tasks such information retrieval becomes easy. Given a sentence, we can retrieve most relevant images from the dataset, and given an image, we can rank the best describing captions from the given set. 

###Model:
Paper on Visual-Semantic Embeddings [5] describes an encoder-decoder pipeline that learns a joint multimodal joint embedding space with images and text. This model encodes visual information from images and semantic information from captions into an embedding space. It uses pairwise ranking cost function to train the model. 

For both images and captions, we first represent them in their respective lower dimensional representations. Awesome property of these representations is that similar objects or words in this lower dimensional space are closer to each other.

These lower dimensional representation are then passed through respective encoders to map them into $D$ dimensional embedding space.

For images, we use CNN to represent images in a low-dimensional space using feature vectors extracted from last convolutional or fully-connected layers.
For captions, we use LSTM recurrent neural networks, which encode the sentences into $D$ dimensional space. 

Input to LSTM's are vector representation of sentences. Vector representation of sentences are also called word embeddings. These word embeddings can be a learnable parameter during training, else we can use pre-trained models such Word2Vec or GloVe. These pre-trained models are trained over entire Wikipedia and other huge datasets. 

I ran a $K\-means$ clustering algorithm to cluster together words on GloVe pretrained vectors. Below are some of the results of words with similar semantic meaning.

```

[u'tuesday', u'wednesday', u'monday', u'thursday', u'friday', u'sunday', u'saturday', u'late', u'night', u'morning']

[u'school', u'university', u'college', u'students', u'education', u'schools']

[u'music', u'album', u'band', u'song']

[u'game', u'play', u'games', u'played', u'players', u'match', u'player', u'playing']

[u'won', u'win', u'victory', u'gold', u'winning']

```


###Sequence of operation for captions:
Captions are parsed into words, GloVe vector is extracted for each of these word. These word vectors are then fed in sequentially into a RNN. The output from the last state of the RNN (*$D$ dimensional vector*) represents the semantic sense of the sentence.


###Cost function:
Our aim is have low cost for similar visual-semantic pairs or high cost for disimilar pairs.
For measuring similarity we use cosine similarity function (*we have same embedding dimension for image and caption*).

$$
Cost = \sum\_{x} \sum\_{k} max(0,\alpha - s(x,v) + s(x,v\_{k})) + \sum\_{v} \sum\_{k} max(0,\alpha - s(v,x) + s(v,x\_{k})) 
$$

Here , x corresponds to image embedding, v to caption embedding, $x\_{k}$ to contrastive image emedding and $v\_{k}$ to contrastive caption embedding.


In case of image embedding, learnable parameters are only $W\_{emb}$, which map the output of CNN(*4096 dimensional vector in case of VGGnet*) into $D$ dimensional embedding space.


In case of captions, LSTM weights are the only learnable paramters.



### References

1. Describing Multimedia Content using Attention-based Encoder--Decoder Networks 
   Link : http://arxiv.org/abs/1507.01053

2. Recurrent Neural Network Regularization

3. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

4. [Heuritech blog](http://blog.heuritech.com/2015/12/01/learning-to-link-images-with-their-descriptions/)
5. Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models
    Link: 