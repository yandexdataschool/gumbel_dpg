# -*- coding: utf-8 -*- 
"""
a bunch of lasagne code implementing gumbel softmax & sigmoid
https://arxiv.org/abs/1611.01144
"""
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
from lasagne.layers import Layer

class GumbelSoftmax:
    """
    A gumbel-softmax nonlinearity with gumbel(0,1) noize
    In short, it's a quasi-one-hot nonlinearity that "samples" from softmax 
    categorical distribution.
    
    Explaination and motivation: https://arxiv.org/abs/1611.01144
    Code mostly follows http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    
    Softmax normalizes over the LAST axis (works exactly as T.nnet.softmax for 2d).
    
    :param t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic.
    :param eps: a small number used for numerical stability
    :returns: a callable that can (and should) be used as a nonlinearity
    
    """
    def __init__(self,
                 t=0.1,
                 eps=1e-20):
        assert t != 0
        self.temperature=t
        self.eps=eps
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
         
    def __call__(self,logits):
        """computes a gumbel softmax sample"""
                
        #sample from Gumbel(0, 1)
        uniform = self._srng.uniform(logits.shape,low=0,high=1)
        gumbel = -T.log(-T.log(uniform + self.eps) + self.eps)
        
        #draw a sample from the Gumbel-Softmax distribution
        return T.nnet.softmax((logits + gumbel) / self.temperature)

def onehot_argmax(logits):
    """computes a hard one-hot vector encoding maximum"""
    return T.extra_ops.to_one_hot(T.argmax(logits,-1),logits.shape[-1])

class GumbelSoftmaxLayer(Layer):
    """
    lasagne.layers.GumbelSoftmaxLayer(incoming,**kwargs)
    A layer that just applies a GumbelSoftmax nonlinearity.
    In short, it's a quasi-one-hot nonlinearity that "samples" from softmax 
    categorical distribution.
    
    If you provide "hard_max=True" in lasagne.layers.get_output
    it will instead compute one-hot of aт argmax.
    
    Softmax normalizes over the LAST axis (works exactly as T.nnet.softmax for 2d).
    
    Explaination and motivation: https://arxiv.org/abs/1611.01144
    Code mostly follows http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    
    
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic (e.g. shared)
    eps: a small number used for numerical stability
    """
    def __init__(self, incoming, t=0.1, eps=1e-20, **kwargs):
        super(GumbelSoftmaxLayer, self).__init__(incoming, **kwargs)
        self.gumbel_softmax = GumbelSoftmax(t=t,eps=eps)

    def get_output_for(self, input, hard_max=False, **kwargs):
        if hard_max:
            return onehot_argmax(input)
        else:
            return self.gumbel_softmax(input)

class GumbelSigmoid:
    """
    A gumbel-sigmoid nonlinearity with gumbel(0,1) noize
    In short, it's a function that mimics #[a>0] indicator where a is the logit
    
    Explaination and motivation: https://arxiv.org/abs/1611.01144
    
    Math:
    Sigmoid is a softmax of two logits: a and 0
    e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)
    
    Gumbel-sigmoid is a gumbel-softmax for same logits:
    gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(gumbel2/t)]
    where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
    gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
    gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)
    
    For computation reasons:
    gumbel1-gumbel2 = -log(-log(uniform1(0,1)) +log(-log(uniform2(0,1)) = -log( log(uniform2(0,1)) / log(uniform1(0,1)) )
    gumbel_sigm(a) = sigm([a-log(log(uniform2(0,1))/log(uniform1(0,1))]/t)
    
    
    :param t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic.
    :param eps: a small number used for numerical stability
    :returns: a callable that can (and should) be used as a nonlinearity
    
    """
    def __init__(self,
                 t=0.1,
                 eps=1e-20):
        assert t != 0
        self.temperature=t
        self.eps=eps
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
         
    def __call__(self,logits):
        """computes a gumbel softmax sample"""
                
        #sample from Gumbel(0, 1)
        uniform1 = self._srng.uniform(logits.shape,low=0,high=1)
        uniform2 = self._srng.uniform(logits.shape,low=0,high=1)
        
        noise = -T.log(T.log(uniform2 + self.eps)/T.log(uniform1 + self.eps) +self.eps)
        
        #draw a sample from the Gumbel-Sigmoid distribution
        return T.nnet.sigmoid((logits + noise) / self.temperature)

def hard_sigm(logits):
    """computes a hard indicator function. Not differentiable"""
    return T.switch(T.gt(logits,0),1,0)

class GumbelSigmoidLayer(Layer):
    """
    lasagne.layers.GumbelSigmoidLayer(incoming,**kwargs)
    A layer that just applies a GumbelSigmoid nonlinearity.
    In short, it's a function that mimics #[a>0] indicator where a is the logit
    
    Explaination and motivation: https://arxiv.org/abs/1611.01144
    
    Math:
    Sigmoid is a softmax of two logits: a and 0
    e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)
    
    Gumbel-sigmoid is a gumbel-softmax for same logits:
    gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(gumbel2/t)]
    where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
    gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
    gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)
    
    For computation reasons:
    gumbel1-gumbel2 = -log(-log(uniform1(0,1)) +log(-log(uniform2(0,1)) = -log( log(uniform2(0,1)) / log(uniform1(0,1)) )
    gumbel_sigm(a) = sigm([a-log(log(uniform2(0,1))/log(uniform1(0,1))]/t)
    
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic (e.g. shared)
    eps: a small number used for numerical stability

    """
    def __init__(self, incoming, t=0.1, eps=1e-20, **kwargs):
        super(GumbelSigmoidLayer, self).__init__(incoming, **kwargs)
        self.gumbel_sigm = GumbelSigmoid(t=t,eps=eps)

    def get_output_for(self, input, hard_max=False, **kwargs):
        if hard_max:
            return hard_sigm(input)
        else:
            return self.gumbel_sigm(input)
