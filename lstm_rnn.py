#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import logistic
#from __future__ import print_function # and flying cars and jetpacks please.

# Make output easier on the eyes.
np.set_printoptions(precision = 2, suppress=True)

# Logging level; an integer : [0,1,2,3,...]
# The higher the global logging level, the more messages will be logged.
# Setting the global logging level, below, to n
# will log every message assigned a logging level _lower_ than n. 
# Messages are assigned a logging level using function logging(), below.
logging_level = 3

def logging(message,level):
    """ Log message, iff level is less than the global logging_level. 
        message: a string, the message to log. This can be a Python format string
                 such as: "My string %s" % with formattting.
        level: an integer, the logging level assigned to message.
        Think of global logging levels as if the following mapping was in
        effect: {0:silent, 1:debug, 2:verbose}
    """
    if level < logging_level:
        print message


def sigmoid(x):
    """ Activation function for LSTM gate layers
        x: a vector, to pass through the sigmoid, squashing it to the range [0,1]
    """
    # Could experiment with more activation functions: 
    #return np.tanh(x)
    return logistic.cdf(x) 


class LSTM_RNN(object):

    def __init__(self,feed,ht_min_1=[],inputs=1,outputs=1):
        """ 
            Initialise one LSTM memory cell
                                                                              
            feed:    internal state at time-step t-1, fed forward to this unit, 
                     possibly as a recurrent connection.
            ht_min_1 cell activation at time-step t-1, fed forward to this unit.
            inputs:  size of vector of inputs to the cell. 
            outputs: size of output vector
                                                                              
            Implements a single Very-Simple Long-Short-Term Memory cell, 
            as outlined in the following diagram in glorious-ASCII  
            (best viewed with a monospaced font, ideally DejaVu Sans Mono
            otherwise subscripts and other symbols will not display correctly): 
       
                                               ^ hₜ
                                               |
             ,---------------------------------^----.
             |                                 |    |    ςₜ
       ςₜ₋₁  |             (∑)------------,----^----^---->
       ------^--.      ,----^----.        |    |    |  
             |  |      |         |       (τ)   |    |  
             |  |      |         |        |    |    |
             |  |      |         |        ⌄    |    |    hₜ
             |  `---> (∏)       (∏)      (∏)---'----^---->
             |         ^     ,---^---.    ^         |
             |         |     |       |    |         |
             |         |     |       |    |         |
             |        (σ)   (σ)     (τ)  (σ)        |
             |         ^     ^       ^    ^         |
             |       wₓ|   wₘ|     wᵢ|  wᵣ|         |
             |         |     |       |    |         |
             |        [ξ]   [μ]     [ι]  [θ]        |
             |         |     |       |    |         |
             |          `----'---,---'----'         |
        hₜ₋₁ |                   |                  |
        -----^------>------,-----'                  |
             |             |                        |
              `------------^------------------------'
                           | xₜ      
       
       > Where: 
         
         Symbol      Explanation
         ------      -----------
         [.]         Neural layer
         (.)         Layer activation 
         μ           Remember gate
         ξ           Forget gate
         θ           Recall gate
         ςₜ          Cell state at time step t
         ι           Input layer
         x̲ₜ          Cell input vector at time step t
         h̲ₜ          Cell output vector at time step t
         Wξ*         Forget gate weight matrix (etc)
         σ           Nonlinear activation (logistic sigmoid)
         τ           Nonlinear activation (tanh)
         /           Linear activation 
         ∑           Pointwise sum of vectors
         ∏           Pointwise product of vectors
 

         LSTM cells can be strung together like beads on a string, 
         feeding the output and internal state of one to the other
         across timesteps. 
         
              timestep 1       timestep 2       timestep 3

         ςₜ-1 ,---------.  ςₜ  ,---------. ςₜ+1 ,---------.   ςₜ+2
         ---->|         |----->|         |----->|         |--->
              |   C1    |      |   C2    |      |   C3    |
         hₜ-1 |         |  hₜ  |         | hₜ+1 |         |   hₜ+2
         ---->|         |----->|         |----->|         |--->
               `--------'       `--------'       `--------'
              xₜ ^           xₜ+1 ^           xₜ+2 ^
             ,---'          ,-----'          ,-----' 

        """

        ## Initialise bias terms
        self.bi = np.random.randn(outputs) # Input node activation bias
        self.bm = np.random.randn(outputs) # Remember gate activation bias
        self.bf = np.random.randn(outputs) # Forget gate activation bias
        self.br = np.random.randn(outputs) # Recall gate activation bias
        #self.bct = np.random.randn(outputs) # Memory unit activation bias - not used.

        ## Initialise weight matrices to random values
        # All gates are the same size as input vector
        # Their weight matrices can also be the same size.
        self.Wi = np.random.randn(outputs, inputs) # Input node
        self.Wm = np.random.randn(outputs, inputs) # Remember gate
        self.Wf = np.random.randn(outputs, inputs) # Forget gate
        self.Wr = np.random.randn(outputs, inputs) # Recall (output) gate

        # We may have a memory from the previous step
        if feed:
            self.memory = feed
            #self.Wct = [[1.0] * len(feed)] * outputs
        else:
            self.memory = [1.0] * outputs
            #self.Wct = [[1.0] * outputs] * outputs

        # We may have an activation from the previous step
        self.ht_min_1 = ht_min_1
        self.W_ht_min_1 = np.random.randn(outputs)

    def input_node(self, xt):
        if self.ht_min_1 != []:
            # If a previous step's activation was given, include it in the calculation
            # of the current step's activation.
            return np.tanh(np.dot(self.Wi, xt) + np.dot(self.W_ht_min_1, self.ht_min_1) + self.bi)
        else:
            return np.tanh(np.dot(self.Wi, xt) + self.bi)
        #return sigmoid(np.dot(self.Wi, xt))
    
    def remember_gate(self,xt):
        return sigmoid(np.dot(self.Wm, xt) + self.bm)

    def forget_gate(self, xt):
        return sigmoid(np.dot(self.Wf, xt) + self.bf)

    def recall_gate(self, xt):
        return sigmoid(np.dot(self.Wr, xt) + self.br)

    ## Cell activation; @TODO rename this function.
    def state_update(self, xt):
        """ xt a vector; one element of a sequence of inputs.
            
            Compute this cell's activation. 
            
            First, the various gate activations are computed:

            ξₜ = σ(cons(Wξ⋅χₜ, hₜ₋₁⋅Wₕₜ₋₁)) 
            μₜ = σ(cons(Wμ⋅χₜ, hₜ₋₁⋅Wₕₜ₋₁))
            ιₜ = tanh(cons(Wι⋅χₜ, hₜ₋₁⋅Wₕₜ₋₁))
            θₜ = σ(cons(Wθ⋅χₜ, hₜ₋₁⋅Wₕₜ₋₁))

            Then, decisions are made to update the cell's memory layer
            and output its activation. 
            
            ςₜ = (ξ * ςₜ₋₁) + (μₜ * ιₜ)
            hₜ = tanh(ςₜ)

            Some of the steps above are done in one-go in the code below, 
            mostly for compactness.
        """
        logging("\tStimulus:          %s" % xt, 1)
        
        # Decide wheter to form a new memory
        remember_decision = self.input_node(xt) * self.remember_gate(xt)
        logging("\tremember_decision: %s" % remember_decision, 1)

        # Decide whether to keep on to or forget an earlier memory.
        forget_decision = self.forget_gate(xt) * self.memory
        logging("\tself.memory        %s" % self.memory,1)
        logging("\tforget_decision:   %s" % forget_decision,1)

        # Update the current state/ memory
        self.memory = remember_decision + forget_decision
        logging("\tself.memory        %s" % self.memory,1)

        # Decide whether to recall the currently held memory
        recall_decision = self.recall_gate(xt) * np.tanh(self.memory)
        logging("\ttanh(self.memory)  %s" % np.tanh(self.memory),1)
        logging("\trecall_decision:   %s" % recall_decision,1)

        return recall_decision


def error_gradient(prediction, error):
    """ Calculate the gradient of error along connections of a given input
        prediction: a vector; one activation of one LSTM block.
        error: a scalar; a measure of the error of prediction
        returns: a vector; the gradient of error at prediction

        This may be insufficient for the LSTM architecture. For instance, 
        what are we supposed to do with gate activations internally to the cell?
        Do we treat those as hidden layer activations? Do we calculate their error
        separately, similarly to hidden layers in feed-forward networks? 
    """

    return np.dot(2, prediction) * error


def error_value(actual,expected):
    """ Calculate an error value on the latest prediction
        actual is a bit-vector in a one-hot encoding (a single digit is 1)
        expected is a vector of real values in the range [-1,1] each representing the 
        unnormalised probability that the corresponding element of actual is high.
        "Unnormalised" in the sense that we could normalise to the range [0,1]
    """    
    # First convert probabilities to the corresponding bit vector 
    expected = reals_to_bit_vectors(expected)

    return cost_function(mean_squared_error, expected, actual)


def reals_to_bit_vectors(vs):
    """ Translates between a vector in of probabilities and its corresponding
        one-hot encoding bit-vector
    """
    
    bit_vectors = []

    for v in vs:
        bit_vectors.append(reals_to_bit_vector(v))

    return bit_vectors


def reals_to_bit_vector(v):
    """ Translate between a vector of probabilities to a vector of binary digits.
        v is a cell activation vector. We interpret its highest-valued element as 
        the position of the single '1' in the returned bit-vector. The corrolary 
        is that we can never have a vector that is all 0's, but it's possible to get
        a vecto of all 1s (if all values in v are equal).
    """

    # Hack flow to allow training on non-binary data.
    # TODO: make that a bit less hacky.
    if not binary_data:
        return v

    # Get the highest value
    vmax = max(v)

    bit_vector = []
    
    # Fill bit_vector with zeroes, except for the highest value 
    # Note that this scheme does not allow for an all-0 vector
    for el in v:
        if el == vmax:
            bit_vector += [1]
        else:
            bit_vector += [0]
    
    return bit_vector


def cost_function(fun, expected, actual):
    """ Calculate a measure of error between acutual and expected (the prediction)
        fun is the function to use: mean squared error or cross-entropy.
    """
    
    # Ensure both inputs are numpy arrays- we'll use numpy operations in fun
    if type(actual) != np.ndarray:
        actual = np.array(actual)

    if type(expected) != np.ndarray:
        expected = np.array(expected)

    return fun(actual, expected)


def mean_squared_error(expected, actual):
    return ((actual - expected) ** 2).mean()


def cross_entropy(expected, actual):
    return - np.mean(expected * np.log(actual) + (1. - expected) * np.log(1. - actual))



## Training and testing. 

def unfold_for_n_epochs(v, out, s, eta, epochs):
    """ Unfold a network over len(v) steps for a number of epochs 
    """

    # List of epoch errors
    Es = []
    # List of cell activations
    Ys = []

    # Initialise network
    xs = len(v[0])
    cell = LSTM_RNN([], outputs=out,inputs=xs)

    for e in range(0, epochs):
        logging("========================================",2)
        logging("================ Epoch %s ===============" % (e + 1),2)
        logging("========================================",2)
        # Unpacking just to make it explicit that there's two things returned:
        # Errors and activations.
        #Es += [unfold(cell, v, s, eta)]
        E,Y = unfold(cell, v, s, eta)
        Es += [E]
        Ys += [Y]

    logging("All errors: %s" % Es,2)
    
    return {'net':cell, 'data':(Es,Ys)}


def unfold(cell, v, s, eta):
    """ Unfold one LSTM cell over len(v) steps
        v is an input vector of vectors holding training data.
        s is a vector of vectors holding test data.
    """

    logging("Cell weights ======================", 2)
    logging("Wi:    %s" % [str(l) for l in cell.Wi],2) 
    logging("Wm:    %s" % [str(l) for l in cell.Wm],2)
    logging("Wf:    %s" % [str(l) for l in cell.Wf],2)
    logging("Wr:    %s" % [str(l) for l in cell.Wr],2)
    
    logging("Time step 0 ======================",2)
    # Calculate activation at first timestep 
    y = cell.state_update(v[0])
    # List of cell activations; we'll pass it out and plot its mean.
    Y = y

    # e: Time-step error; E: epoch error
    e = error_value(s[0], [y])
    E = e

    logging("Prediction:  %s" % reals_to_bit_vector(y),2)
    logging("Actual:      %s" % s[0],2)
    logging("Step error:  %s" % e,2)
    logging("Epoch error: %s" % E,2)

    # Adjust weights by error gradient
    update_weights(cell, v[0], e, eta) 
    
    for i in range(1,len(v)):
        logging("Time step %s ======================" % i,2)
        # Pass the activation of the last time step to the current one
        cell.ht_min_1 = y
        y = cell.state_update(v[i])
        Y += y

        e = error_value(s[i], [y])
        E += e

        update_weights(cell, v[i], e, eta)
        
        logging("Prediction:  %s" % reals_to_bit_vector(y),2)
        logging("Actual:      %s" % s[i],2)
        logging("Step Error:  %s" % (error_value(s[i], [y]),),2)
        logging("Epoch error: %s" % E,2)

    return (E,Y)


def deep_unfold_for_n_epochs(v, out, s, eta, epochs):
    """ Unfold a deep network with two stacked layers over len(v) steps for a number of epochs 
        TODO: generalise this and unfold_for_n_epochs() to a single function 
        with an option for stacking multiple blocks.
    """

    # List of epoch errors
    Es = []
    # List of cell activations
    Ys = []

    # Initialise network
    xs = len(v[0])
    cell1 = LSTM_RNN([], outputs=out,inputs=xs)
    cell2 = LSTM_RNN([], outputs=out,inputs=xs)

    for e in range(0, epochs):
        logging("========================================",2)
        logging("================ Epoch %s ===============" % (e + 1),2)
        logging("========================================",2)
        # Unpacking just to make it explicit that there's two things returned:
        # Errors and activations.
        #Es += [unfold(cell, v, s, eta)]
        E,Y = deep_unfold(cell1, cell2, v, s, eta)
        Es += [E]
        Ys += [Y]

    logging("All errors: %s" % Es,2)
                                           
    return {'block':(cell1,cell2), 'data':(Es,Ys)}


def deep_unfold(cell1, cell2, v, s, eta):
    """ Unfold a deep LSTM block of two stacked cells over len(v) steps
        v is an input vector of vectors holding training data.
        s is a vector of vectors holding test data.

        TODO: see comments on deep_unfold_for_n_epochs(); this can also be abstracted away 
    """

    for cell in enumerate([cell1, cell2]):
        i, c = cell

        #import pdb
        #pdb.set_trace()

        logging("Cell %s weights ======================" % str(i+1), 2)
        logging("Wi:    %s" % [str(l) for l in c.Wi],2) 
        logging("Wm:    %s" % [str(l) for l in c.Wm],2)
        logging("Wf:    %s" % [str(l) for l in c.Wf],2)
        logging("Wr:    %s" % [str(l) for l in c.Wr],2)
    
    logging("Time step 0, cell 1 ======================",2)
    # Calculate activation for first cell in the block:
    y1 = cell1.state_update(v[0])
    # Propagate activation and memory forwards between layers 
    cell2.ht_min_1 = y1
    cell2.memory = cell1.memory
    # Block activation:
    logging("Time step 0, cell 2 ======================",2)
    y2 = cell2.state_update(v[0])
    # List of block activations. 
    Y = y2

    # Error is calculated on the activation of the block as a whole
    e = error_value(s[0], [y2])
    E = e

    logging("Prediction:  %s" % reals_to_bit_vector(y2),2)
    logging("Actual:      %s" % s[0],2)
    logging("Step error:  %s" % e,2)
    logging("Epoch error: %s" % E,2)

    # Adjust weights by error gradient
    update_weights(cell1, v[0], e, eta) 
    update_weights(cell2, v[0], e, eta) 
    
    for i in range(1,len(v)):
        logging("Time step %s cell 1 ======================" % i,2)
        # Proagate block activation and memory from the last time step to the current one
        cell1.ht_min_1 = y2
        cell1.memory = cell2.memory
        y1 = cell1.state_update(v[i])
        cell2.ht_min_1 = y1
        cell2.memory = cell1.memory
        logging("Time step %s cell 2 ======================" % i,2)
        y2 = cell2.state_update(v[i])
        Y += y2

        # Calculate block error for time step
        e = error_value(s[i], [y2])
        E += e

        update_weights(cell1, v[i], e, eta)
        update_weights(cell2, v[i], e, eta)
        
        logging("Prediction:  %s" % reals_to_bit_vector(y2),2)
        logging("Actual:      %s" % s[i],2)
        logging("Step Error:  %s" % e,2)
        logging("Epoch error: %s" % E,2)

    return (E,Y)


def update_weights(cell, x, error, eta):
    """ Adjust weights and bias values toward the minimum of the error 
        of the cell's activation on the output.
        cell: an instance of class LSTM_RNN().
        x: a vector; one element of a sequence of inputs
        error: a scalar; the error of the activation of cell against x
        eta: pronounced "ee-tah"; a learning rate for gradient descent. 
    """
    
    delta = error_gradient(x, error)

    weights = [cell.Wi, cell.Wm, cell.Wf, cell.Wr]

    # Update weights in place
    # Could also do with a list comprehension, possibly. 
    for WM in weights:
        for W_i in enumerate(WM):
            i, w = W_i
            # Nudge weight closer to gradient bottom: 
            WM[i] = w - eta * delta

    logging("Cell weights ======================",2)
    logging("Wi:    %s" % [str(l) for l in cell.Wi],2) 
    logging("Wm:    %s" % [str(l) for l in cell.Wm],2)
    logging("Wf:    %s" % [str(l) for l in cell.Wf],2)
    logging("Wr:    %s" % [str(l) for l in cell.Wr],2)

    bias_terms = [cell.bi, cell.bm, cell.bf, cell.br]

    for b_i in enumerate(bias_terms):
        i,b = b_i
        bias_terms[i] = b - eta * delta

    cell.bi = bias_terms[0]
    cell.bm = bias_terms[1]
    cell.bf = bias_terms[2]
    cell.br = bias_terms[3]

    logging("Cell bias terms ======================",2)
    logging("bi:    %s" % cell.bi,2) 
    logging("bm:    %s" % cell.bm,2)
    logging("bf:    %s" % cell.bf,2)
    logging("br:    %s" % cell.br,2)
