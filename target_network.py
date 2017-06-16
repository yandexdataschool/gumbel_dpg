# Target network for lasagne
# Shamelessly stolen from https://github.com/yandexdataschool/AgentNet/blob/master/agentnet/utils/clone.py


"""
Implements the target network techniques in deep reinforcement learning.
In short, the idea is to estimate reference Qvalues not from the current agent state, but
from an earlier snapshot of weights. This is done to decorrelate target and predicted Qvalues/state_values
and increase stability of learning algorithm.

Some notable alterations of this technique:
- Standard approach with older NN snapshot
-- https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

- Moving average of weights
-- http://arxiv.org/abs/1509.02971

- Double Q-learning and other clever ways of training with target network
-- http://arxiv.org/pdf/1509.06461.pdf

Here we implement a generic TargetNetwork class that supports both standard and moving
average approaches through "moving_average_alpha" parameter of "load_weights".

"""

import lasagne
import theano.tensor as T
import theano
from collections import OrderedDict

class TargetNetwork(object):
    """
    A generic class for target network techniques.
    Works by creating a deep copy of the original network and synchronizing weights through
    "load_weights" method.

    If you just want to duplicate lasagne layers with or without sharing params, use agentnet.utils.clone.clone_network

    :param original_network_outputs: original network outputs to be cloned for target network
    :type original_network_outputs: lasagne.layers.Layer or a list/tuple of such
    :param bottom_layers: the layers that should be shared between networks.
    :type bottom_layers: lasagne.layers.Layer or a list/tuple/dict of such.
    :param share_inputs: if True, all InputLayers will still be shared even if not mentioned in bottom_layers
    :type share_inputs: bool


    :snippet:

    #build network from lasagne.layers
    l_in = InputLayer([None,10])
    l_d0 = DenseLayer(l_in,20)
    l_d1 = DenseLayer(l_d0,30)
    l_d2 = DenseLayer(l_d1,40)
    other_l_d2 = DenseLayer(l_d1,41)

    # TargetNetwork that copies all the layers BUT FOR l_in
    full_clone = TargetNetwork([l_d2,other_l_d2])
    clone_d2, clone_other_d2 = full_clone.output_layers

    # only copy l_d2 and l_d1, keep l_d0 and l_in from original network, do not clone other_l_d2
    partial_clone = TargetNetwork(l_d2,bottom_layers=(l_d0))
    clone_d2 = partial_clone.output_layers

    do_something_with_l_d2_weights()

    #synchronize parameters with original network
    partial_clone.load_weights()

    #OR set clone_params = 0.33*original_params + (1-0.33)*previous_clone_params
    partial_clone.load_weights(0.33)

    """
    def __init__(self,original_network_outputs,bottom_layers=(),share_inputs=True,name="target_net."):
        self.output_layers = clone_network(original_network_outputs,
                                           bottom_layers,
                                           share_inputs=share_inputs,
                                           name_prefix=name)
        self.original_network_outputs = original_network_outputs
        self.bottom_layers = bottom_layers
        self.name = name

        #get all weights that are not shared between networks
        all_clone_params = lasagne.layers.get_all_params(self.output_layers)
        all_original_params = lasagne.layers.get_all_params(self.original_network_outputs)

        #a dictionary {clone param -> original param}
        self.param_dict = OrderedDict({clone_param : original_param
                           for clone_param, original_param in zip(all_clone_params,all_original_params)
                           if clone_param != original_param})

        if len(self.param_dict) ==0:
            raise ValueError("Target network has no loadable. "
                             "Either it consists of non-trainable layers or you messed something up "
                             "(e.g. hand-crafted layers with no automatic params)."
                             "In case you simply want to clone network, use agentnet.utils.clone.clone_network")

        self.load_weights_hard = theano.function([],updates=self.param_dict)

        self.alpha = alpha = T.scalar('moving average alpha',dtype=theano.config.floatX)
        self.param_updates_with_alpha = OrderedDict({ clone_param:  (1-alpha)*clone_param + (alpha)*original_param
                                                     for clone_param,original_param in self.param_dict.items()
                                                    })
        self.load_weights_moving_average = theano.function([alpha],updates=self.param_updates_with_alpha)



    def load_weights(self,moving_average_alpha=1):
        """
        Loads the weights from original network into target network. Should usually be called whenever
        you want to synchronize the target network with the one you train.

        When using moving average approach, one should specify which fraction of new weights is loaded through
        moving_average_alpha param (e.g. moving_average_alpha=0.1)

        :param moving_average_alpha: If 1, just loads the new weights.
            Otherwise target_weights = alpha*original_weights + (1-alpha)*target_weights
        """
        assert 0<=moving_average_alpha<=1

        if moving_average_alpha == 1:
            self.load_weights_hard()
        else:
            self.load_weights_moving_average(moving_average_alpha)




"""
Utility functions that can clone lasagne network layers in a custom way.
[Will be] used for:
- target networks, e.g. older copies of NN used for reference Qvalues.
- DPG-like methods where critic has to process both optimal and actual actions

"""
import lasagne
from copy import deepcopy
from warnings import warn

def clone_network(original_network, bottom_layers=None,
                  share_params=False, share_inputs=True,name_prefix = None):
    """
    Creates a copy of lasagne network layer(s) provided as original_network.

    If bottom_layers is a list of layers or a single layer, function won't
    copy these layers, using existing ones instead.

    Else, if bottom_layers is a dictionary of {existing_layer:new_layer},
    each time original network would have used existing_layer, cloned network uses new_layer

    It is possible to either use existing weights or clone them via share_weights flag.
    If weights are shared, target_network will always have same weights as original one.
    Any changes (e.g. loading or training) will affect both original and cloned network.
    This is useful if you want both networks to train together (i.e. you have same network applied twice)
    One example of such case is Deep DPG algorithm: http://arxiv.org/abs/1509.02971

    Otherwise, if weights are NOT shared, the cloned network will begin with same weights as
    the original one at the moment it was cloned, but than the two networks will be completely independent.
    This is useful if you want cloned network to deviate from original. One example is when you
    need a "target network" for your deep RL agent, that stores older weights snapshot.
    The DQN that uses this trick can be found here: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf


    :param original_network: A network to be cloned (all output layers)
    :type original_network: lasagne.layers.Layer or list/tuple/dict/any_iterable of such.
        If list, layers must be VALUES, not keys.
    :param bottom_layers: the layers which you don't want to clone. See description above.
        This parameter can also contain ARBITRARY objects within the original_network that you want to share.
    :type bottom_layers: lasagne.layers.Layer or a list/tuple/dict of such.
    :param share_params: if True, cloned network will use same shared variables for weights.
        Otherwise new shared variables will be created and set to original NN values.
        WARNING! shared weights must be accessible via lasagne.layers.get_all_params with no flags
        If you want custom other parameters to be shared, use bottom_layers
    :param share_inputs: if True, all InputLayers will still be shared even if not mentioned in bottom_layers
    :type share_inputs: bool
    :param name_prefix: if not None, adds this prefix to all the layers and params of the cloned network
    :type name_prefix: string or None
    :return: a clone of original_network (whether layer, list, dict, tuple or whatever
    """

    if isinstance(original_network,dict):
        original_layers = check_ordered_dict(original_network).values()
    else:#original_layers is a layer or a list/tuple of such
        original_layers = check_list(original_network)

    #filling memo: a dictionary of {id -> stored_item} items that should NOT be copied, but instead reused.
    memo = {}

    if bottom_layers is None:
        #in this case, copy the entire network
        pass
    if isinstance(bottom_layers,dict):
        #make a substitude: each time copier meets original layer, it replaces that layer with custom replacement
        for original,replacement in bottom_layers.items():
            memo[id(original)] = replacement

    else: #case type(bottom_layers) in [lasagne.layers.Layer,list,tuple]
        #make sure inputs are kept the same
        bottom_layers = check_list(bottom_layers)
        for layer in bottom_layers:
            memo[id(layer)] = layer

    #add shared weights
    if share_params:
        all_weights = lasagne.layers.get_all_params(original_layers)
        for weight_var in all_weights:
            #if weight already in memo
            if id(weight_var) in memo:
                #variable is shared if replacement id matches memo key id. Otherwise it's "replaced"
                existing_item = memo[id(weight_var)]
                status = "shared" if id(existing_item) == id(weight_var) else "replaced with {}".format(existing_item)
                warn("Param {} was already {} manually. Default sharing because of share_params was redundant.".format(
                    weight_var, status
                ))
            else:
                #no collisions in memo. Simply add new unit
                memo[id(weight_var)] = weight_var

    #add shared InputLayers
    if share_inputs:
        all_layers = lasagne.layers.get_all_layers(original_layers)
        input_layers = filter(lambda l: isinstance(l,lasagne.layers.InputLayer), all_layers)

        for l_inp in input_layers:
            # if layer already in memo
            if id(l_inp) in memo:
                # layer is shared if replacement id matches memo key id. Otherwise it's "replaced"
                existing_item = memo[id(l_inp)]
                status = "shared" if id(existing_item) == id(l_inp) else "replaced with {}".format(existing_item)
                warn("Layer {} was already {} manually. Default sharing because of share_inputs was redundant.".format(
                    l_inp, status))
            else:
                # no collisions in memo. Simply add new unit
                memo[id(l_inp)] = l_inp

    network_clone = deepcopy(original_network,memo=memo)

    #substitute names, if asked
    if name_prefix is not None:
        #get list of clone output layers
        if isinstance(network_clone, dict):
            clone_layers = check_ordered_dict(network_clone).values()
        else:  # original_layers is a layer or a list/tuple of such
            clone_layers = check_list(network_clone)

        #substitute layer names
        all_original_layers = set(lasagne.layers.get_all_layers(original_layers))
        all_clone_layers = lasagne.layers.get_all_layers(clone_layers)

        for layer in all_clone_layers:
            if layer not in all_original_layers:
                #substitute cloned layer name
                layer.name = name_prefix + (layer.name or '')
            #otherwise it's a shared layer

        #substitute param names
        all_original_params = set(lasagne.layers.get_all_params(original_layers))
        all_clone_params = lasagne.layers.get_all_params(clone_layers)

        for param in all_clone_params:
            if param not in all_original_params:
                # substitute cloned param name
                param.name = name_prefix + (param.name or '')
            # otherwise it's a shared param


    return network_clone


def reapply(layer_or_layers, new_bottom, share_params=True, name_prefix=None):
    """
    Applies a part of lasagne network to a new place. Wraps clone_network
    :param layer_or_layers: layers to be re-applied
    :param new_bottom: a dict {old_layer:new_layer} that defines which layers should be substituted by which other layers
    :param share_params: if True, cloned network will use same shared variables for weights.
        Otherwise new shared variables will be created and set to original NN values.
        WARNING! shared weights must be accessible via lasagne.layers.get_all_params with no flags
        If you want custom other parameters to be shared, agentnet.utils.clone_network
    :param name_prefix: if not None, adds this prefix to all the layers and params of the cloned network

    :return: a new layer or layers that represent re-applying layer_or_layers to new_bottom
    """
    assert isinstance(new_bottom,dict)
    for layer in lasagne.layers.get_all_layers(layer_or_layers,list(new_bottom.keys())):
        if isinstance(layer,lasagne.layers.InputLayer):
            assert layer in new_bottom, "must explicitly provide all new_bottom for each branch of original network. " \
                                        "Assert caused by {}. For dirty hacks, use clone_network.".format(layer.name or layer)
    return clone_network(layer_or_layers, new_bottom, share_params=share_params, share_inputs=False, name_prefix=name_prefix)


## utils/format

from collections import OrderedDict
from warnings import warn

import lasagne
import numpy as np


def is_layer(var):
    """checks if var is lasagne layer"""
    return isinstance(var, lasagne.layers.Layer)


def is_theano_object(var):
    """checks if var is a theano input, transformation, constant or shared variable"""
    return type(var).__module__.startswith("theano")


def is_numpy_object(var):
    """checks if var is a theano input, transformation, constant or shared variable"""
    return type(var).__module__.startswith("numpy")


supported_sequences = (tuple, list)


def check_sequence(variables):
    """
    Ensure that variables is one of supported_sequences or converts to one.
    If naive conversion fails, throws an error.
    """
    if any(isinstance(variables, seq) for seq in supported_sequences):
        return variables
    else:
        # If it is a numpy or theano array, excluding numpy array of objects, return a list with single element
        # Yes, i know it's messy. Better options are welcome for pull requests :)
        if (is_theano_object(variables) or is_numpy_object(variables)) and variables.dtype != np.object:
            return [variables]
        elif hasattr(variables, '__iter__'):
            # Elif it is a different kind of sequence try casting to tuple. If cannot, treat that it will be treated
            # as an atomic object.
            try:
                tupled_variables = tuple(variables)
                message = """{variables} of type {var_type} will be treated as a sequence of {len_casted} elements,
                not a single element.
                If you want otherwise, please pass it as a single-element list/tuple.
                """
                warn(message.format(variables=variables, var_type=type(variables), len_casted=len(tupled_variables)))
                return tupled_variables
            except:
                message = """
                {variables} of type {var_type} will be treated as a single input/output tensor,
                and not a collection of such.
                If you want otherwise, please cast it to list/tuple.
                """
                warn(message.format(variables=variables, var_type=type(variables)))
                return [variables]
        else:
            # otherwise it's a one-element list
            return [variables]


def check_list(variables):
    """Ensure that variables is a list or converts to one.
    If naive conversion fails, throws an error
    :param variables: sequence expected
    """
    return list(check_sequence(variables))


def check_tuple(variables):
    """Ensure that variables is a list or converts to one.
    If naive conversion fails, throws an error
    :param variables: sequence expected
    """
    return tuple(check_sequence(variables))


def check_ordered_dict(variables):
    """Ensure that variables is an OrderedDict
    :param variables: dictionary expected
    """
    assert isinstance(variables, dict)
    try:
        return OrderedDict(list(variables.items()))
    except:
        raise ValueError("Could not convert {variables} to an ordered dictionary".format(variables=variables))


def unpack_list(array, parts_lengths):
    """
    Returns slices of the input list a.
    unpack_list(a, [2,3,5]) -> a[:2], a[2:2+3], a[2+3:2+3+5]

    :param array: array-like or tensor variable
    :param parts_lengths: lengths of subparts

    """
    borders = np.concatenate([[0], np.cumsum(parts_lengths)])

    groups = []
    for low, high in zip(borders[:-1], borders[1:]):
        groups.append(array[low:high])

    return groups
