from morb import base, units, parameters, stats, param_updaters, trainers, monitors

# This example shows how the FIOTRBM model from "Facial Expression Transfer with
# Input-Output Temporal Restricted Boltzmann Machines" by Zeiler et al. (NIPS 
# 2011) can be recreated in Morb.

rbm = base.RBM()
rbm.v = units.GaussianUnits(rbm) # output (visibles)
rbm.h = units.BinaryUnits(rbm) # latent (hiddens)
rbm.s = units.Units(rbm) # input (context)
rbm.vp = units.Units(rbm) # output history (context)

initial_A = ...
initial_B = ...
initial_bv = ...
initial_bh = ...
initial_Wv = ...
initial_Wh = ...
initial_Ws = ...

parameters.FixedBiasParameters(rbm, rbm.v.precision_units) # add precision term to the energy function
rbm.A = parameters.ProdParameters(rbm, [rbm.vp, rbm.v], initial_A) # weights from past output to current output
rbm.B = parameters.ProdParameters(rbm, [rbm.vp, rbm.h], initial_B) # weights from past output to hiddens
rbm.bv = parameters.BiasParameters(rbm, rbm.v, initial_bv) # visible bias
rbm.bh = parameters.BiasParameters(rbm, rbm.h, initial_bh) # hidden bias
rbm.W = parameters.ThirdOrderFactoredParameters(rbm, [rbm.v, rbm.h, rbm.s], [initial_Wv, initial_Wh, initial_Ws]) # factored third order weights
