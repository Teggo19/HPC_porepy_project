# Coupling FEniCS and PorePy using preCICE
### Stefano Galati, Yuhe Zhang, Trygve Tegnakfsjhvgoaesgof
### March 2026

## Code organization
The directories `free-flow-participant` and `porous-media-participant` will contain the solver code. If we manage to build an `Adapter` class, we may create a separate directory or put it in the porous media one. I propose to keep for now any further code like the ones made by me and Yuhe in `examples`.

## Workflow proposal
My idea is to first write a very row code that allow us to test the coupling, and then refactor the code moving the coupling logic into an Adaptor class.
- on the FEniCS side, this means simply writing the NS code and the coupling logic. We already have an adaptor, therefore we have already a nice API ready-to-use. I am working on this now, but in order to debug it I need to test it by coupling it with another code. I think that doing this means also understanding how to use (and what it means to set all the parameters in) the config file. It is a bit confusionary the time-step logic used in all the tutorials of preCICE, since we have a steady-state problem. I will try to test it with another free-flow solver, just to test that the coupling works.
- the same has to me done on the porous media side. This time, we have two choices: either use the precice functionalities (application case) or build our API into an adaptor. I would suggest starting with the first one and then possibly moving to the second with a refactoring. On the [step-to-step online guide](https://precice.org/couple-your-code-preparing-your-solver.html), there are pseudo-codes that give an idea of the workflow. It is presented in the transient case, and I am trying to understand what changes for us. Still, I think it is useful to always have a look at it while inserting the precice calls into the porous media solver.
- Currently, for my case I am taking inspiration from the `flow-over-a-heated-plate` tutorial, specifically `solid-fenics`. It is useful for two reasons:
    - it is a rather simple example showing the workflow presented in the step-by-step guide
    - gives an idea of the functions we should implement for our adaptor (essentally all the precice calls like `initialize()`, `finalize()`, `write_data()`, `is_coupling_ongoing()`, etc. etc.)
- When my steady-state code is ready, then it will become a primary reference code for the coupling logic and substitute the plate tutorial, but in the meantime you can also use that as a reference.

## TODO
What seems necessary at this stage is to do the same work that I am doing on the free-flow side but on the porous media participant, writing pseudo-code where the adaptor or precice is needed. Since testing the coupled code will require some time, it is be important that we all try to understand the coupling logic for our case.
Then we can try inserting some direct precice calls in porepy, and at the same time start to build the adaptor.
What we can surely do in parallel (among us three) is to write the methods of the Adaptor.


## Think about it...
1. What happens if I write an object from a certain FunctionSpace and I read on another FunctionSPace? They must be equal ore precice internally handles the interpolation?
    - RMK: preCICE only cares about the values at points and performs the suitable interpolation as defined in the configuration file!

## Resources
https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx