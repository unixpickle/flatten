# flatten

Extract rectangular shapes from 3D photos.

# Solving the perspective equation

Methods exist to determine projection parameters from corners of a rectangle in a projection, such as [this one](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6960959/). Instead of using these methods, I applied gradient descent. However, pure gradient descent often finds local minima in this space. To work around this, I trained a generative model to produce approximate solutions to the problem, and then finetune these solutions with gradient-based optimization.
