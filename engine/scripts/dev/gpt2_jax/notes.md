
"""
layers:

ok so we have the input obviously of some amount of batches, B
im not sure if the next dimension is sequence or memory length, probably L
after that, the input has to be of dimension V, since its just been tokenized
so input_BLV -> embedded_input_BLD is the first transformation
this would be done by wte linear layer, i think with no bias
after wte, we positionally embed, so we would use wpe to go from 
embedded_BLD -> positional_BLD

now, positional_BLD goes through 12 blocks of transformer
let's call each blocks' input 
block_input_BLD
we want to have a residual stream, so we save block_input_BLD
the residual stream means that we can get more stable training, because learning to do 'add 0' is easier than learning the identity function
each layer only has to 'contribute' some new information, rather than spending compute reconstructing the input
its kind of like playing telephone. having to reconstruct the initial signal every time might get lost
but if we could write down what the first person said, pass it along as a slip of paper, and everyone else could just add their sticky note to the paper, it would be easier to communicate

the two layers we really care about in the block are the attention layer and the MLP layer
token sequences are, obviously, sequential. so we need something that can process multiple vectors of input
TODO: CONNECT BETTER
if we put this straight into the MLP processing layer, it essentially becomes a token lookup table
to illustrate this, notice that the only thing that can be learned about a given token is what the next token to output is
TODO: THIS PART WAS ALSO WEAKLY MOTIVATED
what's the cleanest way to get the most recent token to have information about what came before it? 
when we process a sentence 'my dog ate the ...', we choose the next word to say based on all previous words
so we have to get the syntactical and semantic contextual information compressed into the word 'the'
TODO: CONNECT BETTER
one way you might think to do this is to compress the information of 'my' into 'dog'
then, 'dog', with the compressed info of 'my' can get to 'ate'
notice that this is a compounding/multiplicative effect
which is subject to information loss, and more worryingly explosion/vanishing of the information from a new datapoint
since the gradient update of the new datapoint now has to flow through all of these tokens
a very small gradient might disappear into 0, as small numbers compound to even smaller 
what we really do, as humans, is more of like a lookup across all previous tokens in our working memory, at each position
this sidesteps that explosion/vanishing problem
when we think about the next word to say, we're taking into account the fact that we're referring to 'my' dog, we're thinking about tokens related to dogs, related to eating, and therefore related to 'dogs eating', and we know that its an item described as singular due to 'the'
so we need a mechanism that can do this kind of comparison 
for a given token, we want to know how important the previous tokens are, across some amount of useful axes
'the' might be a useful token in a syntactical way, since it restricts the space of tokens we might want to generate into singular nouns
'dog' and 'ate' might also be useful in a semantic meaning dimension, since they constrict the space of what kind of singular noun to generate, one that is something a dog may eat
if we had even more information in the sentence, 'my' could have been useful
were it to have been mentioned earlier in the sequence 'i bought a treat', the token 'treat' might be much more important, since now a relation to 'i/my' has been established
in any case, we first get these positional attention scores for how related a given word is to another
we can model this as sort of a fuzzy hashmap lookup, how 'similar/related' is 'the' to all of the previous tokens, across those dimensions of syntax/semantics/etc - where 'the' is our query, and the keys are the previous tokens
two vectors can be called 'similar' based on the angle between them
if we take the dot product of two vectors, and divide by their magnitude, we get the cosine of the angle between them
we usually decide to not divide by the magnitude, because the magnitude of the vector is useful information we want the model to be able to learn along
instead, we correct the scale of the dot product by dividing by the square root of the dimension of the key/query vectors
this is because as dimensionality increases, the dot products between the vectors might get really big, because the variance of each vector's magnitude in a given dimension is high
since these are raw values and don't correspond to 'what % of attention should this value get', we use softmax to turn the vector of raw values into one that has all non negative values, and they add up to 1, a probability distribution

NOTE ON SOFTMAX. SKIPPABLE
softmax itself comes from some interesting information theory, literally the element of surprise. 
we can think of the problem as taking our raw vector from before, x, with elements x_i, and turning it into a new vector p, where sum(p_i) = 1 and no p_i < 0. 
since we want to maximally conserve information, over the space of vectors p that fulfill the above conditions, we want to choose the vector with the highest entropy
the most 'random' or 'noisy' vector is the one that is least likely to have any other biases that have come with it, that are not x
so we also want the new vector p to be the one in the space of all such possible vectors P to have max(entropy(p)).
we can define entropy rigorously as the expected 'surprise'. 
this is pretty intuitive. suppose i flip a coin and i get heads. i shouldn't be 'surprised' because its one of two outcomes. 
but if i have 100 blue balls and 2 red balls, and i draw a red one, i should be 'more surprised'
we might want to reach for the inverse of the probability of the event, 1/p, but this kind of breaks at 0 so we throw a log transform at it and call it a day
since we want the expected surprise, we take the surprise of each event, and multiply it by the probability of the event
so we have sum(p_i * log(1/p_i))
log(1/x) = -log(x)
so sum(p_i * -log(p_i))
and we have -(sum(p_i) * log(p_i)) is the comparator between two different vectors p that might both be probability distributions of f
if we don't somehow account for the original vector f, we end up with a uniform distribution since that would have maximum entropy
suppose f = [1,2,3], we know that we need our p vector to follow this trend where the first element is smaller than the second which is smaller than the third
we can think of f_i as representing the energy of a given state. and p_i is the probablity of being in some state i
so we can ground our entropy maximization function in the constraint that sum(p_i * f_i) must stay equivalent to some constant
think of it like this: we want to find an arrangement of a rectangle that has a fixed perimeter, but maximizes area
we don't necessarily care what the maximum area is, but we can still use this constraint to arrive at a formalization/function of the area
let a, b be the sides of the rectangle, and let A = ab, and perimeter P = 2a+2b
so we want to maximize A with respect to the constraint that 2a + 2b = P
from this constraint, we find that b = P/2-a
so we can rewrite A as a function of just a, where A = (P/2-a)a
or A = aP/2 - a^2
taking the partial derivative with respect to a, we find
dA/da = P/2-2a
0 = P/2-2a
2a = P/2
a = P/4
if a = P/4, and b = P/2-a, then b = P/4 as well
so even though we didn't care what the perimeter P was, it became a useful constraint in order to solve our problem with respect to the terms we have
in the same way, solving our optimization problem
maximize entropy -sum(p_i * log(p_i))
p_i >= 0
sum(p_i) = 1
sum(p_i * f_i)) = beta
i won't do the math here but we get 
p_i = exp(beta * f_i)/sum(exp(beta * f_j)) 
so we might have 
index 0 of the p vector of the original f [1, 2, 3] is
e^1/(e^1+e^2+e^3)
assuming beta = 1
note that beta can be expressed as 1/T, where T akin to a 'temperature' parameter
the idea of 'temperature' here basically implying that higher temperature leads to higher entropy in the derived version of the softmax here, since a lower beta constant entropy would get us closer to a uniform p vector distribution

END SOFTMAX TANGENT

so now, once we get how related they are, from this probability distribution, those tokens return some 'value' that represents, in the attention space, what it means, and we weight that value by our relation score
TODO: what is a value? you didn't explain it
so our new contextualized representation of 'the' is what's outputted by this attention layer

THIS MAY ALSO BE A TANGENT
note that during training, we actually process all of the tokens in the sequence with each other at once, rather than one at a time like when we want to generate the next token
so all we have to do during training is automatically force the query/key similarity values to be super negative for any tokens that wouldn't have existed
the first token can only pay attention to itself
then the second token pays attention to itself and the previous
and so on
this is essentially a lower triangular matrix of 0s, with the upper triangle being the maximally negative float of your ML framework
we do it this way because this is the best way to tell the softmax to output a zero for that position in the vector
THIS MAY ALSO BE A TANGENT

TODO: this section on MLPs is a bit scattered
now, the MLP essentially just learns some useful way of processing that information, and adding to the residual stream
TODO: handwavy. wdym 'add to residual stream' how is it added? literal addition? transformation? 
MLPs are very powerful. 
a very small one can learn to recognize handwritten digits at very low resolutions (like 28x28), by literally flattening out the images pixels, assigning the brightness score of each pixel 0-1, and then outputting a guess distribution over the digits 0-9.
we can think of the problem as essentially learning some mapping from the input vector space, above the space of dimension 784, the flat representation of the pixels, into the space of 10, the vector space of digits
more simply, suppose we need to learn some function y = f(x) from a bunch of datapoints
we can tune a linear function y = Wx + b to get a line of best fit over the datapoints
imagine this in 2 dimensions, mapping from the x dimension to the y dimension. 
we will never have something other than a straight line
so if our f(x) is x^2, our learning is capped to some error bound that we can never surpass
in order to learn it more closely, we need to bend the line somehow
we do this by applying a nonlinear function over the linear transformation, something like
y = nonlinear_fn(Wx+b), which can warp and bend the initial line into something that can hug x^2 better
a classic nonlinear function is y = max(0,x), as it introduces a 'kink' at 0 where it goes from a slope of 1 in the positive domain to a slope of 0 in the negative domainnow, if we stack this, we go from
y = nonlinear_fn(Wx+b)
to 
y = nonlinear_fn(W_1(W_0x + b_0) + b_1)
so now we get to have two bends in our line! 
as we stack up nonlinear layers, we basically get to warp space in more ways
bringing this back to the more abstract vector spaces, what we usually do is have a layer that goes from the input vector space dimension into a higher dimension, and then back down into our output dimension
think of the hidden dimension of the model as representing some amount of features or information in each dimension
the model likely has to learn more things than the amount of dimensions it has, so each dimension has multiple meanings
the MLP layer allows the model to have more 'room' to express features, and then come back down to the layer of the residual
we might think of it as the initial piece of paper everyone is passing through the residual stream as being in a caveman or primitive english
and then the person adding information can think in more sophisticated english/images/visualize, and then translate what they were able to learn in that language system back into the caveman english, to add their sticky note
this is also why we do the initial projection from vocabulary space into a model's embedding dimension. having to cram everything we know into only as many dimensions as we have tokens is very limiting
finally after a bunch of these layers, 12 in specific, we take the final residual stream's information, project it back down into the vocabulary space, and then model has a probability distribution over each of the tokens it could generate next, which is what we wanted!
TODO: rushed, forgot the final softmax step 

"""



