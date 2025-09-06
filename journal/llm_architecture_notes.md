# architecture

## computation/thinking layers

## attention

### multi head attention

when we write or speak, we weigh different parts of the sentence so far to figure out what to say next.
as an example: if we were continuing the sentence: 'the boy is eating a blue'
we might naively use a word that follows blue frequently, like 'color' - but this doesn't make any sense!
reading the sentence, it seems clear that the next word should be something a boy might eat, that happens to be blue
so we're paying extra attention to the words 'boy' 'eating' and 'blue' to make our prediction
and as we choose the next token, those tokens get weighted more than others, allowing us to pick a word like 'lollipop'

implementation wise, we're using the most recent token, and looking at how related it might be to previous tokens in the sequence
at the end, we want some kind of 'score' of how well the most recent token matches up with previous tokens in the sequence
we can think of this kind of like a fuzzy hashmap
our model has been given the space to learn about the word 'blue', and has some internal representation of it, specifically as a word we're looking to match to
the model has also been given the space to learn about the words 'boy', 'eating', etc, and has an internal representation of it, specifically as a word to match with
and the model also has been given the space to learn about all the tokens, creating internal representations specifically as a word that's been matched
so when we're trying to figure out which tokens to pay attention to
in our little learned fuzzy hashmap, the model might get a match value between 'blue' and 'boy' of like 0.6, and apply that similarity score to the 'boy' value
and we'll end up with some vector of attention values for the whole sequence, maybe like [3, 14, 2, 15, 4, 11]
applying softmax, we get a probability distribution, in this case used as a weighting mechanism for each of the previous tokens
and now our model can use that weighting to make better predictions about that next token
interestingly, we usually also do this in parallel, so we can learn not just semantic relations but also grammatical, parts of speech, etc

rigorously, we give the model a matrix for queries (how to match to this token), keys (how to match with this token), and values (what this token 'means')
these matrices are shape (embed_dim, embed_dim) -> from the model's initial embedding dimension representation of the raw token, to a new space where it can represent the information it wants relating to attention
while running the forward pass, we take the query token corresponding to the last token in our sequence, and get its cosine similarity with the key vectors of the other tokens in the sequence (including itself)
we do this with a dot product, as the cosine similarity is basically just a scaled dot product
this gets scaled by the square root of the (embed_dim) because we don't want these unscaled dot products to get too big
when training, we mask out the values of the sentence the model isn't supposed to have seen yet - but this doesn't apply to inference
as a quick note on masking, we basically create a lower triangular matrix of really large negative values and add it to our matrix of scaled dot products
now, when we perform softmax to turn attention values into the weighting for how much attention to pay to each token
the large negative values effectively zero out the tokens we shouldn't know about - exactly the behavior we want!
that weighting gets multiplied with the values of each token, which is the final part of the model's core 'prediction' for the next token
and finally projected back into the actual token embedding representation of the model, to get token logits for the output
to add in the parallel computation of different types of attention scores at the same time, we usually split the embedding dimension into an n_head parameter. n_head=2 for two types of relationships, etc
note that we don't add new parameters! the size of each head's embedding dimension (head_dim) is a divisor of the initial embed_dim parameter (head_dim = embed_dim//n_head)
we create these heads simply by splitting up the initial (embed_dim, embed_dim) matrix, and concatenating the results later
so the top half might end up being the syntactic part of attention, the bottom half might end up being the semantic part of attention, etc

### kv caching

attention is great while we train a model: for a given window, we process attention over an entire sequence at a time
since we're evaluating a full sequence at a time, we can parallelize the operation and mask things out as necessary to avoid forward-looking behavior when undesirable
but during inference, this doesn't really apply. say we've generated 3 out of 16 tokens so far. to compute the 4th token, we compute attention over the 4 tokens, 16 computations
now to compute the 5th token, we perform attention again, over 5 tokens, 25 comparisons
but we've already computed the comparisons between positions 0-1, 0-2, ... we only really need to do the comparisons considering the new 5th token
these key/value vectors won't change, so to save on computational load, we store them in a KV cache
when we run each new forward pass, we can load in the old K/V vectors, compute the new ones, and store them for the next forward pass
in this manner, we trade memory for faster compute during inference

### grouped query attention

while we get to save a lot of compute, we still need a lot of memory during inference to maintain the kv cache
for standard multi head attention, this means key and value vectors for each layer, for each head
empirically, it seems to be the case that the different attention heads actually end up encoding quite similar key/value vectors
in multi query attention, we basically just duplicate key and value vectors across every head, and only use different queries across heads
this is because we predict that a well trained model will be able to encode the required information for all heads in one set of key/value vectors
while this method saves a ton on memory, it sometimes degrades model performance, as the heads don't get to encode as much different info across heads in the key/values
so grouped query attention is a flexible compromise between standard multi head attention and grouped query attention
we assign queries to groups of heads, and define a new parameter kv_heads, which dictates how many heads get unique key/value vectors
in standard multi head attention, n_kv_heads is just n_heads, as each head gets its own set of key/value vectors
and in multi query attention, n_kv_heads is equal to 1, as there's just one set of key/value vectors across all heads
in grouped query attention, n_kv_heads must a divisor of n_heads - and we can use multi query or multi head attention as described above, so this is a backwards-compatible improvement to multi head attention

### feedforward/fully connected network

while the attention mechanism is powerful, giving us the context of what tokens to pay attention to
we currently still just have a fancy n-gram model, we haven't added anything beyond adds/multiplications
the feedforward network is a foundational building block of many deep learning architectures
more discussion on this can be found in the activation function section, but roughly the nonlinear functions allow us to approximate a much larger class of functions than affine transformations alone
specifically, we project the embedding dimension into a higher dimensional thinking dimension, usually 4x the embedding dimension (powers of 2 convenient for GPUs + enough space to be meaningful)
then, we apply the activation functions, learn complex relations, and then compress back down into the embed_dim, keeping only the useful stuff
so now we have a neat little cycle we apply over and over again
learn relations between tokens -> learn about each token's meaning
so we combine both cross-token interaction, and the power of deep learning

### swiglu

our gelu activation feedforward network only applies a nonlinearity in one step (after projecting into the 4x embed_dim space)
but deep networks like ours have a decent amount of sources of instability
kind of like with residual connections, we want ways to more easily propagate gradients back through our networks
since these feedforward networks are where a good deal of our token processing occurs, its an important step
taking ideas from LSTMs and RNNs (out of scope), another way to apply nonlinearities to our input is through a gating mechanism combined with a linear unit
our linear unit works as normal, applying a transformation and a shift
but rather than directly transforming the whole thing with a nonlinearity like gelu
in parallel, we do another linear transformation with an activation function like sigmoid
and apply an elementwise multiplication, rather than a dot product
so the nonlinearity kind of gets applied like a mask onto the linear layer that was computed in parallel
now, rather than applying the nonlinearity directly to the higher dimensionalized input
we learn two different projections into the hidden dimension, one meant for processing and one meant for controlling or 'gating' that processing
as an example, the gate matrix might be completely mapped between (0,1) if using an activation function like sigmoid (one of the first ones used in deep learning!)
and then applying this gate matrix elementwise sort of turns on and off different parts of that projected input, still a nonlinearity being introduced but in a new way
this makes it a lot easier to flow gradients back through the network, because the linear layer preserves information and the gating controls what we care about, separately
these layers, Gated Linear Units, have been shown to be quite effective when dropped in for the usual feedforward networks
the best one uses an activation function called swish, which is sigmoid(x) * x
interestingly, using sigmoid(x) * x is kind of like self attention, the input itself determines its own gating process, which works quite well empirically

as for intuitions on why this performs better empirically, noam shazeer literally says the following in the paper where he introduced using SwiGLU in transformers:
"we offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence"

## activation functions

what really is an activation function?
we came up with them when we realized that the almighty 'draw a line through your data' couldn't solve everything
for example, consider a drug that has a dosage-response relationship that is low in both an under/overdose scenario, but high in the right dosage
your silly little 'linear regression', however powerful it may be, literally cannot learn this relation
so someone came with with the idea of adding some kind of nonlinearity, that we could compose and transform into an arbitrary shape
for example, imagine a call option payo- i mean the function max(0,x)
if we transformed and shifted a bunch of these together, we could theoretically fit any polynomial
this is known as the 'universal function approximation theorem'
the activation function is that max(0,x) nonlinear function that we compose together - its also known as ReLU (rectified linear unit) (don't ask me why)
we also have the activation function gelu in this codebase
and i wish i had a cool intuitive reason for why its used, but the people who wrote the GPT-2 paper literally just said it worked better empirically
could expand on this in the future

## regularlization and normalization

### layer norm

the repeated application of the attention and feedforward layers doesn't give the model a chance to 'recalibrate' so to speak, in between processing steps
mathematically, we might be getting some weirdly skewed or shifted values from one layer, and we propagate that skew or whatever it is immediately
so we just take our distribution from the most recent layer, and we normalize it into mean 0 and variance 1
we also allow the model to take an affine transformation (scale/shift) on that distribution, in case something different works better for the model
this has been found to smooth out the loss surface and make it easier for the model to learn

### rms norm

some researchers realized that layer norm is cool but honestly most of the value comes from the re-scaling to some learnable gamma, not the re-centering
so they scaled the inputs using the root mean square, effectively re-scaling without re-centering, and saving computational resources
and kept around the learnable scale parameter gamma
empirically was proven to be computationally cheaper and retained performance, so is now frequently used over layer norm

### residual connections

you may have noticed these models are quite deep and large
one might wonder, why not just make these models deeper and larger?
more opportunities for hierarchical representation, and more space to think is good right?
well back in the day, computer vision researchers had the same idea as you
shocker, it is not so easy
what happened was that the researchers were trying to train really deep models
but found that it didn't strictly improve model performance - which is kind of weird if you think about it
suppose we have a model that is some n layers deep
we would assume that a model n+1 layers deep should always be at least as good as the n layer model
after all, it could just replicate the n layer deep function
and use the identity function in any excess layers
but empirically, training deeper networks actually had higher training loss

roughly, as we go through these complex transformations
our initial weights early in the model can't really easily learn from/access the gradient from the input we just used
its hard for the model to preserve both the complex transformations its learning, as well as the original data
imagine playing a game of telephone, where the first person asks a question about the group of people
and the last person has to answer the question
first of all, its going to be tough to keep the original question correct, but we also need to add info to each transmission
but it would be so much easier to skip ahead and remind someone halfway through the line what the original question was
or better yet, tell everyone about it

rigorously, we're trying to estimate some function H(x)
and we're doing it by finding some intermediate functions a(b(c(d(x)))) = H(x)
since each of those functions is nonlinear, we can't really preserve x, if one of these subfunctions needs to be f(x) = x
and it turns out that one of the functions learning it should exactly preserve an input is kinda hard
but luckily, its pretty easy for functions to learn to just not do anything at all
so we should try to have each function predict what change it needs to make to x in a more pure manner
we reformulate the problem as trying to predict some H(x) = F(x) + x
since each intermediate function predicts the difference to apply to x, and learning F(x) = 0 is easy
this makes training a deep model way smoother

we call this trick a 'residual connection'

## loss

### softmax

next token prediction is basically a classification task
suppose we had to pick between [red, green, blue] in some computer vision model
the correct 'label' might be [1, 0, 0]
a classification model usually compares the label to some probability distribution output
as an example, our model output might be one of [0.9, 0.02, 0.08] or [0.4, 0.3, 0.3]
while the maximum value for both is correct
its clear that one model potentially performs better at the task
as its probability distribution is in some sense 'closer' to the true distribution

but we don't have probability distributions by default
our model might output raw logits like [5, 3, 2]
we need to turn this into a probability

call our initial output logits f(x)
we need to create probability distribution p(x), while losing minimal information from f(x)
the best representation of some knowledge state is the probability distribution that maximizes entropy
entropy (in the information theory context) is the expected value of surprise
surprise, as it turns out, is actually a mathematical quantity
to develop intuition about this, suppose you had 100 red balls and 2 blue balls in a box
drawing a blue ball out is more 'surprising' than a red ball
so we might think the surprise might be 1/p
but if p is 0 this breaks, so we just add a log transform
thus, the rigorous probability definition of surprise is log(1/p)
and entropy is E = sum (p(x) * log(1/p(x)))
this can be simplified into E = -sum(p(x) * log(p(x)))

so we can set up a constrained optimization problem
to figure out a function that can map f(x) into the best p(x)
our objective function is maximizing E = -sum(p(x) * log(p(x)))
our constraints are that
1. sum(p(x)) = 1
2. p_i >= 0

using lagrange multipliers (not shown here, too long), we can solve the above to get
p_i = e^{beta * x_i}/sum(e^{beta * x_j})
for writing the softmax, we'll remove beta
but remember that we can scale the logits before doing the softmax
this beta constant is frequently rewritten as 1/T, where T is temperature
we call it that because of the boltzmann distribution in statistical mechanics, which is out of scope

you might have noticed we have a bit of instability with this computation
e^x where x is a really big number can easily overflow
but e^-x will always be between 0 and 1
so we want to see if we can shift all of our e^x such that x is, at most, 0
suppose we subtracted some constant c from all of our x_i (from the initial f(x))
mathematically, our function is invariant to any shifts (can you prove this)
so we can just subtract the largest value from every value
to get a numerically stable softmax

### cross entropy loss

in order to actually calculate the loss function between some true p(x) and our predicted q(x)
we basically take that entropy function
E = -sum(p(x) * log(p(x)))
and we just replace one of the p(x) distributions with our model output distribution q(x)
C = -sum(p(x) * log(q(x)))
this is known as 'cross-entropy', as its the entropy 'across' two distributions
note that we choose to put the log transform on our q distribution bc log(0) is not stable

### perplexity

while we use cross entropy loss to train the model, its not as easily human-interpretable
since cross-entropy is log transformed, the differences in uncertainty in predicting each token is 'compressed' in a sense
to decompress this, we can take the exponential of cross entropy to derive a rough measure of the amount of tokens the model is choosing from on average
we call this perplexity - we take the average perplexity over a sliding window of a dataset to get a interpretable number describing how certain the model is (or how perplexed it is)

## optimizers

### stochastic gradient descent

let's use a 2 parameter example: we have a parameter x and y, each corresponding to an axis
and the loss surface is some value z for each of these parameters x,y
imagine a blind person on this loss surface: they can tell which way, locally, is the correct next step to take by feeling the slope under their feet
and at each step, they get progressively closer to some minimum point on the full loss surface
the size of the steps is determined by the learning rate of the function
but mathmatically we're literally just updating each parameter by the learning rate multiplied by the gradient with respect to the training batch
there's a little bit of variance at each step, as each step is some random 'batch' of the full dataset, but it's proven to converge for a correct choice of learning rate

### stochastic gradient descent with momentum

stochastic gradient descent works, but is a little slow
the man on the loss surface may be taking multiple steps in the same direction, without taking advantage of the most recent gradients that he felt out
compare this with a bowling ball. the bowling ball carries inertia and momentum from the previous movements made, and reaches the surface's minima faster as a result
so we add in a exponential moving average of the previous gradients to add to each of our updates

### adam

in our examples so far, we've covered the optimization of a 2 parameter model
this is less obvious in our example, but it's actually not clear that each parameter should get an equivalent learning rate
why not make it easier to move north/south than east/west if we need to?
if the surface is very steep to the left, but not steep in front of us, we probably want to take an even bigger step left instead of forward, even after adjusting for the gradient
so we can keep track of the average gradient we've seen so far for a given parameter, as well as the average squared gradient, to get more information on how to update
while, as before, parameters with a high average get bigger steps to keep using momentum
parameters with high variance/squared mean should probably get smaller updates, since its not really clear that the terrain is smooth/stable
as a side note, the initial values of mean/squared mean will be 0 when initialized, so we add a bias correction mechanism that incorporates time step to account for this

## tokenization

### byte pair encoding

our model unfortunately does not have the ability to take in raw strings, as words are not math
you may think 'oh lets just take the ascii/unicode value of each character' but there are some problems with this
the most clear framing is the context length of the model
at longer context lengths, inference would take longer, due to having to generate more characters, but also because attention is a O(seq_len^2) operation
so we maybe don't want each character to be a token
but we also might not want to make each word into a token, as we lose the ability to understand sub-words, and it makes it harder for the model to come up with new words
taking a step back, we're looking at a data compression problem
how can we figure out the representation of our training text/possible training texts that maximally retains information while also decreasing size?

lets think of each character as a byte
inspecting the data, we might find some bytes follow other bytes very often
so we can maybe just predict both of these at the same time, instead of one and then the other after
let's tag this pair of bytes with a new value, say 256 (after 255)
now, we can go back through our data and find the next most common pair of bytes, and compress them into another new value, say 257
repeatedly applying this process until we get to some desired vocab_size, or sufficient compression, we can efficiently represent some training data
this is known as byte pair encoding. the current codebase implements a rudimentary/slow version of it, but theres a lot of room for optimization in the future

## positional encoding

### gpt-2

our gpt2 model learns how to distinguish positions with a learned embedding matrix
we simply add the matrix that we got from taking the dot product of our tokens and our token embedding matrix
to a learnable positional embedding matrix, that was of size (context_len, embed_dim), taking just the slice up to seq_len (pos_embed[:seq_len])

### rotary positional encoding (RoPE)

while the gpt-2 way of doing positional encoding was simple
it was unfortunately flawed
remember that the output of the positional encoding matrix was directly added to the token embedding matrix
it might be important context to pay attention to, that 'hello' is the first word of the sentence (hello there!)
maybe signaling a greeting, rather than at the end, where 'hello' may signal confusion (um, hello?)
but the positional embeddings we've added are kind of polluting the signal of what 'hello' means
because we add the positional encoding matrix, it means something different depending on where in the context length it is
but this is a fixed size, and this doesn't really work well with variable sentence lengths
as an example, how is our model supposed to know that hello at the end of a 8 long sentence has a similar meaning to hello at the end of a 64 long sentence?
so we want an encoding strategy that is relative to position within a sequence's length, rather than absolute
but will dot products will maintain relative positional encoding?
let r_ij be the positional encoding for some distance index j to index i
let e_i, e_j be the token embeddings for tokens at indices j/i
and let q_, k_ denote the query/key vectors
so in our example, we have that the query vector for token at index i is
q_i = e_i + r_{ij}
and the key vector for token at index j is k_j = e_j (since the distance from j to j is 0)
taking the dot product of the q and k, as we do
dot(q_i, k_j) = dot(e_i + r_ij, e_j)
which expands out into
dot(q_i, k_j) = dot(e_i, e_j) + dot(r_ij, e_j)
so we still have the information of just the dot product between the embedding vectors
how can we come up with a way to encode position with a dot product?

the core idea in rotary positional encoding is we can avoid polluting the token embedding with absolute positional info
and effectively split up the token embedding from the positional encoding using complex numbers
remember that a complex number introduces an imaginary axis orthogonal to the real number line
we write a complex number as $z = a + bi$, at point $(a,b)$ on this new 2D axis
we can also express it as a magnitude away from the origin, and an angle from the real axis
where the magnitude would be $r = \sqrt{a^2+b^2}$
and the angle would be $\cos(\theta) + i\sin(\theta)$
where we can find the angle using the inverse of the tangent function on the values $a$ and $b$
using euler's formula, we now rewrite $z = a + bi$ as $z = re^{i\theta}
for us, this means that we want to represent our token embedding as the magnitude of this complex number
and we can have the angle be related to the position
so any tokens close together have a small amount of radians between them, and tokens far apart are separated by a larger angle, or more rotations
but they still maintain their magnitude/individual token information!

quick note, you may realize $i * m\theta$ has very fast oscillations/frequencies for larger values of $m-n$ (where m, n are indices)
so we should use different theta values for different embedding dimension sizes
this allows us to capture shorter and longer range dependencies
like in the 0th dimension, we would have the highest $\theta$ value
and it would decrease for larger and larger dimensions
to accomplish this, we set $\theta_j = 1+e4^{-2j/d}$ where $j$ is the index of the embedding dimension
this gives us log scaling across the embedding dimensions

to code this, we need to turn our xq, xk vectors into complex numbers
note that we apply rope to each head individually, since each head learns different relations between tokens
we first split xq/xk into real/imaginary parts arbitrarily by cutting them in half, reshaping into (..., 2)
then we use the .complex from jax to turn it into a complex number
multiply by our dimension index scaling frequency for faster/slower oscillations
remember this only depends on embed_dim and the seq_len so we usually precompute those matrices
then we extract back the real/imaginary parts of the vectors with our numerical computing library functions
and concatenate the vector back from (..., 2) back to how it was before

### scaled rope

rope is pretty great for encoding positional information using relative distance, as we just discussed
but the frequency values are bounded by the context length of the initial pretraining parameters - how will we handle longer inputs?
the simplest way to do this is simply to scale the frequencies using the new length we want to generate
new_freq = old_freq * old_len/new_len
but this disproportionately impacts the higher frequency/lower wavelen components, which means finer relationships degrade
think of rope like a clock: you can only measure those 12 hours that you set up on the clock
and linearly scaling in the way we just described just means the seconds/minute/hour hands move more slowly by that constant factor
before, if someone showed you two clocks, each a few seconds apart, they were distinguishable, albeit maybe with some difficulty
but after the scaling, that same time has the seconds hands much closer to each other, harder to understand the difference between them
so we basically want to 'protect' some lower bound, like the seconds or similar, and not scale it
and we can have a upper bound where we can fully scale, like hours or days in our example, since they'll still be easy to distinguish
and between these, we interpolate between the scaling factor of the lower bound and the scaling factor used at the upper bound
