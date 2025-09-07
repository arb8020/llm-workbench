"""
PAPER METHODOLOGY - EXACT QUOTES FROM "Outlier Features in Large Language Models"

=== MAIN FINDING ===
"As we scale transformers, outlier features with large magnitudes emerge and strongly affect all layers
and their quantization. Given a hidden state X ∈ R^(s×h) where s is the sequence/token dimension and
h the hidden/feature dimension, we define a feature to be a particular dimension h_i. Our analysis
looks at a particular feature dimension h_i across all layers of a given transformer."

"We find that outlier features strongly affect attention and the overall predictive performance of
transformers. While up to 150k outliers exist per 2048 token sequence for a 13B model, these outlier
features are highly systematic and only representing at most 7 unique feature dimensions h_i."

=== OUTLIER DEFINITION ===
"We define outliers according to the following criteria: the magnitude of the feature
is at least 6.0, affects at least 25% of layers, and affects at least 6% of the sequence dimensions."

"More formally, given a transformer with L layers and hidden state X_l ∈ R^(s×h), l = 0...L where s is
the sequence dimension and h the feature dimension, we define a feature to be a particular dimension
h_i in any of the hidden states X_l. We track dimensions h_i, 0 ≤ i ≤ h, which have at least one value
with a magnitude of α ≥ 6 and we only collect statistics if these outliers occur in the same feature
dimension h_i in at least 25% of transformer layers 0...L and appear in at least 6% of all sequence
dimensions s across all hidden states X_l."

=== WHERE OUTLIERS OCCUR ===
"Since feature outliers only occur in attention projection (key/query/value/output) and the feedforward 
network expansion layer (first sub-layer), we ignore the attention function and the FFN contraction 
layer (second sub-layer) for this analysis."

=== OUTLIER IMPACT ===
"Despite only making up about 0.1% of all features, the outliers are essential for large softmax 
probabilities. The mean top-1 softmax probability shrinks by about 20% if outliers are removed. 
Because the outliers have mostly asymmetric distributions across the sequence dimension s, these 
outlier dimensions disrupt symmetric absmax quantization and favor asymmetric zeropoint quantization."

=== SYSTEMATIC vs PROBABILISTIC ===
"For the number of layers affected by outliers, we find that outlier features are systematic in large
models: they either occur in most layers or not at all. On the other hand, they are probabilistic in
small models: they occur sometimes in some layers for each sequence. As such, we set our threshold
for how many layers need to be affected to detect an outlier feature in such a way as to limit detection
to a single outlier in our smallest model with 125M parameters. This threshold corresponds to that at
least 25% of transformer layers are affected by an outlier in the same feature dimension."

=== PHASE TRANSITION ===
"A phase transition occurs at 6.7B parameters when the same outlier occurs in all layers
in the same feature dimension for about 75% of all sequence dimensions (SDim)."

=== UNIVERSALITY ===
"These observations appear to be universal as they occur for models trained in different software 
frameworks (fairseq, OpenAI, Tensorflow-mesh), and they occur in different inference frameworks 
(fairseq, Hugging Face Transformers). These outliers also appear robust to slight variations of the 
transformer architecture (rotary embeddings, embedding norm, residual scaling, different initializations)."
"""

def find_outliers_paper_methodology(activations, magnitude_threshold=6.0, min_layer_pct=0.25, min_seq_pct=0.06):
    """
    IMPLEMENTATION OF PAPER'S OUTLIER DETECTION
    
    Paper quote: "We define outliers according to the following criteria: the magnitude of the feature
    is at least 6.0, affects at least 25% of layers, and affects at least 6% of the sequence dimensions."
    
    Paper quote: "We track dimensions h_i, 0 ≤ i ≤ h, which have at least one value with a magnitude 
    of α ≥ 6 and we only collect statistics if these outliers occur in the same feature dimension h_i 
    in at least 25% of transformer layers 0...L and appear in at least 6% of all sequence dimensions s 
    across all hidden states X_l."
    """
    # Implementation here...
    pass

def extract_attention_and_ffn_activations():
    """
    PAPER'S EXTRACTION METHODOLOGY
    
    Paper quote: "Since feature outliers only occur in attention projection (key/query/value/output) 
    and the feedforward network expansion layer (first sub-layer), we ignore the attention function 
    and the FFN contraction layer (second sub-layer) for this analysis."
    
    This means we should capture:
    - Attention projections: Q, K, V, O linear layer inputs
    - FFN expansion layer: First linear layer of feedforward network (d_model -> intermediate_dim)
    - NOT: Attention mechanism itself, FFN contraction layer (intermediate_dim -> d_model)
    """
    # Implementation here...
    pass

def validate_systematic_outliers():
    """
    PAPER'S FINDINGS ON OUTLIER BEHAVIOR
    
    Paper quote: "For the number of layers affected by outliers, we find that outlier features are 
    systematic in large models: they either occur in most layers or not at all. On the other hand, 
    they are probabilistic in small models: they occur sometimes in some layers for each sequence."
    
    Paper quote: "Despite only making up about 0.1% of all features, the outliers are essential for 
    large softmax probabilities. The mean top-1 softmax probability shrinks by about 20% if outliers 
    are removed."
    
    Expected behavior:
    - Large models (>6.7B): Outliers appear systematically across most/all layers
    - Small models (<6.7B): Outliers appear probabilistically in some layers
    - Outliers should be rare (~0.1% of features) but high impact
    """
    # Implementation here...
    pass
