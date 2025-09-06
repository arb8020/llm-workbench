# TODO: Outlier Features Analysis Updates

## Tokenizer-Based Sequences
- [x] Update dataset_utils.py to support tokenizer-based sequence chunking
- [x] Modify get_text_sequences to accept tokenizer and use token counts
- [x] Update run_full_analysis.py to load tokenizer alongside model

## Layer Selection Improvements  
- [x] Change --layers to boolean flag and default to all model layers
- [x] Update extract_activations.py to determine all layers when needed

## Future Improvements
- [ ] Address layer naming convention issues for different model architectures