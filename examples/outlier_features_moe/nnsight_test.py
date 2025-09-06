from transformers import AutoModelForCausalLM, AutoTokenizer
from torchinfo import summary
import torch

name = "allenai/OLMoE-1B-7B-0125-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

batch = tok("hello world", return_tensors="pt")
# torchinfo accepts kwargs-style dicts for forward:
print(summary(model, input_data=dict(**batch),
              depth=3, col_names=("input_size","output_size","num_params")))

