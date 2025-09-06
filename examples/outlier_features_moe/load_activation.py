import torch

def main():
    # Load saved activations
    act = torch.load("activations_layer3.pt")

    # Print basic tensor info
    print("Shape:", act.shape)
    print("Dtype:", act.dtype) 
    print("Device:", act.device)

    # Show sample of first token embedding
    # print("First token embedding (first 5 dims):", act[0,0,:5])

    # Calculate and print basic statistics
    print("Mean:", act.mean().item())
    print("Std:", act.std().item())

if __name__ == "__main__":
    main()
