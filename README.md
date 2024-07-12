# quantum_lavalens

pip install fairscale tiktoken blobfile

Usage example:

### For CUDA:

python main.py --llama_dir ../models/Meta-Llama-3-8B --sae_dir ../models/sae-llama-3-8b-32x --sae_layers 0,1,2,3,4,5

### For MPS (Apple Metal):

python main.py –llama_dir ../models/Meta-Llama-3-8B –sae_dir ../models/sae-llama-3-8b-32x –sae_layers 0,1,2,3,4,5 –device mps

