from time import time

import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import re
import os
import gzip
import pickle

def generate_esm_features(input_dir, dataset_name, protein_type='l', binding_type='u'):
    t0 = time()
    save_dir = input_dir + dataset_name

    # -------------------------------
    # 1) Load the ESM model
    # -------------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    client = ESMC.from_pretrained("esmc_300m").to(device)
    client.eval()

    # -------------------------------
    # 2) Read the protein list and extract sequences
    # -------------------------------
    sequences = []
    with open(os.path.join(save_dir, 'proteins.txt'), 'r') as f:
        protein_list = [x.strip() for x in f]

    print("total proteins:", len(protein_list))
    for protein in protein_list:
        fasta_path = os.path.join(save_dir, 'fasta_files', f"{protein}_{protein_type}_{binding_type}.fasta")
        seq_lines = open(fasta_path, 'r').readlines()
        if len(seq_lines) < 2:
            print(f"{protein}: FASTA file has {len(seq_lines)} line(s); the content may be incorrect")
        # Even-indexed lines (starting from line 2) contain sequences in FASTA
        for i in range(1, len(seq_lines), 2):
            sequences.append(seq_lines[i].strip())

    print("sequence length:", len(sequences))

    # Replace invalid characters (ESM does not require inserting spaces)
    sequences_clean = [re.sub(r"[-UZOB]", "X", seq) for seq in sequences]
    print("Sequence example length=", len(sequences_clean))

    # -------------------------------
    # 3) Generate ESM embeddings
    # -------------------------------
    all_protein_features = {}
    for i, seq in enumerate(sequences_clean):
        # Create an ESMProtein object
        protein = ESMProtein(sequence=seq)
        
        # Encode the protein
        protein_tensor = client.encode(protein)
        
        # Extract embeddings
        with torch.no_grad():
            logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            
        # Get embeddings and remove special start/end tokens [:, 1:-1, :]
        embedding = logits_output.embeddings[:, 1:-1, :].cpu().numpy()
        
        # Squeeze to remove the batch dimension
        seq_features = embedding.squeeze(0)  # shape: (seq_len, embedding_dim)
        
        all_protein_features[protein_list[i]] = seq_features

    # -------------------------------
    # 4) Save results
    # -------------------------------
    output_name = 'ESM_all_Ligand_features' if protein_type == 'l' else 'ESM_all_Receptor_features'
    out_path = os.path.join(save_dir, f"{output_name}.pkl.gz")
    pickle.dump(all_protein_features, gzip.open(out_path, 'wb'))

    print('Total time spent generating ESM features:', time() - t0)
