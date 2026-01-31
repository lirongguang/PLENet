import numpy as np
from time import time
import gzip
import warnings
import pickle
import os

warnings.filterwarnings("ignore")

from Bio.PDB import *
from config import DefaultConfig

from Bio import SeqUtils

# configs = DefaultConfig()
parser = PDBParser()
protein_letters_3to1 = SeqUtils.IUPACData.protein_letters_3to1
ks = list(protein_letters_3to1.keys())

ppb = PPBuilder()


def get_residue_coordinates_for_masif(save_dir, protein_name, protein_type):
    """
    Extract residue coordinates for proteins in the MaSIF dataset
    """
    complex_name, ligand, receptor = protein_name.split('_')
    if protein_type == 'l':
        chains = ligand
    else:
        chains = receptor
    
    all_coordinates = []
    residue_info = []
        
    for chain in chains:
        path = save_dir + '/pdb_files/' + complex_name + '_' + chain + '.pdb'
        structure = parser.get_structure(complex_name, path)
        model = structure[0]

        chain_residues = model[chain].get_residues()

        for res in chain_residues:
            res_name = res.get_resname()
            if res_name[0] + res_name[1:].lower() not in protein_letters_3to1:
                continue
            
            # Get the CA atom coordinate of the residue
            # (modify here if all-atom coordinates are needed)
            ca_atom = None
            for atom in res.get_atoms():
                if atom.get_name() == 'CA':
                    ca_atom = atom
                    break
            
            if ca_atom is not None:
                coord = ca_atom.get_coord()
                all_coordinates.append(coord)
                
                # Store residue information for debugging
                residue_info.append({
                    'chain': chain,
                    'residue_id': res.get_id(),
                    'residue_name': res_name,
                    'coordinates': coord
                })
    
    return np.array(all_coordinates), residue_info


def get_residue_coordinates_for_dbd5_dockground(save_dir, protein_name, protein_type, binding_type):
    """
    Extract residue coordinates for proteins in the DBD5/DockGround dataset
    """
    try:
        path = save_dir + '/pdb_files/' + protein_name + '_' + protein_type + '_' + binding_type + '.pdb'
        structure = parser.get_structure(protein_name, path)
    except:
        path = save_dir + '/pdb_files/' + protein_name.lower() + '_' + protein_type + '_' + binding_type + '.pdb'
        structure = parser.get_structure(protein_name, path)

    model = structure[0]
    all_coordinates = []
    residue_info = []
    chains = list(model.child_dict.keys())

    for chain_id in chains:
        for res in model[chain_id].get_residues():
            res_name = res.get_resname()
            if res_name[0] + res_name[1:].lower() not in protein_letters_3to1:
                continue
            
            # Get the CA atom coordinate of the residue
            ca_atom = None
            for atom in res.get_atoms():
                if atom.get_name() == 'CA':
                    ca_atom = atom
                    break
            
            if ca_atom is not None:
                coord = ca_atom.get_coord()
                all_coordinates.append(coord)
                
                # Store residue information for debugging
                residue_info.append({
                    'chain': chain_id,
                    'residue_id': res.get_id(),
                    'residue_name': res_name,
                    'coordinates': coord
                })

    return np.array(all_coordinates), residue_info


def save_checkpoint(residue_coordinates, save_path, processed_count):
    """
    Save a checkpoint of intermediate results
    """
    try:
        checkpoint_path = save_path.replace('.pkl.gz', f'_checkpoint_{processed_count}.pkl.gz')
        with gzip.open(checkpoint_path, 'wb') as f:
            pickle.dump(residue_coordinates, f)
        print(f"Checkpoint saved to: {checkpoint_path}")
    except Exception as e:
        print(f"Error while saving checkpoint: {e}")


def load_existing_results(save_path):
    """
    Load existing result files if available
    """
    # Search for the latest checkpoint file
    checkpoint_files = []
    if os.path.exists(save_path):
        directory = os.path.dirname(save_path)
        basename = os.path.basename(save_path).replace('.pkl.gz', '')
        
        for filename in os.listdir(directory):
            if filename.startswith(basename + '_checkpoint_') and filename.endswith('.pkl.gz'):
                checkpoint_files.append(filename)
    
    if checkpoint_files:
        # Sort by index and select the latest checkpoint
        latest_checkpoint = max(
            checkpoint_files,
            key=lambda x: int(x.split('_checkpoint_')[1].split('.')[0])
        )
        checkpoint_path = os.path.join(os.path.dirname(save_path), latest_checkpoint)
        
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            with gzip.open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return {}
    
    return {}


def generate_residue_coordinates(input_dir, dataset_name, protein_type='l', binding_type='u'):
    """
    Generate residue coordinate data
    """
    t0 = time()
    save_dir = input_dir + '/' + dataset_name
    
    # Output file path
    output_path = save_dir + '/' + protein_type + '_residue_coordinates.pkl.gz'
    
    # Load existing results if any
    residue_coordinates = load_existing_results(output_path)
    
    with open(save_dir + '/proteins.txt', 'r') as f:
        protein_list = [x.strip() for x in f.readlines()]

    print(f"Total proteins to process: {len(protein_list)}")
    print(f"Already processed proteins: {len(residue_coordinates)}")

    processed_count = len(residue_coordinates)
    
    for i, protein_name in enumerate(protein_list):
        # Skip proteins that have already been processed
        if protein_name in residue_coordinates:
            continue
            
        print(f"Processing protein {i + 1}/{len(protein_list)}: {protein_name}")
        
        try:
            if dataset_name == 'masif':
                coordinates, residue_info = get_residue_coordinates_for_masif(
                    save_dir, protein_name, protein_type
                )
            else:
                coordinates, residue_info = get_residue_coordinates_for_dbd5_dockground(
                    save_dir, protein_name, protein_type, binding_type
                )

            # Save coordinate data
            residue_coordinates[protein_name] = {
                'coordinates': coordinates,      # [n_residues, 3] numpy array
                'residue_info': residue_info     # detailed residue information (optional)
            }
            
            print(
                f"Successfully processed {protein_name}, "
                f"number of residues: {len(coordinates)}, "
                f"coordinate shape: {coordinates.shape}"
            )
            
        except Exception as e:
            print(f"Error while processing protein {protein_name}: {e}")
            print(f"Saving empty coordinates for {protein_name}")
            
            # For error cases, save empty coordinate arrays
            residue_coordinates[protein_name] = {
                'coordinates': np.array([]).reshape(0, 3),
                'residue_info': []
            }
        
        processed_count += 1
        
        # Save a checkpoint every 100 proteins
        if processed_count % 100 == 0:
            save_checkpoint(residue_coordinates, output_path, processed_count)
            print(f"{processed_count} proteins processed, checkpoint saved")
    
    # Save final results
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(residue_coordinates, f)
    
    print(f'Final results saved to: {output_path}')
    print('Total time for coordinate extraction:', time() - t0)
    print(f'Successfully processed {len(residue_coordinates)} proteins')

# Example usage
# if __name__ == "__main__":
#     input_dir = "/data/gxmst/PPIs/Pair-EGRET-main/inputs"  # Replace with actual path
#     dataset_name = "dbd5"  # or "masif"
#     protein_type = "l"     # or "r"
#     binding_type = "u"     # only required for non-masif datasets
#
#     generate_residue_coordinates(input_dir, dataset_name, protein_type, binding_type)
