#!/usr/bin/env python3
"""
Transfer non-protein molecules from a crystal structure to pseudo-trajectory frames.
Performs structural alignment accounting for sequence differences.
"""

import sys
import argparse
from Bio.PDB import PDBParser, PDBIO, Superimposer, Select
from Bio.PDB.Polypeptide import is_aa
from Bio import pairwise2
from Bio.PDB.Structure import Structure as BioStructure
from Bio.PDB.Model import Model as BioModel
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter('ignore', PDBConstructionWarning)


def get_protein_sequence(chain):
    """Extract protein sequence from a chain."""
    sequence = []
    for residue in chain:
        if is_aa(residue, standard=True):
            sequence.append(residue.get_resname())
    return sequence


def get_protein_chains(model):
    """Get all protein chains from a model."""
    protein_chains = []
    for chain in model:
        seq = get_protein_sequence(chain)
        if seq:
            protein_chains.append((chain, seq))
    return protein_chains


def get_non_protein_residues(model, protein_chain_ids):
    """Extract non-protein residues (ligands, ions, water, etc.)."""
    non_protein = []
    for chain in model:
        if chain.id not in protein_chain_ids:
            # Entire chain is non-protein
            for residue in chain:
                non_protein.append((chain.id, residue))
        else:
            # Check for non-protein residues in protein chain
            for residue in chain:
                if not is_aa(residue, standard=True) and residue.id[0] != 'W':
                    non_protein.append((chain.id, residue))
    return non_protein


def align_sequences_and_get_ca_atoms(seq1, chain1, seq2, chain2):
    """
    Align two sequences and extract corresponding CA atoms.
    Returns lists of CA atoms that can be superimposed.
    """
    # Convert 3-letter codes to 1-letter for alignment
    aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    seq1_str = ''.join([aa_dict.get(aa, 'X') for aa in seq1])
    seq2_str = ''.join([aa_dict.get(aa, 'X') for aa in seq2])
    
    # Perform sequence alignment
    alignments = pairwise2.align.globalxx(seq1_str, seq2_str)
    if not alignments:
        return [], []
    
    best_alignment = alignments[0]
    aligned_seq1, aligned_seq2 = best_alignment[0], best_alignment[1]
    
    # Get residue lists
    residues1 = [res for res in chain1 if is_aa(res, standard=True)]
    residues2 = [res for res in chain2 if is_aa(res, standard=True)]
    
    # Extract corresponding CA atoms based on alignment
    ca_atoms1 = []
    ca_atoms2 = []
    
    idx1, idx2 = 0, 0
    for i in range(len(aligned_seq1)):
        char1 = aligned_seq1[i]
        char2 = aligned_seq2[i]
        
        if char1 != '-' and char2 != '-':
            # Both positions are amino acids
            if idx1 < len(residues1) and idx2 < len(residues2):
                res1 = residues1[idx1]
                res2 = residues2[idx2]
                if 'CA' in res1 and 'CA' in res2:
                    ca_atoms1.append(res1['CA'])
                    ca_atoms2.append(res2['CA'])
        
        if char1 != '-':
            idx1 += 1
        if char2 != '-':
            idx2 += 1
    
    return ca_atoms1, ca_atoms2


def copy_residue_with_transform(residue, chain_id, superimposer):
    """Copy a residue and apply transformation from superimposer."""
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    import numpy as np
    
    new_residue = Residue(residue.id, residue.resname, residue.segid)
    
    # Get rotation matrix and translation vector
    rot, tran = superimposer.rotran
    
    for atom in residue:
        # Apply transformation: new_coord = rot @ old_coord + tran
        old_coord = atom.coord
        new_coord = np.dot(rot, old_coord) + tran
        
        new_atom = Atom(
            atom.name,
            new_coord,
            atom.bfactor,
            atom.occupancy,
            atom.altloc,
            atom.fullname,
            atom.serial_number,
            atom.element
        )
        new_residue.add(new_atom)
    
    return new_residue


def main():
    parser = argparse.ArgumentParser(
        description='Transfer non-protein molecules from crystal structure to pseudo-trajectory frames'
    )
    parser.add_argument('trajectory_pdb', help='Input pseudo-trajectory PDB file')
    parser.add_argument('crystal_pdb', help='Input crystal structure PDB file')
    parser.add_argument('output_pdb', help='Output PDB file with transferred molecules')
    
    args = parser.parse_args()
    
    # Parse PDB files
    pdb_parser = PDBParser(QUIET=True)
    
    print("Loading pseudo-trajectory...")
    trajectory_structure = pdb_parser.get_structure('trajectory', args.trajectory_pdb)
    
    print("Loading crystal structure...")
    crystal_structure = pdb_parser.get_structure('crystal', args.crystal_pdb)
    
    # Step 1: Validate pseudo-trajectory
    print("\nValidating pseudo-trajectory...")
    trajectory_sequences = []
    for model in trajectory_structure:
        protein_chains = get_protein_chains(model)
        
        if len(protein_chains) != 1:
            print(f"ERROR: Model {model.id} has {len(protein_chains)} protein chains. Expected 1.")
            sys.exit(1)
        
        trajectory_sequences.append(protein_chains[0][1])
    
    # Check all sequences are the same
    reference_seq = trajectory_sequences[0]
    for i, seq in enumerate(trajectory_sequences[1:], 1):
        if seq != reference_seq:
            print(f"ERROR: Model {i} has different sequence than model 0")
            sys.exit(1)
    
    print(f"✓ All {len(trajectory_sequences)} models have identical sequences ({len(reference_seq)} residues)")
    
    # Step 2: Validate crystal structure
    print("\nValidating crystal structure...")
    crystal_model = list(crystal_structure)[0]
    crystal_protein_chains = get_protein_chains(crystal_model)
    
    if len(crystal_protein_chains) != 1:
        print(f"ERROR: Crystal structure has {len(crystal_protein_chains)} protein chains. Expected 1.")
        sys.exit(1)
    
    crystal_chain, crystal_seq = crystal_protein_chains[0]
    print(f"✓ Crystal structure has 1 protein chain ({len(crystal_seq)} residues)")
    
    # Step 3: Identify non-protein molecules
    print("\nIdentifying non-protein molecules...")
    protein_chain_ids = {crystal_chain.id}
    non_protein_residues = get_non_protein_residues(crystal_model, protein_chain_ids)
    
    print(f"✓ Found {len(non_protein_residues)} non-protein residues/molecules:")
    for chain_id, residue in non_protein_residues:
        print(f"  - Chain {chain_id}, {residue.resname} {residue.id}")
    
    if len(non_protein_residues) == 0:
        print("WARNING: No non-protein molecules found to transfer")
    
    # Step 4 & 5: Align and transfer for each model
    print("\nAligning and transferring molecules...")
    output_structure = BioStructure('enriched_trajectory')
    
    for model_idx, traj_model in enumerate(trajectory_structure):
        print(f"Processing model {model_idx}...")
        
        # Get trajectory protein chain
        traj_protein_chains = get_protein_chains(traj_model)
        traj_chain, traj_seq = traj_protein_chains[0]
        
        # Align sequences and get corresponding CA atoms
        ca_fixed, ca_moving = align_sequences_and_get_ca_atoms(
            traj_seq, traj_chain,
            crystal_seq, crystal_chain
        )
        
        if len(ca_fixed) < 3:
            print(f"ERROR: Insufficient aligned CA atoms ({len(ca_fixed)}) for model {model_idx}")
            sys.exit(1)
        
        print(f"  Aligned {len(ca_fixed)} CA atoms (out of {len(traj_seq)} traj, {len(crystal_seq)} crystal)")
        
        # Perform superimposition
        super_imposer = Superimposer()
        super_imposer.set_atoms(ca_fixed, ca_moving)
        rmsd = super_imposer.rms
        print(f"  RMSD: {rmsd:.3f} Å")
        
        # Create new model with trajectory protein + transferred molecules
        new_model = BioModel(model_idx)
        
        # Copy trajectory protein chain as-is
        new_model.add(traj_chain.copy())
        
        # Transfer non-protein molecules with transformation
        from Bio.PDB.Chain import Chain
        chain_map = {}
        
        for orig_chain_id, residue in non_protein_residues:
            # Create chain if it doesn't exist
            if orig_chain_id not in chain_map:
                new_chain = Chain(orig_chain_id)
                chain_map[orig_chain_id] = new_chain
                try:
                    new_model.add(new_chain)
                except:
                    # Chain might already exist from protein
                    new_chain = new_model[orig_chain_id]
                    chain_map[orig_chain_id] = new_chain
            
            # Transform and add residue
            transformed_residue = copy_residue_with_transform(
                residue, orig_chain_id, super_imposer
            )
            chain_map[orig_chain_id].add(transformed_residue)
        
        output_structure.add(new_model)
    
    # Step 6: Save output
    print(f"\nSaving output to {args.output_pdb}...")
    io = PDBIO()
    io.set_structure(output_structure)
    io.save(args.output_pdb)
    
    print(f"✓ Successfully saved {len(output_structure)} models to {args.output_pdb}")
    print("Done!")


if __name__ == '__main__':
    main()