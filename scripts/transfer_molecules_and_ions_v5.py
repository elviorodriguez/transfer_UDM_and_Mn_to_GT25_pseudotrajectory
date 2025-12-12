#!/usr/bin/env python3
"""
Transfer non-protein molecules from a crystal structure to pseudo-trajectory frames.
Performs structural alignment accounting for sequence differences using an iterative
pruning strategy similar to ChimeraX Matchmaker.
"""

import sys
import argparse
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Superimposer
from Bio.PDB.Polypeptide import is_aa
from Bio import pairwise2
from Bio.PDB.Structure import Structure as BioStructure
from Bio.PDB.Model import Model as BioModel
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
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
            for residue in chain:
                non_protein.append((chain.id, residue))
        else:
            for residue in chain:
                if not is_aa(residue, standard=True) and residue.id[0] != 'W':
                    non_protein.append((chain.id, residue))
    return non_protein


def align_sequences_and_get_ca_atoms(seq1, chain1, seq2, chain2):
    """
    Align two sequences and extract corresponding CA atoms.
    Returns lists of CA atoms that can be superimposed.
    """
    aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    seq1_str = ''.join([aa_dict.get(aa, 'X') for aa in seq1])
    seq2_str = ''.join([aa_dict.get(aa, 'X') for aa in seq2])
    
    from Bio.Align import substitution_matrices
    matrix = substitution_matrices.load("BLOSUM62")
    
    alignments = pairwise2.align.globalds(
        seq1_str, seq2_str, matrix, -10, -0.5,
        one_alignment_only=True
    )
    
    if not alignments:
        return [], []
    
    best_alignment = alignments[0]
    aligned_seq1, aligned_seq2 = best_alignment[0], best_alignment[1]
    
    residues1 = [res for res in chain1 if is_aa(res, standard=True)]
    residues2 = [res for res in chain2 if is_aa(res, standard=True)]
    
    ca_atoms1 = []
    ca_atoms2 = []
    
    idx1, idx2 = 0, 0
    for i in range(len(aligned_seq1)):
        char1 = aligned_seq1[i]
        char2 = aligned_seq2[i]
        
        if char1 != '-' and char2 != '-':
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


def iterative_superposition_pruning(fixed_atoms, moving_atoms, 
                                     cutoff_distance=2.0, 
                                     iteration_cutoff_multiplier=2.0,
                                     min_pairs=10):
    """
    Iterative pruning algorithm similar to ChimeraX Matchmaker.
    
    The algorithm:
    1. Performs initial superposition with all atom pairs
    2. Calculates distances for each pair after transformation
    3. Removes pairs that exceed cutoff_distance * iteration_cutoff_multiplier
    4. Repeats until no more pairs are removed or minimum is reached
    5. Returns the transformation matrix based on the pruned set
    
    Parameters:
    - fixed_atoms: List of atoms from the fixed structure
    - moving_atoms: List of atoms from the moving structure  
    - cutoff_distance: Base distance cutoff in Angstroms
    - iteration_cutoff_multiplier: Multiplier for cutoff in initial iterations
    - min_pairs: Minimum number of atom pairs to retain
    """
    
    current_fixed = list(fixed_atoms)
    current_moving = list(moving_atoms)
    
    if len(current_fixed) < min_pairs:
        print(f"  Warning: Only {len(current_fixed)} atom pairs available")
        return current_fixed, current_moving
    
    sup = Superimposer()
    iteration = 0
    max_iterations = 20
    
    while iteration < max_iterations:
        # Perform superposition
        sup.set_atoms(current_fixed, current_moving)
        rot_matrix, trans_vector = sup.rotran
        
        # Calculate distances after transformation
        distances = []
        for fixed_atom, moving_atom in zip(current_fixed, current_moving):
            fixed_coord = fixed_atom.coord
            moving_coord_transformed = np.dot(rot_matrix, moving_atom.coord) + trans_vector
            dist = np.linalg.norm(fixed_coord - moving_coord_transformed)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Dynamic cutoff: start lenient, get stricter
        if iteration < 5:
            current_cutoff = cutoff_distance * iteration_cutoff_multiplier
        else:
            current_cutoff = cutoff_distance
        
        # Keep atoms within cutoff
        keep_mask = distances <= current_cutoff
        n_keep = np.sum(keep_mask)
        
        # Stop if we can't prune anymore or would go below minimum
        if n_keep == len(current_fixed) or n_keep < min_pairs:
            break
        
        # Prune the atom lists
        current_fixed = [atom for i, atom in enumerate(current_fixed) if keep_mask[i]]
        current_moving = [atom for i, atom in enumerate(current_moving) if keep_mask[i]]
        
        iteration += 1
        
        if iteration % 5 == 0 or iteration == 1:
            mean_dist = np.mean(distances[keep_mask])
            max_dist = np.max(distances[keep_mask])
            print(f"  Iteration {iteration}: {n_keep} pairs, cutoff={current_cutoff:.2f}Å, "
                  f"mean={mean_dist:.2f}Å, max={max_dist:.2f}Å")
    
    # Final superposition with pruned atoms
    sup.set_atoms(current_fixed, current_moving)
    final_rmsd = sup.rms
    
    print(f"  Final: {len(current_fixed)} pairs ({len(current_fixed)/len(fixed_atoms)*100:.1f}% retained), "
          f"RMSD={final_rmsd:.3f}Å after {iteration} iterations")
    
    return current_fixed, current_moving


def compute_all_atom_rmsd(chain1, chain2, superimposer):
    """
    Compute all-atom RMSD between two chains after applying transformation.
    This gives a better sense of overall structural similarity.
    """
    rot_matrix, trans_vector = superimposer.rotran
    
    atoms1 = []
    atoms2 = []
    
    # Get all atoms from both chains
    for res1 in chain1:
        if is_aa(res1, standard=True):
            for atom in res1:
                if atom.name in ['N', 'CA', 'C', 'O']:  # Backbone atoms
                    atoms1.append(atom)
    
    for res2 in chain2:
        if is_aa(res2, standard=True):
            for atom in res2:
                if atom.name in ['N', 'CA', 'C', 'O']:
                    atoms2.append(atom)
    
    if len(atoms1) == 0 or len(atoms2) == 0:
        return None
    
    # Match atoms by position (after sequence alignment this should work)
    min_len = min(len(atoms1), len(atoms2))
    
    sum_sq_dist = 0.0
    for i in range(min_len):
        coord1 = atoms1[i].coord
        coord2_transformed = np.dot(rot_matrix, atoms2[i].coord) + trans_vector
        sum_sq_dist += np.sum((coord1 - coord2_transformed) ** 2)
    
    rmsd = np.sqrt(sum_sq_dist / min_len)
    return rmsd


def copy_residue_with_transform(residue, chain_id, superimposer):
    """Copy a residue and apply transformation from superimposer."""
    new_residue = Residue(residue.id, residue.resname, residue.segid)
    
    rot, tran = superimposer.rotran
    
    for atom in residue:
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
        description='Transfer non-protein molecules from crystal structure to pseudo-trajectory frames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default parameters (recommended)
  %(prog)s trajectory.pdb crystal.pdb output.pdb
  
  # More lenient for distant homologs
  %(prog)s trajectory.pdb crystal.pdb output.pdb --cutoff 3.0 --multiplier 3.0
  
  # Stricter alignment
  %(prog)s trajectory.pdb crystal.pdb output.pdb --cutoff 1.5 --multiplier 1.5
        """
    )
    parser.add_argument('trajectory_pdb', help='Input pseudo-trajectory PDB file')
    parser.add_argument('crystal_pdb', help='Input crystal structure PDB file')
    parser.add_argument('output_pdb', help='Output PDB file with transferred molecules')
    parser.add_argument('--cutoff', type=float, default=2.0,
                        help='Final distance cutoff for pruning (Å) (default: 2.0)')
    parser.add_argument('--multiplier', type=float, default=2.0,
                        help='Initial cutoff multiplier for early iterations (default: 2.0)')
    parser.add_argument('--min-pairs', type=int, default=10,
                        help='Minimum number of atom pairs to retain (default: 10)')
    
    args = parser.parse_args()
    
    pdb_parser = PDBParser(QUIET=True)
    
    print("=" * 70)
    print("MOLECULAR TRANSFER TOOL - ChimeraX Matchmaker-style alignment")
    print("=" * 70)
    
    print("\n[1/6] Loading structures...")
    trajectory_structure = pdb_parser.get_structure('trajectory', args.trajectory_pdb)
    crystal_structure = pdb_parser.get_structure('crystal', args.crystal_pdb)
    print(f"  ✓ Loaded trajectory: {args.trajectory_pdb}")
    print(f"  ✓ Loaded crystal: {args.crystal_pdb}")
    
    print("\n[2/6] Validating pseudo-trajectory...")
    trajectory_sequences = []
    for model in trajectory_structure:
        protein_chains = get_protein_chains(model)
        
        if len(protein_chains) != 1:
            print(f"  ✗ ERROR: Model {model.id} has {len(protein_chains)} protein chains. Expected 1.")
            sys.exit(1)
        
        trajectory_sequences.append(protein_chains[0][1])
    
    reference_seq = trajectory_sequences[0]
    for i, seq in enumerate(trajectory_sequences[1:], 1):
        if seq != reference_seq:
            print(f"  ✗ ERROR: Model {i} has different sequence than model 0")
            sys.exit(1)
    
    print(f"  ✓ All {len(trajectory_sequences)} models have identical sequences")
    print(f"  ✓ Sequence length: {len(reference_seq)} residues")
    
    print("\n[3/6] Validating crystal structure...")
    crystal_model = list(crystal_structure)[0]
    crystal_protein_chains = get_protein_chains(crystal_model)
    
    if len(crystal_protein_chains) != 1:
        print(f"  ✗ ERROR: Crystal has {len(crystal_protein_chains)} protein chains. Expected 1.")
        sys.exit(1)
    
    crystal_chain, crystal_seq = crystal_protein_chains[0]
    print(f"  ✓ Crystal structure validated")
    print(f"  ✓ Sequence length: {len(crystal_seq)} residues")
    
    print("\n[4/6] Identifying non-protein molecules...")
    protein_chain_ids = {crystal_chain.id}
    non_protein_residues = get_non_protein_residues(crystal_model, protein_chain_ids)
    
    if len(non_protein_residues) == 0:
        print("  ⚠ WARNING: No non-protein molecules found to transfer!")
    else:
        print(f"  ✓ Found {len(non_protein_residues)} non-protein molecules:")
        for chain_id, residue in non_protein_residues:
            print(f"    • Chain {chain_id}: {residue.resname} {residue.id}")
    
    print("\n[5/6] Aligning and transferring molecules...")
    print(f"  Parameters: cutoff={args.cutoff}Å, multiplier={args.multiplier}x, min_pairs={args.min_pairs}")
    print("-" * 70)
    
    output_structure = BioStructure('enriched_trajectory')
    
    for model_idx, traj_model in enumerate(trajectory_structure):
        print(f"\n  Model {model_idx}/{len(trajectory_structure)-1}:")
        
        traj_protein_chains = get_protein_chains(traj_model)
        traj_chain, traj_seq = traj_protein_chains[0]
        
        # Sequence alignment
        ca_fixed, ca_moving = align_sequences_and_get_ca_atoms(
            traj_seq, traj_chain,
            crystal_seq, crystal_chain
        )
        
        if len(ca_fixed) < 3:
            print(f"  ✗ ERROR: Insufficient aligned CA atoms ({len(ca_fixed)})")
            sys.exit(1)
        
        print(f"  Sequence alignment: {len(ca_fixed)} CA pairs")
        
        # Iterative structural alignment with pruning
        core_fixed, core_moving = iterative_superposition_pruning(
            ca_fixed, ca_moving,
            cutoff_distance=args.cutoff,
            iteration_cutoff_multiplier=args.multiplier,
            min_pairs=args.min_pairs
        )
        
        if len(core_fixed) < 3:
            print(f"  ✗ ERROR: Insufficient atoms after pruning ({len(core_fixed)})")
            sys.exit(1)
        
        # Get final transformation
        final_sup = Superimposer()
        final_sup.set_atoms(core_fixed, core_moving)
        
        # Create new model
        new_model = BioModel(model_idx)
        new_model.add(traj_chain.copy())
        
        # Transfer non-protein molecules
        chain_map = {}
        for orig_chain_id, residue in non_protein_residues:
            if orig_chain_id not in chain_map:
                new_chain = Chain(orig_chain_id)
                chain_map[orig_chain_id] = new_chain
                try:
                    new_model.add(new_chain)
                except:
                    new_chain = new_model[orig_chain_id]
                    chain_map[orig_chain_id] = new_chain
            
            transformed_residue = copy_residue_with_transform(
                residue, orig_chain_id, final_sup
            )
            chain_map[orig_chain_id].add(transformed_residue)
        
        output_structure.add(new_model)
    
    print("\n" + "-" * 70)
    print("\n[6/6] Saving output...")
    io = PDBIO()
    io.set_structure(output_structure)
    io.save(args.output_pdb)
    
    print(f"  ✓ Successfully saved {len(output_structure)} models")
    print(f"  ✓ Output file: {args.output_pdb}")
    print("\n" + "=" * 70)
    print("TRANSFER COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()