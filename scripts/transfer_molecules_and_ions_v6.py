#!/usr/bin/env python3
"""
Transfer non-protein molecules from a crystal structure to pseudo-trajectory frames.
Uses the ChimeraX Matchmaker algorithm for structural alignment.
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


def chimerax_matchmaker_align(fixed_atoms, moving_atoms, cutoff_distance=2.0, verbose=False):
    """
    Implements the ChimeraX Matchmaker iterative pruning algorithm.
    
    Algorithm from ChimeraX documentation:
    "In each cycle of iteration, atom pairs are removed from the match list and 
    the remaining pairs are fitted, until no matched pair is more than cutoff_distance 
    Å apart. The atom pairs removed are either the 10% farthest apart of all pairs 
    or the 50% farthest apart of all pairs exceeding the cutoff, whichever is the 
    lesser number of pairs."
    
    Parameters:
    - fixed_atoms: List of CA atoms from the reference structure
    - moving_atoms: List of CA atoms from the structure to be aligned
    - cutoff_distance: Distance cutoff in Angstroms (default: 2.0)
    - verbose: Print detailed iteration information
    
    Returns:
    - final_fixed: Pruned list of fixed atoms
    - final_moving: Pruned list of moving atoms
    - superimposer: The final Superimposer object with transformation
    """
    
    current_fixed = list(fixed_atoms)
    current_moving = list(moving_atoms)
    
    if len(current_fixed) < 3:
        print(f"  ERROR: Need at least 3 atom pairs, got {len(current_fixed)}")
        return current_fixed, current_moving, None
    
    sup = Superimposer()
    iteration = 0
    max_iterations = 100  # Safety limit
    
    if verbose:
        print(f"\n  Starting ChimeraX Matchmaker alignment with {len(current_fixed)} atom pairs")
        print(f"  Cutoff distance: {cutoff_distance} Å")
    
    while iteration < max_iterations:
        # Perform superposition with current atom pairs
        sup.set_atoms(current_fixed, current_moving)
        rot_matrix, trans_vector = sup.rotran
        
        # Calculate distances for all current pairs after transformation
        distances = []
        for fixed_atom, moving_atom in zip(current_fixed, current_moving):
            fixed_coord = fixed_atom.coord
            moving_coord_transformed = np.dot(rot_matrix, moving_atom.coord) + trans_vector
            dist = np.linalg.norm(fixed_coord - moving_coord_transformed)
            distances.append(dist)
        
        distances = np.array(distances)
        n_pairs = len(distances)
        
        # Check if all pairs are within cutoff
        max_distance = np.max(distances)
        if max_distance <= cutoff_distance:
            if verbose:
                print(f"  Iteration {iteration}: Converged! All {n_pairs} pairs within {cutoff_distance} Å")
            break
        
        # Count pairs exceeding cutoff
        exceeding_mask = distances > cutoff_distance
        n_exceeding = np.sum(exceeding_mask)
        
        if n_exceeding == 0:
            # All within cutoff, we're done
            break
        
        # ChimeraX algorithm: Remove the LESSER of:
        # 1. 10% farthest apart of ALL pairs
        # 2. 50% farthest apart of pairs EXCEEDING cutoff
        
        # Calculate number to remove by each method
        n_to_remove_method1 = max(1, int(0.10 * n_pairs))  # 10% of all pairs
        n_to_remove_method2 = max(1, int(0.50 * n_exceeding))  # 50% of exceeding pairs
        
        # Take the LESSER number
        n_to_remove = min(n_to_remove_method1, n_to_remove_method2)
        
        # Don't remove so many that we go below 3 pairs
        if n_pairs - n_to_remove < 3:
            n_to_remove = n_pairs - 3
        
        if n_to_remove <= 0:
            # Can't remove any more
            break
        
        # Sort by distance and remove the farthest n_to_remove pairs
        sorted_indices = np.argsort(distances)  # ascending order
        keep_indices = sorted_indices[:-n_to_remove]  # Remove last n_to_remove
        
        # Update the atom lists
        current_fixed = [current_fixed[i] for i in keep_indices]
        current_moving = [current_moving[i] for i in keep_indices]
        
        iteration += 1
        
        if verbose or iteration % 10 == 0:
            mean_dist = np.mean(distances[keep_indices])
            removed_dist = np.mean(distances[sorted_indices[-n_to_remove:]])
            print(f"  Iteration {iteration}: {len(current_fixed)} pairs kept, "
                  f"removed {n_to_remove} pairs, mean={mean_dist:.3f}Å, "
                  f"max={max_distance:.3f}Å, removed_mean={removed_dist:.3f}Å")
    
    # Final superposition
    sup.set_atoms(current_fixed, current_moving)
    final_rmsd = sup.rms
    
    if verbose or True:  # Always print final result
        print(f"  ✓ Alignment complete: {len(current_fixed)} pairs retained "
              f"({len(current_fixed)/len(fixed_atoms)*100:.1f}%), "
              f"RMSD={final_rmsd:.3f}Å, {iteration} iterations")
    
    return current_fixed, current_moving, sup


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
        description='Transfer non-protein molecules using ChimeraX Matchmaker algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool uses the ChimeraX Matchmaker algorithm for structural alignment:
1. Sequences are aligned using Needleman-Wunsch with BLOSUM62
2. Corresponding CA atoms are identified from the sequence alignment
3. Iterative pruning removes poorly aligned pairs until convergence

Examples:
  # Default (2.0 Å cutoff, same as ChimeraX)
  %(prog)s trajectory.pdb crystal.pdb output.pdb
  
  # More lenient cutoff for distant homologs
  %(prog)s trajectory.pdb crystal.pdb output.pdb --cutoff 3.0
  
  # Verbose output
  %(prog)s trajectory.pdb crystal.pdb output.pdb --verbose
        """
    )
    parser.add_argument('trajectory_pdb', help='Input pseudo-trajectory PDB file')
    parser.add_argument('crystal_pdb', help='Input crystal structure PDB file')
    parser.add_argument('output_pdb', help='Output PDB file with transferred molecules')
    parser.add_argument('--cutoff', type=float, default=2.0,
                        help='Iteration cutoff distance in Angstroms (default: 2.0, same as ChimeraX)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed alignment information')
    
    args = parser.parse_args()
    
    pdb_parser = PDBParser(QUIET=True)
    
    print("=" * 80)
    print(" MOLECULAR TRANSFER TOOL - ChimeraX Matchmaker Algorithm")
    print("=" * 80)
    
    print("\n[1/6] Loading structures...")
    trajectory_structure = pdb_parser.get_structure('trajectory', args.trajectory_pdb)
    crystal_structure = pdb_parser.get_structure('crystal', args.crystal_pdb)
    print(f"  ✓ Trajectory: {args.trajectory_pdb}")
    print(f"  ✓ Crystal: {args.crystal_pdb}")
    
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
    
    print(f"  ✓ All {len(trajectory_sequences)} models validated")
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
    
    print("\n[5/6] Aligning structures with ChimeraX Matchmaker algorithm...")
    print(f"  Cutoff distance: {args.cutoff} Å")
    print("=" * 80)
    
    output_structure = BioStructure('enriched_trajectory')
    
    for model_idx, traj_model in enumerate(trajectory_structure):
        print(f"\nModel {model_idx + 1}/{len(trajectory_structure)}:")
        
        traj_protein_chains = get_protein_chains(traj_model)
        traj_chain, traj_seq = traj_protein_chains[0]
        
        # Step 1: Sequence alignment
        ca_fixed, ca_moving = align_sequences_and_get_ca_atoms(
            traj_seq, traj_chain,
            crystal_seq, crystal_chain
        )
        
        if len(ca_fixed) < 3:
            print(f"  ✗ ERROR: Only {len(ca_fixed)} aligned CA atoms (need ≥3)")
            sys.exit(1)
        
        print(f"  Sequence alignment: {len(ca_fixed)} CA atom pairs")
        
        # Step 2: ChimeraX Matchmaker structural alignment with iterative pruning
        core_fixed, core_moving, final_sup = chimerax_matchmaker_align(
            ca_fixed, ca_moving,
            cutoff_distance=args.cutoff,
            verbose=args.verbose
        )
        
        if final_sup is None or len(core_fixed) < 3:
            print(f"  ✗ ERROR: Alignment failed")
            sys.exit(1)
        
        # Step 3: Create new model with trajectory protein + transferred molecules
        new_model = BioModel(model_idx)
        new_model.add(traj_chain.copy())
        
        # Transfer non-protein molecules with the aligned transformation
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
    
    print("\n" + "=" * 80)
    print("\n[6/6] Saving output...")
    io = PDBIO()
    io.set_structure(output_structure)
    io.save(args.output_pdb)
    
    print(f"  ✓ Saved {len(output_structure)} models to: {args.output_pdb}")
    print("\n" + "=" * 80)
    print(" TRANSFER COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()