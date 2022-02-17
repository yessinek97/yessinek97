"""Functions using pyrosetta."""
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import pyrosetta as pr

from biondeep_ig.data_gen.ig.util import log

logger = log.get_logger(__name__)


def init_rosetta(options: Optional[List[str]] = None) -> str:
    """Initialize pyrosetta.

    Returns:
        options passed to pyrosetta.
    """
    init_options = [
        "-ignore_unrecognized_res true",
        "-ignore_zero_occupancy false",
        "-ex1 -ex2aro",
        "-multithreading:total_threads 1 ",
        "-multithreading:interaction_graph_threads 1",
        "-out:pdb_gz",
        "-out:level 10",
    ]
    if options is not None:
        init_options = init_options + options

    pr.init(options=" ".join(init_options), set_logging_handler=False, silent=True)
    return " ".join(init_options)


def build_score_function(score_fn_name: str = "ref2015_cart.wts") -> Callable:
    """Build the score function.

    Args:
        score_fn_name: name of the score function. Defaults to "ref2015_cart.wts", the
        Rosetta Energy Function (https://pubs.acs.org/doi/10.1021/acs.jctc.7b00125).

    Returns:
        score function.
    """
    return pr.create_score_function(score_fn_name)


def build_move_map() -> pr.MoveMap:
    """Build MoveMap.

    Returns:
        MoveMap.
    """
    bigmm = pr.MoveMap()
    bigmm.set_bb(True)  # unfreeze backbone
    bigmm.set_chi(True)  # unfreeze side chain
    bigmm.set_jump(True)  # allow peptide to move wrt protein
    return bigmm


def build_min_mover(
    score_fn: Callable,
    movemap: pr.MoveMap = None,
    tolerance: float = 0.0001,
) -> pr.rosetta.protocols.minimization_packing.MinMover:
    """Build MinMover.

    Args:
        score_fn: score function.
        movemap: pyrosetta movemap instance.
        tolerance: min mover tolerance parameter.

    Returns:
        MinMover.
    """
    if movemap is None:
        movemap = build_move_map()
    minmover = pr.rosetta.protocols.minimization_packing.MinMover(
        movemap, score_fn, "lbfgs_armijo_nonmonotone", tolerance, True
    )
    return minmover


def build_fast_relax(score_fn: Callable) -> Any:
    """Build Fast Relax.

    Args:
        score_fn: score function.

    Returns:
        Fast Relax.
    """
    fast_relax = pr.rosetta.protocols.relax.FastRelax()
    fast_relax.set_scorefxn(score_fn)
    fast_relax.max_iter(100)
    return fast_relax


def build_randomizer() -> Any:
    """Build randomizer.

    Returns:
        randomizer.
    """
    randomize = pr.rosetta.protocols.moves.SequenceMover()
    randomize.add_mover(
        pr.rosetta.protocols.rigid.RigidBodyPerturbMover(1, 6, 3)
    )  # Move about 6 A away, rotate about 3 degrees`
    randomize.add_mover(
        pr.rosetta.protocols.docking.DockingSlideIntoContact(1)
    )  # 1 is the structure jump - leave it be
    return randomize


def build_dock_min_mover(score_fn: Callable, name: str = "DockMinMover") -> Tuple[Any, int]:
    """Build docker.

    - If hrdock is DockMinMover, we may need ~100-1000 runs.
    - If hrdock in DockingHighResLegacy - we need about 3-5 runs.

    Args:
        score_fn: score function.
        name: name of the docking method.

    Returns:
        - Docking method.
        - number of iterations
    """
    if name == "DockMinMover":
        hrdock = pr.rosetta.protocols.docking.DockMinMover()
        num_iter = 4
    else:
        hrdock = pr.rosetta.protocols.docking.DockingHighResLegacy()
        num_iter = 100
    hrdock.set_movable_jumps(pr.Vector1([1]))
    hrdock.set_scorefxn(score_fn)
    return hrdock, num_iter


def redock(score_fn: Callable, minmover: Any, pose: pr.Pose, temperature: int) -> pr.Pose:
    """Redocker the pose.

    Args:
        score_fn: score function.
        minmover: min mover.
        pose: pose of pyrosetta.
        temperature: temperature hyper-param.

    Returns:
        a new pose.
    """
    p = pose.clone()
    mc = pr.MonteCarlo(p, score_fn, temperature)
    randomizer = build_randomizer()
    hrdock, num_iter = build_dock_min_mover(score_fn=score_fn)
    for _ in range(num_iter):
        randomizer.apply(p)
        hrdock.apply(p)
        mc.boltzmann(p)
    minmover.apply(p)
    return p


def relax(score_fn: Callable, pose: pr.Pose) -> pr.Pose:
    """Relax the pose.

    Args:
        score_fn: scoring function.
        pose: pose of pyrosetta.

    Returns:
        a new pose.
    """
    fast_relax = build_fast_relax(score_fn)
    copied = pose.clone()
    fast_relax.apply(copied)
    return copied


def insert_extra_residues(
    pose: pr.Pose,
    seq: str,
    minmover: pr.rosetta.protocols.minimization_packing.MinMover,
    movemap: pr.MoveMap,
    mutant_position_start: int,
    num_extra_residues: int,
    insert_pos_offset: int = 3,
):
    """Insert extra residues in pose.

    Compare pose sequence and target sequence, and insert residues
    in the middle of the chain if needed.

    Args:
        pose: rosetta pose to mutate.
        seq: mutant sequence.
        minmover: min mover.
        movemap: pyrosetta movemap instance.
        mutant_position_start: first position in the target chain.
        num_extra_residues: number of extra residues to add.
        insert_pos_offset: offset position from where residues are added.
    """
    if num_extra_residues == 0:
        logger.warning("insert_extra_residues was called but there is not residue to insert.")
    if num_extra_residues < 0:
        raise ValueError("The peptide sequence is shorter than the one found in the structure.")

    if insert_pos_offset > len(seq):
        raise ValueError("insert_pos_offset can't be greater than the sequence length.")

    # Unfreeze backbone and chi torsions for first 3 residues in chain
    for i in range(insert_pos_offset):
        movemap.set_bb(i + mutant_position_start, True)  # unfreeze backbone
        movemap.set_chi(i + mutant_position_start, True)  # unfreeze side chains

    chm = pr.rosetta.core.chemical.ChemicalManager.get_instance()
    resiset = chm.residue_type_set("fa_standard")
    for i in range(num_extra_residues):
        res_type = resiset.get_representative_type_name1(seq[i + insert_pos_offset])
        residue = pr.rosetta.core.conformation.ResidueFactory.create_residue(res_type)
        # residues are inserted in the middle of the chain,
        # insert_pos_offset residues away for its head
        residue_pos = mutant_position_start + i + insert_pos_offset
        logger.info(
            "Inserting residue %s at position %i",
            seq[i + insert_pos_offset],
            residue_pos,
        )
        # insert extra residue at position residue_pos
        pose.prepend_polymer_residue_before_seqpos(residue, residue_pos + 1, True)

        movemap.set_bb(residue_pos, True)  # unfreeze backbone
        movemap.set_chi(residue_pos, True)  # unfreeze side chains

        pose.set_omega(residue_pos, 180)
        minmover.apply(pose)


def get_mutate_residues(
    pose: pr.Pose,
    seq: str,
    movemap: pr.MoveMap,
    mutant_position_start: int,
) -> Tuple[
    pr.rosetta.core.pack.task.PackerTask, pr.rosetta.utility.vector1_unsigned_long, Any, Set[int]
]:
    """Mutate residues in pose according to a mutant sequence.

    Args:
        pose: rosetta pose to mutate.
        seq: mutant sequence.
        movemap: pyrosetta movemap instance.
        mutant_position_start: first position in the target chain.

    Returns:
        a tuple containing the packer task to be executed,
        the lists of corresponding pivots and centers,
        and the list of the mutated positions.
    """
    centers = []  # List of centers of residues
    mutant_positions = set()  # set of residues' positions
    pivots = pr.rosetta.utility.vector1_unsigned_long()
    # task to position side chains
    task = pr.rosetta.core.pack.task.TaskFactory.create_packer_task(pose)
    for i, aa in enumerate(seq):
        mutant_aa = pr.rosetta.core.chemical.aa_from_oneletter_code(aa)  # str -> int
        mutant_position = mutant_position_start + i
        logger.info(
            "Mutating residue at position %i to %s",
            mutant_position,
            mutant_aa,
        )
        mutant_positions.add(mutant_position)
        aa_bool = pr.rosetta.utility.vector1_bool()
        movemap.set_bb(mutant_position, True)  # unfreeze backbone
        movemap.set_chi(mutant_position, True)  # unfreeze side chains
        pivots.append(mutant_position)
        for j in range(1, 21):
            aa_bool.append(j == mutant_aa)
        # define the allowed AA at the mutation positions (peptide positions)
        task.nonconst_residue_task(mutant_position).restrict_absent_canonical_aas(aa_bool)
        centers.append(pose.residue(mutant_position).nbr_atom_xyz())

    return task, pivots, centers, mutant_positions


def substitute_chain(  # noqa:CCR001
    score_fn: Callable,
    pose: pr.Pose,
    seq: str,
    chain: str,
    start: int,
    temperature: int,
    num_extra_residues: int,
    pack_radius: int = 8,
    insert_pos_offset: int = 3,
) -> pr.Pose:
    """Replace one chain (peptide in the pose) with a sequence (new peptide), and minimize.

    More information about amino acid insertion in the blog:
    https://blog.matteoferla.com/2020/07/filling-missing-loops-proper-way.html
    Args:
        score_fn: score function.
        pose: pose of pyrosetta.
        seq: AA sequence.
        chain: chain ID.
        start: starting index.
        temperature: temperature hyper-param.
        num_extra_residues: number of extra residues to add.
        pack_radius: linked to the definition of neighbor.
        insert_pos_offset: offset position from where residues are added.

    Returns:
        updated pose with new desired peptide.
    """
    movemap = build_move_map()
    movemap.set_bb(False)  # freeze backbone
    movemap.set_chi(False)  # freeze side chain
    movemap.set_jump(True)  # allow peptide to move wrt protein

    # Position of first residue in chain
    mutant_position_start = pose.pdb_info().pdb2pose(chain, start)
    # TODO: find out why some C chains don't start with AA at pos 1
    increment = 1
    while mutant_position_start == 0:
        if increment > len(seq):
            raise RuntimeError(
                f"The chain {chain} start position is wrongly defined."
                f"Please make sure the first residue in chain {chain} has chain index 1."
            )
        mutant_position_start = pose.pdb_info().pdb2pose(chain, start + increment)
        increment += 1

    if num_extra_residues > 0:
        movemap.set_jump(False)
        insert_minmover = build_min_mover(score_fn=score_fn, movemap=movemap, tolerance=0.01)
        insert_minmover.max_iter(2000)

        insert_extra_residues(
            pose=pose,
            seq=seq,
            minmover=insert_minmover,
            movemap=movemap,
            mutant_position_start=mutant_position_start,
            num_extra_residues=num_extra_residues,
            insert_pos_offset=insert_pos_offset,
        )

        movemap.set_bb(False)  # freeze backbone
        movemap.set_chi(False)  # freeze side chain
        movemap.set_jump(True)  # allow peptide to move wrt protein

    task, pivots, centers, mutant_positions = get_mutate_residues(
        pose=pose,
        seq=seq,
        movemap=movemap,
        mutant_position_start=mutant_position_start,
    )

    rr = pack_radius ** 2
    logger.info("Finding neighbours in the MHC...")
    for i in range(1, pose.total_residue() + 1):
        if i in mutant_positions:
            # skip peptide positions
            continue
        neighbor = False
        if pack_radius > 3:
            for center in centers:
                if center.distance_squared(pose.residue(i).nbr_atom_xyz()) < rr:
                    # allow backbone and side chains (that are neighbors of peptide) to move
                    neighbor = True
                    task.nonconst_residue_task(i).restrict_to_repacking()
                    pivots.append(i)
                    movemap.set_bb(i, True)  # unfreeze backbone
                    movemap.set_chi(i, True)  # unfreeze side chain
                    break
        if not neighbor and i not in mutant_positions:
            # freeze all non-neighbours residues
            task.nonconst_residue_task(i).prevent_repacking()

    # optimize the pose wrt to the score
    # set mover
    logger.info("Apply packing...")
    packer = pr.rosetta.protocols.minimization_packing.PackRotamersMover(score_fn, task)
    packer.apply(pose)

    # small moves on backbone, normally bad
    bm = pr.rosetta.protocols.backrub.BackrubMover()
    bm.set_pivot_residues(pivots)

    # use MC to revert pose in case the moves are bad
    logger.info("Apply Monte Carlo...")
    mc = pr.MonteCarlo(pose, score_fn, temperature)
    for _ in range(2500):
        bm.apply(pose)
        mc.boltzmann(pose)

    # minimize
    logger.info("Final minimization...")
    minmover = build_min_mover(score_fn=score_fn, movemap=movemap)
    minmover.apply(pose)

    return pose


def rechain(pose: pr.Pose, chain_mapping: Dict[str, str]) -> pr.Pose:
    """Rename chains in the pose.

    https://www.rosettacommons.org/manuals/latest/core+protocols/de/db3/classcore_1_1pose_1_1_p_d_b_info.html

    Args:
        pose: pyrosetta pose.
        chain_mapping: mapping dictionary between original chain id and new chain id

    Returns:
        a new pose.
    """
    for i in range(1, pose.total_residue() + 1):
        # returns the chain letter for pose residue i
        chain = pose.pdb_info().chain(i)
        if chain in chain_mapping:
            # sets the chain letter of pose residue i
            pose.pdb_info().chain(i, chain_mapping[chain])
    return pose
