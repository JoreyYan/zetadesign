from model.np.relax import relax
import time
import os
import logging
import json
import ml_collections as mlc

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def update_timings(timing_dict, output_file=os.path.join(os.getcwd(), "timings.json")):
    """
    Write dictionary of one or more run step times to a file
    """
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                timings = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Overwriting non-standard JSON in {output_file}.")
                timings = {}
    else:
        timings = {}
    timings.update(timing_dict)
    with open(output_file, "w") as f:
        json.dump(timings, f)
    return

config = mlc.ConfigDict({
    "relax": {
        "max_iterations": 0,  # no max
        "tolerance": 2.39,
        "stiffness": 10,  #
        "max_outer_iterations": 20, #
        "exclude_residues": [],
    }}
)

def relax_protein( model_device, unrelaxed_protein, output_directory, output_name):
    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(model_device != "cpu"),
        **config.relax,
    )

    t = time.perf_counter()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
    if "cuda" in model_device:
        device_no = model_device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_no
    relaxed_pdb_str, debug_data, violations = amber_relaxer.process(prot=unrelaxed_protein)

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    relaxation_time = time.perf_counter() - t

    logger.info(f"Relaxation time: {relaxation_time}")
    update_timings({"relaxation": relaxation_time}, os.path.join(output_directory, "timings.json"))

    # Save the relaxed PDB.
    relaxed_output_path = os.path.join(
        output_directory, f'{output_name}_relaxed.pdb'
    )
    with open(relaxed_output_path, 'w') as fp:
        fp.write(relaxed_pdb_str)

    logger.info(f"Relaxed output written to {relaxed_output_path}...")

def clean_protein( model_device, unrelaxed_protein, output_directory, output_name):
    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(model_device != "cpu"),
        **config.relax,
    )

    t = time.perf_counter()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
    if "cuda" in model_device:
        device_no = model_device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_no
    relaxed_pdb_str = amber_relaxer.clean(prot=unrelaxed_protein)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    relaxation_time = time.perf_counter() - t

    logger.info(f"Relaxation time: {relaxation_time}")
    update_timings({"relaxation": relaxation_time}, os.path.join(output_directory, "timings.json"))

    # Save the relaxed PDB.
    relaxed_output_path = os.path.join(
        output_directory, f'{output_name}_relaxed.pdb'
    )
    with open(relaxed_output_path, 'w') as fp:
        fp.write(relaxed_pdb_str)

    logger.info(f"Relaxed output written to {relaxed_output_path}...")