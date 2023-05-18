# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datetime
from multiprocessing import cpu_count
from typing import Mapping, Optional, Sequence, Any

import numpy as np


from model.np import residue_constants, protein

FeatureDict = Mapping[str, np.ndarray]









def make_sequence_features(
        sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype_onehot"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features




def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]]
        for i in range(len(aatype))
    ])


def expand_misspoints(protein_object: protein.Protein,):
    residue_index=protein_object.residue_index
    residue_index=residue_index-np.min(residue_index) + 0  #start from 0
    num_res=np.max(residue_index)+1

    newaatype=np.ones(num_res)*20
    for i in residue_index:
        newaatype[i]=protein_object.aatype[i]
    x=newaatype


def make_protein_features(
        protein_object: protein.Protein,
        description: str,
        _is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    # expand_misspoints(protein_object)
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["aatype"] = aatype
    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)
    pdb_feats["b_factors"] = protein_object.b_factors.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(
        1. if _is_distillation else 0.
    ).astype(np.float32)

    return pdb_feats


def make_pdb_features(
        protein_object: protein.Protein,
        description: str,
        is_distillation: bool = True,
        confidence_threshold: float = 50.,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, description, _is_distillation=True
    )

    if (is_distillation):
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats






class DataPipeline:
    """Assembles input features."""

    def __init__(
            self,):
        self.k=1



    def process_pdb(
            self,
            pdb_path: str,
            is_distillation: bool = False,
            chain_id: Optional[str] = None,
            _structure_index: Optional[str] = None,

    ) -> FeatureDict:
        """
            Assembles features for a protein in a PDB file.
        """
        if (_structure_index is not None):
            db_dir = os.path.dirname(pdb_path)
            db = _structure_index["db"]
            db_path = os.path.join(db_dir, db)
            fp = open(db_path, "rb")
            _, offset, length = _structure_index["files"][0]
            fp.seek(offset)
            pdb_str = fp.read(length).decode("utf-8")
            fp.close()
        else:
            with open(pdb_path, 'r') as f:
                pdb_str = f.read()

        #  replace some aa

        for search_text,replace_text in residue_constants.replace_aa.items() :
            pdb_str = pdb_str.replace(search_text, replace_text)
        protein_object = protein.from_pdb_string(pdb_str, chain_id)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype)
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        old_res=protein_object.residue_index
        pdb_feats = make_pdb_features(
            protein_object,
            description,
            is_distillation=is_distillation
        )




        return {**pdb_feats},old_res

