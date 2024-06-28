import json
import os
from pathlib import Path

import pandas as pd
import numpy as np

class EmbeddingSave:
    def __init__(self, config):
        self.config = config
        self._init_paths()
        self.save_config()

    def _init_paths(self):
        self.results_path = os.path.join('embeddings', f'{self.config["save_dir"]}')
        for modal in self.config['modalities']:
            Path(os.path.join(self.results_path, modal)).mkdir(parents=True, exist_ok=True)

    def __call__(self, embd, input_name, modality, flatten=True):
        embd_np = np.array(embd)
        if flatten:
            embd_np = embd_np.flatten()
        save_path = os.path.join(self.results_path, modality, f"{input_name}.npy")
        np.save(save_path, embd_np)

    def save_config(self):
        with open(os.path.join(self.results_path, 'config.yaml'), 'w') as f:
            json.dump(self.config, f)


    def save_label_space(self, embd, modality):
        save_path = os.path.join(self.results_path, modality, "labels.npy")
        if not os.path.exists(save_path):
            embd_np = np.array(embd)
            np.save(save_path, embd_np)