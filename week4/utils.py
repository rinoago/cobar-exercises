import numpy as np
from flygym.mujoco.util import get_data_path, load_config


def get_ommatidia_id_matrix():
    data_dir = get_data_path("flygym", "data")
    filename = load_config()["paths"]["ommatidia_id_map"]
    ommatidia_id_map = np.load(data_dir / filename)
    rows = [np.unique(row) for row in ommatidia_id_map]
    max_width = max(len(row) for row in rows)
    rows = np.array([row for row in rows if len(row) == max_width])[:, 1:] - 1
    cols = [np.unique(col) for col in rows.T]
    min_height = min(len(col) for col in cols)
    cols = [col[:min_height] for col in cols]
    rows = np.array(cols).T
    return rows


ommatidia_id_matrix = get_ommatidia_id_matrix()


def crop_hex_to_rect(visual_input):
    return visual_input[..., ommatidia_id_matrix, :].max(-1)
