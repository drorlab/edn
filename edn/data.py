import os
import torch

import numpy as np
import pandas as pd
import random
import torch_geometric as tg

import e3nn.point.data_helpers as dh


element_mapping = {
    'C': 0,
    'O': 1,
    'N': 2,
    'S': 3
}


class EDN_Transform:
    def __init__(self, use_labels, num_nearest_neighbors, **kwargs):
        self.use_labels = use_labels
        self.num_nearest_neighbors = num_nearest_neighbors

    def __repr__(self):
        return f"edn-{self.num_nearest_neighbors}"

    __str__ = __repr__

    def __call__(self, item):
        if not self.use_labels:
            # Don't use any label
            item['label'] = 0
        else:
            item['label'] = item['scores']['gdt_ts']

        ### Atom coords, elements, flags, and one-hot encodings
        elements = item['atoms']['element'].to_numpy()
        coords = item['atoms'][['x', 'y', 'z']].to_numpy()
        c_alpha_flags = (item['atoms']['name'] == 'CA').to_numpy()

        # Filter out elements not in mapping
        sel = np.isin(elements, list(element_mapping.keys()))
        coords = coords[sel]
        elements = elements[sel]
        c_alpha_flags = c_alpha_flags[sel]

        # Make one-hot
        elements_int = np.array([element_mapping[e] for e in elements])
        one_hot = np.zeros((elements.shape[0], len(element_mapping)))
        one_hot[np.arange(elements.shape[0]), elements_int] = 1

        geometry = torch.tensor(coords, dtype=torch.float32)
        features = torch.tensor(one_hot, dtype=torch.float32)
        label = torch.tensor([item['label']], dtype=torch.float32)

        # Figure out the neighbors
        ra = geometry.unsqueeze(0)
        rb = geometry.unsqueeze(1)
        pdist = (ra - rb).norm(dim=2)
        tmp = torch.topk(-pdist, min(self.num_nearest_neighbors, pdist.shape[0]), axis=1)

        nei_list = []
        geo_list = []
        for source, x in enumerate(tmp.indices):
            cart = geometry[x]
            nei_list.append(
                torch.tensor(
                    [[source, dest] for dest in x], dtype=torch.long))
            geo_list.append(cart - geometry[source])
        nei_list = torch.cat(nei_list, dim=0).transpose(1, 0)
        geo_list = torch.cat(geo_list, dim=0)

        data = tg.data.Data(
            x=features,
            edge_index=nei_list,
            edge_attr=geo_list,
            pos=geometry,
            Rs_in=[(len(element_mapping), 0)],
            label=label,
            id=item['id'],
            file_path=item['file_path'],
            select_ca=torch.tensor(c_alpha_flags),
            )
        return data


if __name__=="__main__":
    from atom3d.datasets import LMDBDataset
    import dotenv as de
    de.load_dotenv(de.find_dotenv(usecwd=True))

    dataset_path = os.path.join(os.environ['DATA_DIR'], 'test')
    print(f"Loading dataset from {dataset_path:}")
    dataset = LMDBDataset(
        dataset_path,
        transform=EDN_Transform(True, num_nearest_neighbors=40)
        )

    dataloader = tg.loader.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        sampler= None,
        )
    print(f"Finished creating dataloader of final size {len(dataloader):}")

    for i, batch in enumerate(dataloader):
        print(f'BATCH {i}: feature shape:', batch['x'].shape, end='')
        print(', label:', batch['label'])

