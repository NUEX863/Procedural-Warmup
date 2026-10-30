"""Read local WebDataset tar shards; never download or extract archives."""

import hashlib
import io
import re
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.data_pipeline import IMAGE_TRANSFORM
from src.training_core import derived_seed

CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse',
           'monkey', 'ship', 'truck']


def read_split(root, split):
    shards = sorted(Path(root).glob(f'{split}-*.tar'))
    if not shards:
        raise FileNotFoundError(f'No {split} archives in {root}')
    records = []
    for path in shards:
        samples = {}
        with tarfile.open(path, 'r|*') as archive:
            for member in archive:
                if not member.isfile() or '.' not in member.name:
                    continue
                key, extension = member.name.rsplit('.', 1)
                if extension not in ('png', 'cls'):
                    continue
                data = archive.extractfile(member).read()
                sample = samples.setdefault(key, {})
                if extension == 'cls':
                    sample['label'] = int(data.decode('utf-8').strip())
                else:
                    with Image.open(io.BytesIO(data)) as image:
                        sample['image'] = np.asarray(image.convert('RGB')).copy()
        for key, sample in samples.items():
            records.append((key, sample['image'], sample['label']))
    # Use stable sample IDs rather than archive traversal order.
    records.sort(key=lambda row: tuple(int(x) if x.isdigit() else x
                                      for x in re.split(r'(\d+)', row[0])))
    ids = [row[0] for row in records]
    if len(set(ids)) != len(ids):
        raise RuntimeError('Duplicate sample IDs across tar shards')
    images = np.stack([row[1] for row in records])
    labels = np.array([row[2] for row in records], dtype=np.int64)
    if images.shape[1:] != (96, 96, 3) or np.any((labels < 0) | (labels >= 10)):
        raise RuntimeError('Unexpected STL-10 image shape or label range')
    print(f'Loaded {split}: {len(labels)} images from {len(shards)} archives', flush=True)
    return images, labels, ids


def noisy_labels(clean, seed, percent):
    """Nested stratified corruption; wrong labels do not depend on noise level."""
    observed = clean.copy()
    class_indices = [np.flatnonzero(clean == c) for c in range(10)]
    quotas = np.array([len(indices) * percent / 100 for indices in class_indices])
    counts = np.floor(quotas).astype(int)
    remainder = round(len(clean) * percent / 100) - int(counts.sum())
    for category in np.argsort(-(quotas - counts), kind='stable')[:remainder]:
        counts[category] += 1
    for category, indices in enumerate(class_indices):
        rng = np.random.default_rng(derived_seed(seed, f'stl10-noise:{category}'))
        ordered = rng.permutation(indices)
        wrong = rng.integers(0, 9, size=len(ordered))
        wrong += wrong >= category
        selected = ordered[:counts[category]]
        observed[selected] = wrong[:counts[category]]
    return observed


class STL10Dataset(Dataset):
    def __init__(self, images, clean, ids, indices, labels=None):
        self.images = images
        self.indices = np.asarray(indices, dtype=np.int64)
        self.clean = clean[self.indices]
        self.labels = self.clean.copy() if labels is None else np.asarray(labels)
        self.ids = [ids[int(i)] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        return dict(image=IMAGE_TRANSFORM(Image.fromarray(self.images[index])),
                    label=int(self.labels[position]), clean_label=int(self.clean[position]),
                    sample_id=self.ids[position], source_index=index)

    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update('\n'.join(self.ids).encode())
        digest.update(self.indices.astype('<i8').tobytes())
        digest.update(self.clean.astype('<i8').tobytes())
        digest.update(self.labels.astype('<i8').tobytes())
        return digest.hexdigest()
