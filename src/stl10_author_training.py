"""Author transforms and full-batch mixing with memory-limited accumulation.

Intentional adaptations: CPU mixing, microbatch forward/backward, per-sample
augmentation streams, epoch-shuffled sampling, and fixed-final test evaluation.
This is recipe alignment, not bitwise reproduction of an upstream training run.
"""

import contextlib
import hashlib
import itertools
import random
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import Dataset, DataLoader, Sampler

from src.model_factory import load_official_file, read_lock, official_utils
from src.training_core import ROOT, derived_seed, require


@contextlib.contextmanager
def cpu_random_stream(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def build_transforms(cfg):
    path = ROOT / read_lock()['code']['directory'] / 'downstream/datasets.py'
    module = load_official_file('_stl10_author_transforms', path)
    args = SimpleNamespace(**cfg)
    return module.build_transform(True, args), module.build_transform(False, args)


class AuthorDataset(Dataset):
    def __init__(self, images, labels, ids, transform, seed, training):
        # Shared raw uint8 storage avoids copying the dataset into Windows workers.
        self.images = torch.from_numpy(images).share_memory_()
        self.labels = np.asarray(labels, dtype=np.int64)
        self.ids = ids
        self.transform, self.seed, self.training = transform, seed, training
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index].numpy())
        if self.training:
            seed = derived_seed(self.seed, f'stl10-augmentation:{int(self.epoch)}:{self.ids[index]}')
            with cpu_random_stream(seed):
                image = self.transform(image)
        else:
            image = self.transform(image)
        return dict(image=image, label=int(self.labels[index]), source_index=index)


class FullBatchSampler(Sampler):
    def __init__(self, dataset, effective_batch):
        self.dataset = dataset
        self.count = len(dataset) // effective_batch * effective_batch

    def __iter__(self):
        # Matches the ordering rule of DistributedSampler with one replica and
        # set_epoch(epoch). Upstream non-distributed mode does not call set_epoch.
        generator = torch.Generator().manual_seed(self.dataset.seed + int(self.dataset.epoch))
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist()[:self.count])

    def __len__(self):
        return self.count


def make_loader(dataset, cfg, training):
    workers = cfg['num_workers'] if training else 0
    options = dict(dataset=dataset, batch_size=cfg['batch_size'] if training else cfg['eval_batch_size'],
                   sampler=FullBatchSampler(dataset, cfg['batch_size'] * cfg['accumulation_steps'])
                   if training else None, num_workers=workers, drop_last=training,
                   pin_memory=True, generator=torch.Generator().manual_seed(cfg['seed']))
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)


def make_mixup(cfg):
    return Mixup(mixup_alpha=cfg['mixup'], cutmix_alpha=cfg['cutmix'],
                 prob=cfg['mixup_prob'], switch_prob=cfg['mixup_switch_prob'],
                 mode=cfg['mixup_mode'], label_smoothing=cfg['smoothing'], num_classes=10)


def schedule(cfg, count):
    steps = count // (cfg['batch_size'] * cfg['accumulation_steps'])
    return official_utils().cosine_scheduler(cfg['lr'], cfg['min_lr'], cfg['epochs'],
                                            steps, warmup_epochs=cfg['warmup_epochs'])


def train_epoch(model, optimizer, scaler, loader, cfg, epoch, learning_rates, mixup):
    model.train()
    loader.dataset.epoch.fill_(epoch)
    micro = cfg['batch_size']
    accumulation = cfg['accumulation_steps']
    effective = micro * accumulation
    steps = len(loader.sampler) // effective
    iterator = iter(loader)
    criterion = SoftTargetCrossEntropy()
    loss_total = 0.0
    target_mass_total = 0.0
    applied = skipped = consecutive_skips = 0
    trace = hashlib.sha256()
    for step in range(steps):
        batches = list(itertools.islice(iterator, accumulation))
        require(len(batches) == accumulation, 'Incomplete effective training batch')
        images = torch.cat([b['image'] for b in batches])
        labels = torch.cat([b['label'] for b in batches])
        indices = torch.cat([b['source_index'] for b in batches])
        require(len(images) == effective, 'Unexpected effective batch size')
        del batches
        mixing_seed = derived_seed(cfg['seed'], f'stl10-mixing:{epoch}:{step}')
        with cpu_random_stream(mixing_seed):
            # Apply the installed timm implementation across all 512 samples.
            images, targets = mixup(images, labels)
        trace.update(indices.numpy().astype('<i8').tobytes())
        trace.update(targets.numpy().astype('<f4').tobytes())
        trace.update(str((epoch, mixing_seed)).encode())
        lr = float(learning_rates[epoch * steps + step])
        for group in optimizer.param_groups:
            group['lr'] = lr * group.get('lr_scale', 1.0)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, effective, micro):
            x = images[start:start + micro].cuda(non_blocking=True)
            y = targets[start:start + micro].cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
                logits = model(x)
                mean_loss = criterion(logits, y)
                loss = mean_loss / accumulation
            require(torch.isfinite(loss).item(), 'Non-finite soft-target training loss')
            scaler.scale(loss).backward()
            loss_total += float(mean_loss.detach()) * len(x)
            # Diagnostic probability mass, NOT ordinary clean-label accuracy.
            target_mass_total += float(y.gather(1, logits.argmax(1, keepdim=True)).sum())
            del x, y, logits, mean_loss, loss
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        overflow = scaler.get_scale() < old_scale
        skipped += int(overflow)
        applied += int(not overflow)
        consecutive_skips = consecutive_skips + 1 if overflow else 0
        require(consecutive_skips < 8, 'Persistent AMP overflow; stopped without overwriting last epoch')
        optimizer.zero_grad(set_to_none=True)
        del images, labels, targets, indices
        print(f'  update={step + 1}/{steps}, lr={lr:.8g}, AMP skipped={overflow}', flush=True)
    require(applied > 0, 'No optimizer update succeeded')
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), 'Non-finite model parameters')
    samples = steps * effective
    return dict(epoch=epoch + 1, samples=samples, omitted_samples=len(loader.dataset) - samples,
                loss=loss_total / samples, predicted_class_soft_target_mass=target_mass_total / samples,
                attempted_updates=steps, applied_updates=applied, amp_skipped_updates=skipped,
                last_lr=lr, pairing_sha256=trace.hexdigest())
