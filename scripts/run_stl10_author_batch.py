"""Train the two seed-11, zero-noise STL-10 recipe-alignment baselines."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'config/stl10_author_baseline.json'
INITIALIZATIONS = ('random', 'procedural')

# Reuse only storage and queue-lock helpers; the old training path is untouched.
from scripts.run_stl10_batch import write_json, write_csv, atomic_torch_save, queue_lock


def name(initialization):
    return f'stl10_seed11_noise00_{initialization}'


def load_config():
    from src.training_core import read_config, require
    cfg = read_config(CONFIG)
    require(cfg['noise_percent'] == 0 and cfg['mixup_mode'] == 'batch',
            'This entry point is for the zero-noise, batch-mixing baseline only')
    require(cfg['best_selection'] == 'minimum_augmented_soft_target_training_loss',
            'Unexpected checkpoint selection criterion')
    return cfg


def train(initialization, cfg, resume_previous=False):
    import numpy as np
    import torch
    from src.stl10_hf_data import read_split
    from src.stl10_author_training import (build_transforms, AuthorDataset, make_loader,
                                           make_mixup, schedule, train_epoch)
    from src.training_core import (setup, make_components, pack_checkpoint, save_checkpoint,
                                   read_checkpoint, restore_checkpoint, cpu_copy, require)
    from src.model_factory import read_lock

    setup(cfg)
    train_transform, _ = build_transforms(cfg)
    images, labels, ids = read_split(ROOT / cfg['data_directory'], 'train')
    require(len(labels) == 5000, 'Expected 5000 labeled training samples')
    dataset = AuthorDataset(images, labels, ids, train_transform, cfg['seed'], True)
    del images
    dataset_digest = hashlib.sha256('\n'.join(ids).encode() + labels.astype('<i8').tobytes()).hexdigest()
    sources = ['src/model_factory.py', 'src/training_core.py', 'src/stl10_hf_data.py',
               'src/data_pipeline.py', 'src/stl10_author_training.py',
               'scripts/run_stl10_batch.py', 'scripts/run_stl10_author_batch.py']
    meta = dict(config=cfg, initialization=initialization, num_classes=10,
                dataset='stl10_hf', training_samples=len(dataset), dataset_ids_labels_sha256=dataset_digest,
                upstream=read_lock(), torch=str(torch.__version__),
                archive_inventory=[dict(name=p.name, size=p.stat().st_size)
                                   for p in sorted((ROOT / cfg['data_directory']).glob('train-*.tar'))],
                source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources})
    directory = ROOT / cfg['output_directory'] / name(initialization)
    checkpoint_path = directory / ('previous.pt' if resume_previous else 'last.pt')
    if resume_previous and not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model, optimizer, scaler = make_components(cfg, initialization, 10)
    rates = schedule(cfg, len(dataset))
    loader = make_loader(dataset, cfg, training=True)
    mixup = make_mixup(cfg)
    start, history, best = 0, [], None
    if checkpoint_path.exists():
        payload = read_checkpoint(checkpoint_path)
        start, history = restore_checkpoint(payload, meta, model, optimizer, scaler)
        best = payload['best']
        del payload
        atomic_torch_save(directory / 'best_model.pth', best)
    print(f'{name(initialization)}: completed={start}/300; full-batch updates per epoch=9', flush=True)
    print(train_transform, flush=True)
    print('Mixing: full 512-sample CPU batch; CUDA microbatch=32; accumulation=16', flush=True)
    print('Best: minimum augmented soft-target training loss. Test set closed during training.', flush=True)
    for epoch in range(start, cfg['epochs']):
        print(f'\n{name(initialization)} / epoch={epoch + 1}/300', flush=True)
        row = train_epoch(model, optimizer, scaler, loader, cfg, epoch, rates, mixup)
        history.append(row)
        if best is None or row['loss'] < best['selection_value']:
            best = dict(meta=meta, epoch=epoch + 1, selection=cfg['best_selection'],
                        selection_value=row['loss'], model=cpu_copy(model.state_dict()))
            atomic_torch_save(directory / 'best_model.pth', best)
        payload = pack_checkpoint(meta, model, optimizer, scaler, epoch + 1, history)
        payload['best'] = best
        save_checkpoint(directory, payload)
        del payload
        write_csv(directory / 'history.csv', history)
        print(json.dumps(row), flush=True)
        print(f'Saved {directory / "last.pt"}; best epoch={best["epoch"]}', flush=True)
    atomic_torch_save(directory / 'best_model.pth', best)
    atomic_torch_save(directory / 'epoch300_model.pth',
                      dict(meta=meta, epoch=cfg['epochs'], model=cpu_copy(model.state_dict())))
    write_csv(directory / 'history.csv', history)
    print(f'TRAINING_COMPLETED: {directory.name}', flush=True)


def plot(rows, cfg, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from src.figure_style import apply_paper_style, COLORS, LABELS, LINESTYLES

    apply_paper_style()
    fig_root = ROOT / 'paper/figures'
    fig_root.mkdir(parents=True, exist_ok=True)
    registry_path = fig_root / 'figure_manifest.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else dict(schema_version=1, figures=[])

    def save(fig, figure_id, caption):
        entry = dict(id=figure_id, font_family='Times New Roman', font_size_pt=8,
                     language='English', png_dpi=600, caption_english=caption,
                     evaluation_report=str(output / 'evaluation.json'))
        for extension in ('svg', 'png'):
            path = fig_root / f'{figure_id}.{extension}'
            fig.savefig(path, dpi=600)
            entry[extension] = str(path)
            entry[f'{extension}_relative'] = path.relative_to(ROOT).as_posix()
            entry[f'{extension}_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        plt.close(fig)
        registry['figures'] = [f for f in registry['figures'] if f['id'] != figure_id] + [entry]
        print(f'Figure: {entry["svg"]}\n{entry["png"]}', flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), layout='constrained')
    for ax, selection in zip(axes, ('epoch300', 'best')):
        points = [next(r for r in rows if r['selection'] == selection and r['initialization'] == init)
                  for init in INITIALIZATIONS]
        accuracy = np.array([r['accuracy_percent'] for r in points])
        errors = np.array([[r['accuracy_percent'] - r['accuracy_wilson_low_percent'] for r in points],
                           [r['accuracy_wilson_high_percent'] - r['accuracy_percent'] for r in points]])
        ax.bar([0, 1], accuracy, yerr=errors, capsize=3,
               color=[COLORS[i] for i in INITIALIZATIONS], width=0.55)
        ax.set_xticks([0, 1], ['Random', 'Procedural'])
        ax.set_xlabel('Initialization')
        ax.set_ylabel('Test accuracy (%)')
        ax.set_ylim(0, 100)
        ax.set_title('Epoch 300' if selection == 'epoch300' else 'Lowest training loss checkpoint')
        for x, row in enumerate(points):
            ax.text(x, row['accuracy_wilson_high_percent'] + 1.5,
                    f'{row["accuracy_percent"]:.2f}%', ha='center', va='bottom')
    save(fig, 'stl10_author_baseline_v1_test_accuracy',
         'CleanSTL-10, zero label noise, training seed 11, author-recipe alignment with CPU batch mixing '
         'and gradient accumulation. Error bars are pointwise 95% Wilson intervals over test images, '
         'not training-seed uncertainty. Best is selected by augmented training loss, never test accuracy. '
         'Official data equivalence was not checked.')
    fig, ax = plt.subplots(figsize=(7.16, 3.1), layout='constrained')
    for initialization in INITIALIZATIONS:
        path = ROOT / cfg['output_directory'] / name(initialization) / 'history.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            history = list(csv.DictReader(stream))
        ax.plot([int(r['epoch']) for r in history], [float(r['loss']) for r in history],
                label=LABELS[initialization], color=COLORS[initialization], linestyle=LINESTYLES[initialization])
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Soft-target cross-entropy loss')
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, 'stl10_author_baseline_v1_training_loss',
         'Augmented training soft-target cross-entropy over 300 epochs. Mixup, CutMix and label smoothing '
         'are active; these values are not directly comparable to the earlier hard-label losses. '
         'Raw per-epoch means without smoothing; seed 11 and zero label noise.')
    write_json(registry_path, registry)


def evaluate(cfg):
    import numpy as np
    import torch
    from src.training_core import setup, read_checkpoint, require
    from src.stl10_hf_data import read_split, CLASSES
    from src.stl10_author_training import build_transforms, AuthorDataset, make_loader
    from src.model_factory import build_model
    from src.pet_test_metrics import classification_metrics

    setup(cfg)
    run_root = ROOT / cfg['output_directory']
    for initialization in INITIALIZATIONS:
        if not (run_root / name(initialization) / 'epoch300_model.pth').exists():
            raise RuntimeError(f'Training not complete: {name(initialization)}')
    # Read only small training records to compare randomization before opening test.
    histories = []
    for initialization in INITIALIZATIONS:
        with (run_root / name(initialization) / 'history.csv').open(encoding='utf-8', newline='') as stream:
            histories.append(list(csv.DictReader(stream)))
    require(len(histories[0]) == len(histories[1]) == cfg['epochs'], 'Incomplete training histories')
    require([r['pairing_sha256'] for r in histories[0]] == [r['pairing_sha256'] for r in histories[1]],
            'The paired sample/mixed-target traces differ')
    _, transform = build_transforms(cfg)
    images, labels, ids = read_split(ROOT / cfg['data_directory'], 'test')
    require(len(labels) == 8000, 'Expected 8000 test samples')
    dataset = AuthorDataset(images, labels, ids, transform, cfg['seed'], False)
    del images
    loader = make_loader(dataset, cfg, training=False)
    output = run_root / 'evaluation'
    rows = []
    for initialization in INITIALIZATIONS:
        for selection, filename in [('epoch300', 'epoch300_model.pth'), ('best', 'best_model.pth')]:
            path = run_root / name(initialization) / filename
            checkpoint = read_checkpoint(path)
            require(checkpoint['meta']['config'] == cfg, 'Evaluation configuration differs from checkpoint')
            model = build_model('random', 10, cfg['seed']).cuda()
            model.load_state_dict(checkpoint['model'], strict=True)
            epoch = checkpoint['epoch']
            del checkpoint
            model.eval()
            outputs = []
            with torch.inference_mode():
                for batch in loader:
                    outputs.append(model(batch['image'].cuda(non_blocking=True)).float().cpu().numpy())
            logits = np.concatenate(outputs)
            result = classification_metrics(logits, labels, CLASSES)
            row = dict(initialization=initialization, noise_percent=0, selection=selection,
                       epoch=epoch, **result['summary'])
            rows.append(row)
            target = output / initialization / selection
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'predictions.npz', logits=logits, labels=labels,
                                sample_ids=np.array(ids), predicted=result['predicted'])
            write_csv(target / 'predictions.csv', [dict(sample_id=ids[i], label=int(labels[i]),
                      predicted=int(result['predicted'][i]), confidence=float(result['confidence'][i]),
                      correct=bool(result['correct'][i])) for i in range(len(labels))])
            write_csv(target / 'per_class.csv', result['per_class'])
            np.savetxt(target / 'confusion_matrix.csv', result['confusion'], fmt='%d', delimiter=',')
            write_json(target / 'metrics.json', row)
            print(json.dumps(row), flush=True)
            del model
            torch.cuda.empty_cache()
    write_json(output / 'evaluation.json', dict(config=cfg, results=rows, test_samples=len(labels),
                                               paired_randomization_traces_match=True,
                                               official_data_equivalence='NOT_CHECKED'))
    write_csv(output / 'summary.csv', rows)
    print('\nSelection   Random (%)   Procedural (%)   Difference (pp)')
    for selection in ('epoch300', 'best'):
        r, p = [next(row['accuracy_percent'] for row in rows
                     if row['selection'] == selection and row['initialization'] == init)
                for init in INITIALIZATIONS]
        print(f'{selection:10} {r:10.4f} {p:16.4f} {p-r:+17.4f}')
    plot(rows, cfg, output)
    print(f'BASELINE_TEST_EVALUATION_COMPLETED\nResults: {output}', flush=True)


def run_child(arguments, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    with log_path.open('a', encoding='utf-8') as log:
        with subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), *arguments],
                              cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', env=env) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise RuntimeError(f'Job stopped (exit={code}). See {log_path}; rerun to resume.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--worker', choices=INITIALIZATIONS, help=argparse.SUPPRESS)
    parser.add_argument('--resume-previous', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    run_root = ROOT / cfg['output_directory']
    if args.worker:
        train(args.worker, load_config(), args.resume_previous)
    elif args.evaluate_only:
        evaluate(load_config())
    elif args.status:
        import torch
        for initialization in INITIALIZATIONS:
            path = run_root / name(initialization) / 'last.pt'
            if not path.exists():
                print(f'{name(initialization)}: not started')
                continue
            payload = torch.load(path, map_location='cpu', weights_only=False)
            print(f'{name(initialization)}: {payload["next_epoch"]}/300; best epoch={payload["best"]["epoch"]}')
            del payload
    else:
        with queue_lock(run_root):
            for initialization in INITIALIZATIONS:
                run_child(['--worker', initialization], run_root / 'logs' / f'{initialization}.log')
            run_child(['--evaluate-only'], run_root / 'logs/evaluation.log')
        print('STL10_AUTHOR_BASELINE_BATCH_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
