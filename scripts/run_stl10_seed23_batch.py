"""Repeat the frozen STL-10 author-aligned recipe with seed 23 at 0/10/20/30% noise.

The zero-noise baseline and all existing source/configuration files stay intact.
Noise affects hard training labels BEFORE Mixup/CutMix and smoothing. Test labels
are never corrupted. Default execution trains eight new models and evaluates
them. Seed-11 results are kept separate and are never reused as seed-23 results.
"""

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
CONFIG = ROOT / 'config/stl10_author_seed23.json'
OUTPUT = 'runs/stl10_author_seed23_v1'
SEED = 23
INITIALIZATIONS = ('random', 'procedural')
NOISES = (0, 10, 20, 30)

from scripts.run_stl10_batch import write_json, write_csv, atomic_torch_save, queue_lock


def name(noise, initialization):
    return f'stl10_seed{SEED}_noise{noise:02d}_{initialization}'


def configuration(noise):
    from scripts.run_stl10_author_batch import load_config
    base = load_config()
    report_path = ROOT / base['output_directory'] / 'evaluation/evaluation.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['config'] != base:
        raise RuntimeError('Baseline configuration has changed since zero-noise evaluation')
    template = json.loads(CONFIG.read_text(encoding='utf-8'))
    if template['seed'] != SEED or template['output_directory'] != OUTPUT:
        raise RuntimeError('This entry point must use seed 23 and its separate output directory')
    if dict(template, seed=base['seed'], output_directory=base['output_directory']) != base:
        raise RuntimeError('Seed-23 hyperparameters must match the completed seed-11 recipe')
    if noise not in NOISES:
        raise ValueError(f'Unsupported noise rate: {noise}')
    cfg = dict(template, noise_percent=noise)
    return cfg, base, report


def shared_labels(clean, ids, noise, seed):
    import numpy as np
    from src.stl10_hf_data import noisy_labels
    observed = noisy_labels(clean, seed, noise)
    path = ROOT / OUTPUT / 'labels' / f'noise{noise:02d}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if not (np.array_equal(saved['sample_ids'], ids)
                    and np.array_equal(saved['clean_labels'], clean)
                    and np.array_equal(saved['observed_labels'], observed)):
                raise RuntimeError(f'Shared label assignment mismatch: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, sample_ids=np.array(ids), clean_labels=clean,
                                observed_labels=observed, corrupted=(observed != clean))
        os.replace(temporary, path)
    return observed


def train(noise, initialization, resume_previous=False):
    import numpy as np
    import torch
    from src.stl10_hf_data import read_split
    from src.stl10_author_training import (build_transforms, AuthorDataset, make_loader,
                                           make_mixup, schedule, train_epoch)
    from src.training_core import (setup, make_components, pack_checkpoint, save_checkpoint,
                                   read_checkpoint, restore_checkpoint, cpu_copy, require)
    from src.model_factory import read_lock

    cfg, base, _ = configuration(noise)
    setup(cfg)
    transform, _ = build_transforms(cfg)
    images, clean, ids = read_split(ROOT / cfg['data_directory'], 'train')
    require(len(clean) == 5000, 'Expected 5000 training images')
    observed = shared_labels(clean, ids, noise, cfg['seed'])
    dataset = AuthorDataset(images, observed, ids, transform, cfg['seed'], True)
    del images
    clean_digest = hashlib.sha256('\n'.join(ids).encode() + clean.astype('<i8').tobytes()).hexdigest()
    # Match the source IDs/clean labels actually used by the completed baseline.
    baseline_path = ROOT / base['output_directory'] / 'stl10_seed11_noise00_random' / 'epoch300_model.pth'
    baseline = read_checkpoint(baseline_path)
    require(baseline['meta']['config'] == base, 'Baseline checkpoint configuration mismatch')
    require(baseline['meta']['dataset_ids_labels_sha256'] == clean_digest,
            'Training sample IDs or clean labels differ from zero-noise baseline')
    sources = ['src/model_factory.py', 'src/training_core.py', 'src/stl10_hf_data.py',
               'src/data_pipeline.py', 'src/stl10_author_training.py',
               'scripts/run_stl10_batch.py', 'scripts/run_stl10_author_batch.py']
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    require(hashes == baseline['meta']['source_sha256'],
            'Shared recipe source changed since the baseline; do not silently mix protocols')
    inventory = [dict(name=p.name, size=p.stat().st_size)
                 for p in sorted((ROOT / cfg['data_directory']).glob('train-*.tar'))]
    require(inventory == baseline['meta']['archive_inventory'], 'Training archive inventory changed')
    del baseline
    hashes['scripts/run_stl10_seed23_batch.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta = dict(config=cfg, initialization=initialization, num_classes=10, dataset='stl10_hf',
                training_samples=len(dataset), dataset_ids_clean_labels_sha256=clean_digest,
                observed_labels_sha256=hashlib.sha256(observed.astype('<i8').tobytes()).hexdigest(),
                corrupted_count=int((observed != clean).sum()), upstream=read_lock(),
                torch=str(torch.__version__), archive_inventory=inventory, source_sha256=hashes)
    directory = ROOT / OUTPUT / name(noise, initialization)
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
    print(f'{directory.name}: completed={start}/300; corrupted={meta["corrupted_count"]}/5000', flush=True)
    print('Frozen baseline recipe: 512 effective batch; 32 x 16 accumulation; 9 updates/epoch', flush=True)
    print('Best uses augmented training loss. The test dataset remains closed during training.', flush=True)
    for epoch in range(start, cfg['epochs']):
        print(f'\n{directory.name} / epoch={epoch + 1}/300', flush=True)
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


def plot(rows, base, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.figure_style import apply_paper_style, COLORS, LABELS, LINESTYLES
    apply_paper_style()
    noises = sorted({r['noise_percent'] for r in rows})
    fig_root = ROOT / 'paper/figures'
    fig_root.mkdir(parents=True, exist_ok=True)
    registry_path = fig_root / 'figure_manifest.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else dict(schema_version=1, figures=[])
    tag = '_'.join(f'{n:02d}' for n in noises)

    def save(fig, suffix, caption):
        figure_id = f'stl10_author_seed23_v1_{suffix}_noise{tag}'
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

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), layout='constrained')
    for ax, selection in zip(axes, ('epoch300', 'best')):
        for initialization in INITIALIZATIONS:
            points = sorted([r for r in rows if r['selection'] == selection
                             and r['initialization'] == initialization], key=lambda r: r['noise_percent'])
            errors = [[r['accuracy_percent'] - r['accuracy_wilson_low_percent'] for r in points],
                      [r['accuracy_wilson_high_percent'] - r['accuracy_percent'] for r in points]]
            ax.errorbar([r['noise_percent'] for r in points], [r['accuracy_percent'] for r in points],
                        yerr=errors, capsize=2, marker='o', color=COLORS[initialization],
                        linestyle=LINESTYLES[initialization], label=LABELS[initialization])
        ax.set_title('Epoch 300' if selection == 'epoch300' else 'Lowest training loss checkpoint')
        ax.set_xlabel('Training label noise (%)')
        ax.set_ylabel('Test accuracy (%)')
        ax.set_xticks(noises)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
    axes[0].legend()
    save(fig, 'test_accuracy', 'CleanSTL-10 test accuracy, seed 23, fixed author-aligned recipe. '
         'All eight models are newly initialized for seed 23. Best is selected by augmented '
         'training loss, not test accuracy. Error bars are pointwise 95% Wilson intervals over test images, '
         'not training-seed variability. Lines connect measured conditions. Official data equivalence is unverified.')
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6), layout='constrained')
    for ax, noise in zip(axes.flat, (0, 10, 20, 30)):
        if noise not in noises:
            ax.set_visible(False)
            continue
        run_root = ROOT / OUTPUT
        for initialization in INITIALIZATIONS:
            with (run_root / name(noise, initialization) / 'history.csv').open(encoding='utf-8', newline='') as stream:
                history = list(csv.DictReader(stream))
            ax.plot([int(r['epoch']) for r in history], [float(r['loss']) for r in history],
                    label=LABELS[initialization], color=COLORS[initialization], linestyle=LINESTYLES[initialization])
        ax.set_title(f'Training label noise: {noise}%')
        ax.set_xlabel('Training epoch')
        ax.set_ylabel('Soft-target cross-entropy loss')
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    save(fig, 'training_loss', 'Seed 23: raw mean augmented soft-target training loss over 300 epochs. '
         'Mixup, CutMix and label smoothing are active at every noise condition. No curve smoothing. '
         'Hard-label corruption is applied before batch mixing; test labels remain clean.')
    write_json(registry_path, registry)


def evaluate(noises):
    import numpy as np
    import torch
    from src.training_core import setup, read_checkpoint, require
    from src.stl10_hf_data import read_split, CLASSES
    from src.stl10_author_training import build_transforms, AuthorDataset, make_loader
    from src.model_factory import build_model
    from src.pet_test_metrics import classification_metrics

    _, reference, _ = configuration(noises[0])
    base = json.loads(CONFIG.read_text(encoding='utf-8'))
    setup(base)
    run_root = ROOT / OUTPUT
    output = run_root / 'evaluation'
    rows = []
    for noise in noises:
        traces = []
        for initialization in INITIALIZATIONS:
            directory = run_root / name(noise, initialization)
            require((directory / 'epoch300_model.pth').exists(), f'Training incomplete: {directory.name}')
            with (directory / 'history.csv').open(encoding='utf-8', newline='') as stream:
                history = list(csv.DictReader(stream))
            require(len(history) == base['epochs'], 'Incomplete training history')
            traces.append([r['pairing_sha256'] for r in history])
        require(traces[0] == traces[1], f'Paired randomization mismatch at noise={noise}')
    _, transform = build_transforms(base)
    images, labels, ids = read_split(ROOT / base['data_directory'], 'test')
    require(len(labels) == 8000, 'Expected 8000 clean test samples')
    reference_path = ROOT / reference['output_directory'] / 'evaluation/random/epoch300/predictions.npz'
    with np.load(reference_path, allow_pickle=False) as reference_predictions:
        require(np.array_equal(reference_predictions['labels'], labels) and np.array_equal(reference_predictions['sample_ids'], ids),
                'Test labels/sample ordering differ from the baseline')
    dataset = AuthorDataset(images, labels, ids, transform, base['seed'], False)
    del images
    loader = make_loader(dataset, base, training=False)
    for noise in noises:
        cfg, _, _ = configuration(noise)
        for initialization in INITIALIZATIONS:
            for selection, filename in [('epoch300', 'epoch300_model.pth'), ('best', 'best_model.pth')]:
                path = run_root / name(noise, initialization) / filename
                checkpoint = read_checkpoint(path)
                require(checkpoint['meta']['config'] == cfg, 'Evaluation/checkpoint configuration mismatch')
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
                row = dict(initialization=initialization, noise_percent=noise, selection=selection,
                           epoch=epoch, **result['summary'])
                rows.append(row)
                target = output / name(noise, initialization) / selection
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
    report_path = output / 'evaluation.json'
    if report_path.exists():
        old = json.loads(report_path.read_text(encoding='utf-8'))
        require(old['config'] == base, 'Existing evaluation uses another seed or recipe')
        rows += [r for r in old['results'] if r['noise_percent'] not in noises]
    rows.sort(key=lambda r: (r['selection'], r['noise_percent'], r['initialization']))
    write_json(report_path, dict(config=base, seed=SEED, results=rows, test_samples=len(labels),
                                 reference_seed11_report=str(ROOT / reference['output_directory'] / 'evaluation/evaluation.json'),
                                 paired_randomization_traces_match=True, official_data_equivalence='NOT_CHECKED'))
    write_csv(output / 'summary.csv', rows)
    print('\nSelection  Noise  Random (%)  Procedural (%)  Difference (pp)')
    for selection in ('epoch300', 'best'):
        for noise in sorted({r['noise_percent'] for r in rows}):
            r, p = [next(row['accuracy_percent'] for row in rows if row['selection'] == selection
                         and row['noise_percent'] == noise and row['initialization'] == init)
                    for init in INITIALIZATIONS]
            print(f'{selection:9} {noise:3}%  {r:10.4f}  {p:14.4f}  {p-r:+15.4f}')
    plot(rows, base, output)
    print(f'STL10_SEED23_EVALUATION_COMPLETED\nResults: {output}', flush=True)


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
    parser.add_argument('--noises', nargs='+', type=int, choices=NOISES, default=list(NOISES))
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--worker', choices=INITIALIZATIONS, help=argparse.SUPPRESS)
    parser.add_argument('--noise', type=int, choices=NOISES, help=argparse.SUPPRESS)
    parser.add_argument('--resume-previous', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    noises = sorted(set(args.noises))
    run_root = ROOT / OUTPUT
    if args.worker:
        if args.noise is None:
            parser.error('--noise is required for a training worker')
        train(args.noise, args.worker, args.resume_previous)
    elif args.evaluate_only:
        evaluate(noises)
    elif args.status:
        import torch
        for noise in noises:
            for initialization in INITIALIZATIONS:
                path = run_root / name(noise, initialization) / 'last.pt'
                if not path.exists():
                    print(f'{path.parent.name}: not started')
                    continue
                payload = torch.load(path, map_location='cpu', weights_only=False)
                print(f'{path.parent.name}: {payload["next_epoch"]}/300; best epoch={payload["best"]["epoch"]}')
                del payload
    else:
        configuration(noises[0])
        with queue_lock(run_root):
            for noise in noises:
                for initialization in INITIALIZATIONS:
                    run_child(['--worker', initialization, '--noise', str(noise)],
                              run_root / 'logs' / f'{name(noise, initialization)}.log')
            run_child(['--evaluate-only', '--noises', *map(str, noises)], run_root / 'logs/evaluation.log')
        print('STL10_SEED23_BATCH_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
