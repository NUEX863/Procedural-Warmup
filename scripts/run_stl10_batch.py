"""Sequential local STL-10 training and final evaluation; no downloads."""

import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'config/stl10_training.json'
INITIALIZATIONS = ('random', 'procedural')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def atomic_torch_save(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def job_name(noise, initialization):
    return f'stl10_seed11_noise{noise:02d}_{initialization}'


def train_one(args, cfg):
    import numpy as np
    from src.stl10_hf_data import read_split, noisy_labels, STL10Dataset
    from src.training_core import (setup, make_loader, make_components, make_schedule,
                                   train_epoch, pack_checkpoint, save_checkpoint,
                                   read_checkpoint, restore_checkpoint, cpu_copy)

    setup(cfg)
    noise, initialization = args.worker_noise, args.initialization
    run_root = ROOT / cfg['output_directory']
    directory = run_root / job_name(noise, initialization)
    images, clean, ids = read_split(ROOT / cfg['data_directory'], 'train')
    observed = noisy_labels(clean, cfg['seed'], noise)
    dataset = STL10Dataset(images, clean, ids, np.arange(len(clean)), observed)
    # One shared, persistent label assignment for both initializations.
    label_path = run_root / 'labels' / f'noise{noise:02d}.npz'
    if label_path.exists():
        with np.load(label_path, allow_pickle=False) as saved:
            if not (np.array_equal(saved['ids'], ids)
                    and np.array_equal(saved['clean'], clean)
                    and np.array_equal(saved['observed'], observed)):
                raise RuntimeError('Saved shared label assignment differs from current data')
    else:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        temp = label_path.with_suffix('.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, ids=np.array(ids), clean=clean, observed=observed)
        os.replace(temp, label_path)

    sources = ['src/stl10_hf_data.py', 'src/model_factory.py', 'src/training_core.py',
               'src/data_pipeline.py', 'scripts/run_stl10_batch.py', 'config/upstream.lock.json']
    meta = dict(config=cfg, initialization=initialization, noise_percent=noise,
                dataset='stl10_hf', num_classes=10, training_samples=len(dataset),
                labels_sha256=dataset.fingerprint(),
                archive_inventory=[dict(name=p.name, size=p.stat().st_size)
                                   for p in sorted((ROOT / cfg['data_directory']).glob('train-*.tar'))],
                source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                               for p in sources})
    model, optimizer, scaler = make_components(cfg, initialization, 10)
    schedule = make_schedule(cfg, len(dataset))
    loader = make_loader(dataset, cfg)
    history, start, best = [], 0, None
    checkpoint_path = directory / ('previous.pt' if args.resume_previous else 'last.pt')
    if checkpoint_path.exists():
        payload = read_checkpoint(checkpoint_path)
        start, history = restore_checkpoint(payload, meta, model, optimizer, scaler)
        best = payload['best']
        del payload
        print(f'Resuming {directory.name}: {start}/{cfg["epochs"]}', flush=True)
    elif args.resume_previous:
        raise FileNotFoundError(checkpoint_path)

    print(f'Training images={len(dataset)}; corrupted labels={int((observed != clean).sum())}', flush=True)
    print('Best selection: minimum observed training loss; no validation or test selection', flush=True)
    for epoch in range(start, cfg['epochs']):
        print(f'\n{directory.name}: epoch={epoch + 1}/{cfg["epochs"]}', flush=True)
        row = train_epoch(model, optimizer, scaler, loader, cfg, epoch, schedule)
        history.append(row)
        if best is None or row['loss'] < best['selection_value']:
            best = dict(meta=meta, epoch=epoch + 1, selection='minimum_observed_training_loss',
                        selection_value=row['loss'], model=cpu_copy(model.state_dict()))
            atomic_torch_save(directory / 'best_model.pth', best)
        payload = pack_checkpoint(meta, model, optimizer, scaler, epoch + 1, history)
        payload['best'] = best
        save_checkpoint(directory, payload)
        del payload
        write_csv(directory / 'history.csv', history)
        print(json.dumps(row), flush=True)
        print(f'Saved: {directory / "last.pt"}; best epoch={best["epoch"]}', flush=True)

    # Also reconstruct exports when resuming an already completed run.
    atomic_torch_save(directory / 'best_model.pth', best)
    atomic_torch_save(directory / 'epoch300_model.pth',
                      dict(meta=meta, epoch=cfg['epochs'], model=cpu_copy(model.state_dict())))
    write_csv(directory / 'history.csv', history)
    print(f'TRAINING_COMPLETED: {directory.name}; test dataset not opened', flush=True)


def plot_results(rows, output, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.figure_style import apply_paper_style, COLORS, LABELS, LINESTYLES
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), layout='constrained')
    for ax, selection in zip(axes, ('epoch300', 'best')):
        for initialization in INITIALIZATIONS:
            points = sorted([r for r in rows if r['selection'] == selection
                             and r['initialization'] == initialization], key=lambda r: r['noise_percent'])
            ax.plot([r['noise_percent'] for r in points],
                    [r['accuracy_percent'] for r in points],
                    marker='o', color=COLORS[initialization], label=LABELS[initialization],
                    linestyle=LINESTYLES[initialization])
        ax.set_title('Epoch 300' if selection == 'epoch300' else 'Lowest training loss checkpoint')
        ax.set_xlabel('Training label noise (%)')
        ax.set_ylabel('Test accuracy (%)')
        ax.set_xticks(sorted({r['noise_percent'] for r in rows}))
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig_root = ROOT / 'paper/figures'
    fig_root.mkdir(parents=True, exist_ok=True)
    tag = '_'.join(f'{n:02d}' for n in sorted({r['noise_percent'] for r in rows}))
    figure_id = f'stl10_hf_pet300_v1_test_accuracy_noise{tag}'
    entry = dict(id=figure_id, font_family='Times New Roman', font_size_pt=8,
                 language='English', png_dpi=600, evaluation_report=str(output / 'evaluation.json'),
                 caption_english='CleanSTL-10 test accuracy for seed 11. Left: fixed epoch 300. '
                 'Right: checkpoint selected by lowest observed training loss. '
                 'Test labels remain clean. Lines connect measured conditions; '
                 'no test-based checkpoint selection. Official data equivalence was not checked.')
    for extension in ('svg', 'png'):
        path = fig_root / f'{figure_id}.{extension}'
        fig.savefig(path, dpi=600)
        entry[extension] = str(path)
        entry[f'{extension}_relative'] = path.relative_to(ROOT).as_posix()
    plt.close(fig)
    registry_path = fig_root / 'figure_manifest.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else dict(schema_version=1, figures=[])
    registry['figures'] = [f for f in registry['figures'] if f['id'] != figure_id] + [entry]
    write_json(registry_path, registry)
    print(f'Figures: {entry["svg"]}\n{entry["png"]}', flush=True)


def evaluate(args, cfg):
    import numpy as np
    import torch
    from src.training_core import setup, make_loader, read_checkpoint
    from src.stl10_hf_data import read_split, STL10Dataset, CLASSES
    from src.model_factory import build_model
    from src.pet_test_metrics import classification_metrics
    setup(cfg)
    run_root = ROOT / cfg['output_directory']
    # Do not open the test archives until all requested runs have finished.
    for noise in args.noises:
        for initialization in INITIALIZATIONS:
            path = run_root / job_name(noise, initialization) / 'epoch300_model.pth'
            if not path.exists():
                raise RuntimeError(f'Training not completed: {path}')
    images, labels, ids = read_split(ROOT / cfg['data_directory'], 'test')
    dataset = STL10Dataset(images, labels, ids, np.arange(len(labels)))
    loader = make_loader(dataset, cfg, training=False)
    output = run_root / 'evaluation'
    rows = []
    for noise in args.noises:
        for initialization in INITIALIZATIONS:
            directory = run_root / job_name(noise, initialization)
            for selection, filename in [('epoch300', 'epoch300_model.pth'), ('best', 'best_model.pth')]:
                payload = read_checkpoint(directory / filename)
                if payload['meta']['config'] != cfg:
                    raise RuntimeError(f'Checkpoint configuration mismatch: {directory}')
                model = build_model('random', 10, cfg['seed']).cuda()
                model.load_state_dict(payload['model'], strict=True)
                epoch = payload['epoch']
                del payload
                model.eval()
                logits = []
                with torch.inference_mode():
                    for batch in loader:
                        logits.append(model(batch['image'].cuda(non_blocking=True)).float().cpu().numpy())
                logits = np.concatenate(logits)
                result = classification_metrics(logits, labels, CLASSES)
                row = dict(noise_percent=noise, initialization=initialization, selection=selection,
                           epoch=epoch, **result['summary'])
                rows.append(row)
                target = output / directory.name / selection
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / 'predictions.npz', logits=logits, labels=labels,
                                    sample_ids=np.array(ids), predicted=result['predicted'])
                write_csv(target / 'predictions.csv', [dict(sample_id=ids[i], label=int(labels[i]),
                          predicted=int(result['predicted'][i]), confidence=float(result['confidence'][i]),
                          correct=bool(result['correct'][i])) for i in range(len(labels))])
                write_csv(target / 'per_class.csv', result['per_class'])
                np.savetxt(target / 'confusion_matrix.csv', result['confusion'], fmt='%d', delimiter=',')
                write_json(target / 'metrics.json', row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
                del model
                torch.cuda.empty_cache()
    # Retain earlier noise conditions if a later batch is evaluated in this directory.
    report_path = output / 'evaluation.json'
    if report_path.exists():
        old = json.loads(report_path.read_text(encoding='utf-8'))
        if old['config'] != cfg:
            raise RuntimeError('Existing evaluation belongs to another configuration')
        rows += [r for r in old['results'] if r['noise_percent'] not in args.noises]
    rows.sort(key=lambda r: (r['selection'], r['noise_percent'], r['initialization']))
    write_json(report_path, dict(config=cfg, results=rows, test_samples=len(labels),
                                official_data_equivalence='NOT_CHECKED'))
    write_csv(output / 'summary.csv', rows)
    print('\nSelection  Noise  Random (%)  Procedural (%)  Difference (pp)')
    lookup = {(r['selection'], r['noise_percent'], r['initialization']): r for r in rows}
    for selection in ('epoch300', 'best'):
        for noise in sorted({r['noise_percent'] for r in rows}):
            r = lookup[selection, noise, 'random']['accuracy_percent']
            p = lookup[selection, noise, 'procedural']['accuracy_percent']
            print(f'{selection:9} {noise:>3}%   {r:>9.4f}   {p:>13.4f}   {p-r:>+12.4f}')
    plot_results(rows, output, cfg)
    print(f'STL10_TEST_EVALUATION_COMPLETED\nResults: {output}', flush=True)


@contextlib.contextmanager
def queue_lock(run_root):
    import msvcrt
    import ctypes
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / 'batch.lock').open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        # Prevent idle sleep while the queue is active; restore on exit.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            yield
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def subprocess_logged(arguments, log_path):
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
            raise RuntimeError(f'Job stopped (exit={code}). See {log_path}. Rerun to resume.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noises', nargs='+', type=int, choices=[0, 10, 20, 30], default=[0, 10])
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--worker-noise', type=int, choices=[0, 10, 20, 30], help=argparse.SUPPRESS)
    parser.add_argument('--initialization', choices=INITIALIZATIONS, help=argparse.SUPPRESS)
    parser.add_argument('--resume-previous', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    run_root = ROOT / cfg['output_directory']
    if args.worker_noise is not None:
        if args.initialization is None:
            parser.error('--initialization is required for a training worker')
        from src.training_core import read_config
        train_one(args, read_config(CONFIG))
    elif args.evaluate_only:
        evaluate(args, cfg)
    elif args.status:
        import torch
        for noise in sorted(set(args.noises)):
            for initialization in INITIALIZATIONS:
                path = run_root / job_name(noise, initialization) / 'last.pt'
                if path.exists():
                    payload = torch.load(path, map_location='cpu', weights_only=False)
                    print(f'{path.parent.name}: {payload["next_epoch"]}/{cfg["epochs"]}; '
                          f'best epoch={payload["best"]["epoch"]}')
                    del payload
                else:
                    print(f'{path.parent.name}: not started')
    else:
        with queue_lock(run_root):
            for noise in sorted(set(args.noises)):
                for initialization in INITIALIZATIONS:
                    print(f'\n[RUN] {job_name(noise, initialization)}', flush=True)
                    subprocess_logged(['--worker-noise', str(noise), '--initialization', initialization],
                                      run_root / 'logs' / f'{job_name(noise, initialization)}.log')
            subprocess_logged(['--evaluate-only', '--noises', *map(str, sorted(set(args.noises)))],
                              run_root / 'logs/evaluation.log')
        print('STL10_BATCH_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
