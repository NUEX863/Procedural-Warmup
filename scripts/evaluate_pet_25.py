"""Evaluate the eight fixed epoch-25 models; preserve full prediction evidence."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import contextlib
import datetime
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from evaluate_pet import sha256_file, write_csv, write_json
from run_pet_batch import JOBS, job_name, queue_lock, check_pairing
from run_pet_batch_25 import prepare_context, inspect_context, NEW_ROOT, Tee
from src.data_pipeline import build_dataset
from src.model_factory import build_model
from src.pet_test_metrics import classification_metrics, paired_accuracy, self_check
from src.training_core import read_checkpoint, read_config, setup, make_loader, require

OUTPUT = ROOT / 'runs/evaluation/oxford_pet_epoch025'


@torch.inference_mode()
def predict(model, loader, dataset):
    model.eval()
    all_logits = []
    seen = 0
    for batch in loader:
        end = seen + len(batch['label'])
        require(list(batch['sample_id']) == dataset.sample_ids[seen:end], 'Test image order mismatch')
        require(batch['source_index'].tolist() == dataset.indices[seen:end].tolist(), 'Source order mismatch')
        require(torch.equal(batch['label'], batch['clean_label'])
                and batch['label'].tolist() == dataset.clean_labels[seen:end].tolist()
                and not batch['is_corrupted'].any().item(), 'Test labels must be clean')
        with torch.autocast('cuda', enabled=False):
            logits = model(batch['image'].cuda(non_blocking=True).float())
        require(logits.shape == (end-seen, 37) and logits.dtype == torch.float32
                and torch.isfinite(logits).all().item(), 'Invalid logits')
        all_logits.append(logits.cpu().numpy())
        seen = end
        if seen % 1024 == 0 or seen == len(dataset):
            print(f'  Inference: {seen}/{len(dataset)}', flush=True)
    require(seen == 3669 and len(set(dataset.sample_ids)) == seen, 'Incomplete/duplicate test samples')
    require(all(p.grad is None for p in model.parameters()), 'Unexpected gradients')
    return np.concatenate(all_logits)


def run(output):
    self_check()
    cfg = read_config(ROOT / 'config/training.json')
    setup(cfg)
    states, contexts, hashes = {}, {}, {}
    print('Verifying eight epoch-25 models and epoch-24 backups ...', flush=True)
    for key in JOBS:
        context = prepare_context(*key)
        for filename in ('last.pt', 'previous.pt'):
            path = context['directory'] / filename
            hashes[str(path)] = sha256_file(path)
        state = inspect_context(context)
        require(state['status'] == 'complete' and state['completed_epochs'] == 25, 'Incomplete model')
        states[key], contexts[key] = state, context
        print(f'[PASS] {context["name"]}: last=25, previous=24; source/configuration/history verified', flush=True)
    check_pairing(states)
    test = build_dataset('oxford_pet', 'test', allow_test=True)
    require(len(test) == 3669 and test.num_classes == 37 and not test.is_corrupted.any(), 'Invalid test split')
    labels = test.clean_labels.copy()
    classes = list(test.base.classes)
    require(len(classes) == 37 and len(set(classes)) == 37, 'Invalid class names')
    # Verify class naming independently against each image's breed prefix.
    for sample_id, label in zip(test.sample_ids, labels):
        breed = Path(sample_id).stem.rsplit('_', 1)[0].replace('_', ' ').lower()
        require(breed == classes[int(label)].lower(), 'Breed name/label mismatch')
    loader = make_loader(test, cfg, training=False)
    results, summaries, predictions, per_class, calibration, confusions = {}, [], [], [], [], []
    sources = {str(Path(__file__).resolve()): sha256_file(__file__),
               str(ROOT/'src/pet_test_metrics.py'): sha256_file(ROOT/'src/pet_test_metrics.py')}
    for number, key in enumerate(JOBS, 1):
        noise, initialization = key
        context = contexts[key]
        path = context['directory'] / 'last.pt'
        require(sha256_file(path) == hashes[str(path)], 'Checkpoint changed before evaluation')
        payload = read_checkpoint(path)
        require(payload['next_epoch'] == 25 and len(payload['history']) == 25
                and payload['meta'] == context['meta'], 'Checkpoint identity mismatch')
        require(payload['meta']['manifests']['split_sha256'] == test.manifest_hashes['split_sha256'],
                'Training/test split provenance mismatch')
        trained = payload['model']
        model = build_model('random', num_classes=37, seed=11)
        model.load_state_dict(trained, strict=True)
        require(all(torch.equal(t, trained[k]) for k,t in model.state_dict().items()), 'Weight loading mismatch')
        print(f'\n[{number}/8] Evaluating {context["name"]} at epoch 25 ...', flush=True)
        logits = predict(model.cuda(), loader, test)
        model.cpu()
        require(all(torch.equal(t, trained[k]) for k,t in model.state_dict().items()), 'Model tensors changed')
        require(sha256_file(path) == hashes[str(path)], 'Checkpoint file changed')
        result = classification_metrics(logits, labels, classes)
        results[key] = result
        identity = dict(noise_percent=noise, initialization=initialization)
        summary = dict(dataset='oxford_pet', seed=11, epoch=25, **identity, **result['summary'],
                       checkpoint_path=str(path), checkpoint_sha256=hashes[str(path)],
                       noise_manifest_sha256=payload['meta']['manifests']['noise_sha256'])
        summaries.append(summary)
        stem = f'noise{noise:02d}_{initialization}'
        # Float32 logits are the original network output; probabilities and metrics
        # use a stable float64 softmax of those logits, without changing inference.
        np.savez_compressed(output / f'{stem}_predictions.npz', logits=logits,
                            probabilities=result['probabilities'], true_labels=labels,
                            sample_ids=np.asarray(test.sample_ids), class_names=np.asarray(classes))
        for i, sample_id in enumerate(test.sample_ids):
            predictions.append(dict(**identity, sample_id=sample_id, source_index=int(test.indices[i]),
                                    true_label=int(labels[i]), true_class=classes[labels[i]],
                                    predicted_label=int(result['predicted'][i]),
                                    predicted_class=classes[result['predicted'][i]],
                                    correct=int(result['correct'][i]), top5_correct=int(result['top5'][i]),
                                    confidence=float(result['confidence'][i]),
                                    negative_log_likelihood=float(result['nll'][i])))
        per_class.extend(dict(**identity, **row) for row in result['per_class'])
        calibration.extend(dict(**identity, **row) for row in result['calibration'])
        for c in range(37):
            for p in range(37):
                confusions.append(dict(**identity, true_class_id=c, predicted_class_id=p,
                                       count=int(result['confusion'][c,p])))
        print('[RESULT] ' + ' | '.join(f'{k}={v:.4f}' for k,v in result['summary'].items()), flush=True)
        del model, trained, payload, logits
        gc.collect()
        torch.cuda.empty_cache()
    comparisons = []
    delta_fields = ['accuracy_percent', 'top5_accuracy_percent', 'macro_precision_percent',
                    'macro_recall_percent', 'macro_f1_percent', 'weighted_f1_percent',
                    'negative_log_likelihood', 'brier_score', 'expected_calibration_error_percent',
                    'mean_confidence_percent']
    for noise in (0, 10, 20, 30):
        r, p = results[(noise,'random')], results[(noise,'procedural')]
        pair = paired_accuracy(r['correct'], p['correct'], labels, noise)
        for field in delta_fields:
            pair[f'random_{field}'] = r['summary'][field]
            pair[f'procedural_{field}'] = p['summary'][field]
            pair[f'procedural_minus_random_{field}'] = p['summary'][field] - r['summary'][field]
        comparisons.append(pair)
    for path, digest in hashes.items():
        require(sha256_file(path) == digest, f'Checkpoint changed: {path}')
    for path, digest in sources.items():
        require(sha256_file(path) == digest, f'Evaluation source changed: {path}')
    require(len(summaries) == 8 and len(predictions) == 29352, 'Incomplete evaluation')
    for filename, rows in [('summary.csv', summaries), ('comparison.csv', comparisons),
                            ('predictions.csv', predictions), ('per_class.csv', per_class),
                            ('calibration_bins.csv', calibration), ('confusion_matrices.csv', confusions),
                            ('class_mapping.csv', [dict(class_id=i,class_name=c) for i,c in enumerate(classes)])]:
        write_csv(output / filename, rows)
    report = dict(status='PASS', completed_at=datetime.datetime.now().astimezone().isoformat(),
                  checkpoint_epoch=25, seed=11, model_count=8, test_samples_per_model=3669,
                  model_selection='Fixed epoch 25 for every model; epoch 24 not evaluated',
                  training_schedule='Original 300-epoch schedule, 50 warmup epochs; stopped at epoch 25',
                  analysis_scope='Exploratory follow-up after previous test-set inspection; one training seed only',
                  uncertainty='Intervals describe test sampling conditional on these fitted models; not training-seed variability or simultaneous intervals',
                  precision='FP32 inference, no autocast/TF32; stable float64 probability/metric calculations',
                  preprocessing='Shared IMAGE_TRANSFORM: RGB, bilinear resize 224x224, tensor, ImageNet normalization',
                  evaluation_batch_size=cfg['eval_batch_size'], class_names=classes,
                  metric_definitions={
                      'accuracy': 'Top-1 correct / 3669; 95% Wilson score interval',
                      'top5_accuracy': 'True class among five highest logits; ties broken by ascending class index',
                      'macro_metrics': 'Unweighted mean over all 37 classes; undefined precision/F1 set to zero',
                      'balanced_accuracy': 'Macro recall (same quantity, not an independent additional measure)',
                      'weighted_f1': 'Per-class F1 weighted by true test support',
                      'negative_log_likelihood': 'Mean negative natural-log probability of true label, stable log-softmax; lower is better',
                      'brier_score': 'Mean sum across all 37 classes of squared probability minus one-hot target; range 0 to 2; lower is better',
                      'expected_calibration_error': '15 equal-width confidence bins [a,b), last includes 1; sum(count/N * abs(accuracy-mean confidence)); empty bins omitted; lower is better',
                      'paired_comparison': 'Procedural minus Random; percentage-valued score differences are percentage points',
                      'paired_accuracy_interval': '95% percentile paired bootstrap, stratified by true class, 10000 replicates; preserves test class counts and pairs; RNG SeedSequence([11,25,noise,20261027]) is analysis only'},
                  model_tensors_unchanged=True, checkpoint_files_unchanged=True,
                  checkpoint_hashes=hashes, source_sha256=sources,
                  test_manifest_sha256=test.manifest_hashes['split_sha256'],
                  training_runtime=contexts[(0,'random')]['meta']['runtime'],
                  results=summaries, comparisons=comparisons,
                  output_sha256={p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.suffix in ('.csv','.npz')})
    write_json(output/'evaluation.json', report)
    print('\nNoise  Initialization  Accuracy  Top-five  Macro precision  Macro recall  Macro F1  Weighted F1')
    for row in summaries:
        print(f"{row['noise_percent']:>3}%   {row['initialization']:<14}" + ''.join(
            f" {row[k]:>10.4f}" for k in ('accuracy_percent','top5_accuracy_percent','macro_precision_percent',
                                        'macro_recall_percent','macro_f1_percent','weighted_f1_percent')))
    print('\nNoise  Procedural minus Random accuracy (percentage points), paired 95% interval')
    for row in comparisons:
        print(f"{row['noise_percent']:>3}%  {row['accuracy_difference_percentage_points']:+.4f} "
              f"[{row['paired_bootstrap_low_percentage_points']:+.4f}, {row['paired_bootstrap_high_percentage_points']:+.4f}]")
    print('\nPET_25_TEST_EVALUATION_COMPLETED\nstatus=PASS\nmodels=8\nprediction_rows=29352')
    print('model_tensors_unchanged=True\ncheckpoint_files_unchanged=True')
    print(f'Results: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), f'Output exists; will not overwrite: {output}')
    with queue_lock():
        output.mkdir(parents=True, exist_ok=False)
        with (output/'console.log').open('x', encoding='utf-8', buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                run(output)


if __name__ == '__main__':
    main()
