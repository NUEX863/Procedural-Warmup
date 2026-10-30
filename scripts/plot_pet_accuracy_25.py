"""Standalone clean-test accuracy versus training-label noise, from saved results."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate_pet import sha256_file, write_json
from plot_pet_test_25 import save
from src.figure_style import apply_paper_style, COLORS, LABELS, LINESTYLES


def main():
    apply_paper_style()
    directory = ROOT / 'runs/evaluation/oxford_pet_epoch025'
    report_path = directory / 'evaluation.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['status'] == 'PASS' and report['checkpoint_epoch'] == 25
    assert sha256_file(directory/'summary.csv') == report['output_sha256']['summary.csv']
    lookup = {(r['noise_percent'], r['initialization']): r for r in report['results']}
    noises = (0, 10, 20, 30)
    assert set(lookup) == {(n,i) for n in noises for i in COLORS}
    fig, ax = plt.subplots(figsize=(7.16, 3.6))
    fig.subplots_adjust(left=.10, right=.97, bottom=.17, top=.86)
    for initialization in ('random', 'procedural'):
        values = [lookup[(n,initialization)]['accuracy_percent'] for n in noises]
        ax.plot(noises, values, color=COLORS[initialization], linestyle=LINESTYLES[initialization],
                marker='o' if initialization=='random' else 's', markersize=4,
                label=LABELS[initialization])
        for noise, value in zip(noises, values):
            above = (initialization == 'procedural') if noise == 0 else (initialization == 'random')
            ax.annotate(f'{value:.2f}%', (noise,value), xytext=(0,9 if above else -10),
                        textcoords='offset points', ha='center', va='bottom' if above else 'top',
                        color=COLORS[initialization])
    ax.set(xlim=(-2,32), ylim=(0,12), xlabel='Training label noise (%)',
           ylabel='Test accuracy (%)', title='Oxford-IIIT Pet: epoch 25, seed 11')
    ax.set_xticks(noises)
    ax.set_yticks(range(0,13,2))
    ax.grid(axis='y', color='#dddddd', linewidth=.45)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=False)
    entries = []
    caption = (
        'Clean-test top-one accuracy versus training label-noise rate for random and procedural initialization '
        'on Oxford-IIIT Pet (3669 official test images, 37 classes). All models are evaluated at fixed epoch 25 '
        'with training seed 11. Noise is applied only to training labels; test labels remain clean. '
        'The original 300-epoch learning-rate schedule and 50-epoch warmup are retained. '
        'Markers are measured results; connecting lines are visual guides. Labels round to two decimals. '
        'For clarity, intervals are shown separately in pet_test_epoch025_paired_accuracy_difference.svg '
        'and saved in comparison.csv. This exploratory single-seed comparison does not establish '
        'training-seed robustness or directly measure memorization of corrupted training labels.'
    )
    save(fig, 'accuracy_vs_noise', caption, entries, report, report_path)
    entry = entries[0]
    entry['generator'] = 'scripts/plot_pet_accuracy_25.py'
    entry['generator_sha256'] = sha256_file(__file__)
    entry['helper_generator'] = 'scripts/plot_pet_test_25.py'
    entry['helper_generator_sha256'] = sha256_file(ROOT/entry['helper_generator'])
    registry = ROOT/'paper/figures/figure_manifest.json'
    manifest = json.loads(registry.read_text(encoding='utf-8'))
    manifest['figures'] = [f for f in manifest['figures'] if f['id'] != entry['id']] + [entry]
    evaluation_figures = [f for f in manifest['figures'] if f.get('evaluation_report') == str(report_path)]
    manifest['pet_epoch025_evaluation']['figure_count'] = len(evaluation_figures)
    write_json(registry, manifest)
    write_json(directory/'figures.json', dict(status='PASS', figures=evaluation_figures, registry=str(registry)))
    print('PET_25_ACCURACY_FIGURE_COMPLETED\nstatus=PASS\nTest inference was not rerun.')


if __name__ == '__main__':
    main()
