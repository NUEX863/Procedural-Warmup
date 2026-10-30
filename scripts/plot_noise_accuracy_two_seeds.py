"""Paper accuracy figures: supplied CIFAR-100 values and saved STL-10 evaluations."""

import hashlib
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.figure_style import apply_paper_style, COLORS, LABELS, LINESTYLES

NOISES = [0, 10, 20, 30]
SEEDS = [11, 23]
FIGURES = ROOT / 'paper/figures'
SOURCE_IMAGE = 'C:/Users/18369/AppData/Local/Temp/codex-clipboard-2870e30c-536a-44c3-aeec-bc97024b0d77.png'


def source_data():
    # Transcribed accuracy columns; the displayed Difference column is not used.
    data = {
        'CIFAR-100': {
            'provenance': {'type': 'user_supplied_screenshot', 'path': SOURCE_IMAGE},
            'values': {
                '11': {'random': [68.245, 65.8842189, 63.52055, 60.026342],
                       'procedural': [72.5725, 70.275, 68.1245, 65.184758]},
                '23': {'random': [66.30225, 64.0168, 61.521548, 60.265455],
                       'procedural': [71.91247417, 70.3248484, 66.84543, 65.31785]},
            },
        },
        'STL-10': {'provenance': {}, 'values': {}},
    }
    for seed, relative in [(11, 'runs/stl10_author_noise_v1/evaluation/evaluation.json'),
                           (23, 'runs/stl10_author_seed23_v1/evaluation/evaluation.json')]:
        path = ROOT / relative
        report = json.loads(path.read_text(encoding='utf-8'))
        data['STL-10']['provenance'][str(seed)] = {
            'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        values = {}
        for initialization in ('random', 'procedural'):
            points = []
            for noise in NOISES:
                rows = [r for r in report['results'] if r['selection'] == 'epoch300'
                        and r['initialization'] == initialization and r['noise_percent'] == noise]
                assert len(rows) == 1 and rows[0]['epoch'] == 300
                row = rows[0]
                # Recover exact percentages from integer counts before display rounding.
                accuracy = Decimal(row['correct']) * 100 / Decimal(row['test_samples'])
                assert abs(float(accuracy) - row['accuracy_percent']) < 1e-10
                points.append(float(accuracy))
            values[initialization] = points
        data['STL-10']['values'][str(seed)] = values
    return {'selection': 'epoch300', 'accuracy_unit': 'percent',
            'noise_percent': NOISES, 'datasets': data}


def label(value):
    return str(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def main():
    font = apply_paper_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = source_data()
    source_path = FIGURES / 'noise_accuracy_epoch300_seeds11_23_data.json'
    source_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest_path = FIGURES / 'figure_manifest.json'
    registry = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'schema_version': 1, 'figures': []}
    for dataset, record in data['datasets'].items():
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=True, layout='constrained')
        fig.suptitle(f'{dataset}: test accuracy at epoch 300')
        annotations = []
        for ax, seed, panel in zip(axes, SEEDS, ['a', 'b']):
            values = record['values'][str(seed)]
            for initialization, marker in [('random', 'o'), ('procedural', 's')]:
                ys = values[initialization]
                assert len(ys) == 4 and all(0 <= y <= 100 for y in ys)
                ax.plot(NOISES, ys, marker=marker, markersize=3.8,
                        color=COLORS[initialization], linestyle=LINESTYLES[initialization],
                        label=LABELS[initialization], zorder=3)
                other = values['procedural' if initialization == 'random' else 'random']
                for x, y, other_y in zip(NOISES, ys, other):
                    upper = y > other_y
                    annotations.append(ax.annotate(label(y), (x, y),
                        xytext=(0, 7 if upper else -7), textcoords='offset points',
                        ha='center', va='bottom' if upper else 'top', fontsize=8,
                        color=COLORS[initialization], annotation_clip=False))
            ax.set_title(f'({panel}) Seed = {seed}')
            ax.set_xlabel('Training label noise (%)')
            ax.set_ylabel('Test accuracy (%)')
            ax.set_xticks(NOISES)
            ax.set_xlim(-3, 33)
            ax.set_ylim(45, 80)
            ax.set_yticks(range(45, 81, 5))
            ax.tick_params(axis='y', labelleft=True)
            ax.grid(axis='y', color='#D7D7D7', linewidth=0.5, alpha=0.8)
            ax.set_axisbelow(True)
            ax.legend(loc='lower left', frameon=False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for a in annotations:
            assert fig.bbox.contains(*a.get_window_extent(renderer).get_points()[0])
            assert fig.bbox.contains(*a.get_window_extent(renderer).get_points()[1])
        figure_id = f'{dataset.lower().replace("-", "")}_epoch300_noise_accuracy_seeds11_23_labeled'
        entry = dict(id=figure_id, font_family='Times New Roman', font_size_pt=8,
                     language='English', png_dpi=600, source_data=str(source_path),
                     selection='epoch300', seeds=SEEDS, y_axis_limits=[45, 80],
                     caption_english=f'{dataset} test accuracy after 300 training epochs at '
                     '0%, 10%, 20%, and 30% training label noise. Panels show seeds 11 and 23 '
                     'separately. Labels are rounded to two decimal places; plotted values retain '
                     'their source precision. Lines connect measured conditions; no seed averaging '
                     'or smoothing is applied. Both dataset figures use the same vertical scale.')
        for extension in ('svg', 'png'):
            path = FIGURES / f'{figure_id}.{extension}'
            fig.savefig(path, dpi=600)
            entry[extension] = str(path)
            entry[f'{extension}_relative'] = path.relative_to(ROOT).as_posix()
            entry[f'{extension}_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            print(path)
        plt.close(fig)
        registry['figures'] = [r for r in registry['figures'] if r['id'] != figure_id] + [entry]
    manifest_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Font verified: {font}')
    print('PAPER_FIGURES_COMPLETED: SVG + PNG; 8 pt; two-decimal labels; epoch 300 only.')


if __name__ == '__main__':
    main()
