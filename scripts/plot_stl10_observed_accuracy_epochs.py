"""Plot measured STL-10 checkpoint accuracies without inventing epoch histories."""
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
from src.figure_style import apply_paper_style, COLORS, LABELS


def main():
    apply_paper_style()
    out = ROOT / 'paper/figures'
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'figure_manifest.json'
    registry = json.loads(manifest.read_text(encoding='utf-8'))
    for seed, folder in [(11, 'stl10_author_noise_v1'), (23, 'stl10_author_seed23_v1')]:
        source = ROOT / 'runs' / folder / 'evaluation/evaluation.json'
        report = json.loads(source.read_text(encoding='utf-8'))
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))
        fig.subplots_adjust(left=.085, right=.96, bottom=.14, top=.89, wspace=.28, hspace=.42)
        fig.suptitle(f'STL-10: measured checkpoint test accuracy (seed = {seed})', y=.97)
        fig.text(.5, .025, 'Only evaluated checkpoints are shown; intermediate epoch accuracies were not recorded.',
                 ha='center', fontsize=8)
        all_annotations = []
        for ax, noise, panel in zip(axes.flat, [0, 10, 20, 30], 'abcd'):
            for init, marker in [('random', 'o'), ('procedural', 's')]:
                rows = sorted([r for r in report['results'] if r['initialization'] == init
                               and r['noise_percent'] == noise], key=lambda r: r['epoch'])
                assert len(rows) == 2 and {r['selection'] for r in rows} == {'best', 'epoch300'}
                xs = [r['epoch'] for r in rows]
                ys = [float(Decimal(r['correct']) * 100 / Decimal(r['test_samples'])) for r in rows]
                ax.scatter(xs, ys, marker=marker, s=18, color=COLORS[init], label=LABELS[init], zorder=3)
                for row, x, y in zip(rows, xs, ys):
                    upper = init == 'procedural'
                    final = row['selection'] == 'epoch300'
                    value = Decimal(row['correct']) * 100 / Decimal(row['test_samples'])
                    text = str(value.quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
                    text += f'\n(epoch {x})'
                    all_annotations.append(ax.annotate(text, (x, y),
                        xytext=(4 if final else -4, 7 if upper else -7), textcoords='offset points',
                        ha='left' if final else 'right', va='bottom' if upper else 'top',
                        color=COLORS[init], fontsize=8, annotation_clip=False))
            ax.set_title(f'({panel}) Training label noise: {noise}%')
            ax.set_xlabel('Training epoch')
            ax.set_ylabel('Test accuracy (%)')
            ax.set_xlim(0, 350)
            ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
            ax.set_ylim(40, 80)
            ax.set_yticks([40, 50, 60, 70, 80])
            ax.grid(axis='y', alpha=.25, linewidth=.5)
            ax.set_axisbelow(True)
            ax.legend(loc='upper left', frameon=False)
        fig.canvas.draw()
        for a in all_annotations:
            box = a.get_window_extent(fig.canvas.get_renderer())
            assert all(fig.bbox.contains(*point) for point in box.get_points())
        figure_id = f'stl10_seed{seed}_observed_test_accuracy_by_epoch'
        entry = dict(id=figure_id, language='English', font_family='Times New Roman', font_size_pt=8,
                     png_dpi=600, seed=seed, evaluation_report=str(source),
                     source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     caption_english='STL-10 test accuracy at the evaluated checkpoints. Each panel '
                     'shows one training-label noise level. For each initialization, the checkpoint '
                     'with minimum augmented training loss and the fixed epoch-300 checkpoint are '
                     'shown at their actual epochs. Intermediate epoch accuracies were not recorded; '
                     'no interpolation, fitted curves, or synthetic values are displayed. All labels '
                     'report test accuracy in percent rounded to two decimal places.')
        for extension in ['svg', 'png']:
            path = out / f'{figure_id}.{extension}'
            fig.savefig(path, dpi=600)
            entry[extension] = str(path)
            entry[f'{extension}_relative'] = path.relative_to(ROOT).as_posix()
            entry[f'{extension}_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            print(path)
        plt.close(fig)
        registry['figures'] = [r for r in registry['figures'] if r['id'] != figure_id] + [entry]
    manifest.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('MEASURED_CHECKPOINT_FIGURES_SAVED')


if __name__ == '__main__':
    main()
