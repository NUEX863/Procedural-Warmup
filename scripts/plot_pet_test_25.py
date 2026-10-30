"""Replot saved epoch-25 test results without opening images or loading models."""
import argparse
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from evaluate_pet import sha256_file, write_json
from plot_paper_pet_curves import check_figure_text
from src.figure_style import apply_paper_style, COLORS, LABELS, LINESTYLES
from src.pet_test_metrics import classification_metrics

NOISES = (0, 10, 20, 30)
INITIALIZATIONS = ('random', 'procedural')
OUT = ROOT / 'paper/figures'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_results(directory):
    report = json.loads((directory/'evaluation.json').read_text(encoding='utf-8'))
    require(report['status'] == 'PASS' and report['checkpoint_epoch'] == 25, 'Invalid evaluation')
    for name, digest in report['output_sha256'].items():
        require(sha256_file(directory/name) == digest, f'Evaluation file changed: {name}')
    results = {}
    for row in report['results']:
        key = row['noise_percent'], row['initialization']
        with np.load(directory/f'noise{key[0]:02d}_{key[1]}_predictions.npz', allow_pickle=False) as data:
            r = classification_metrics(data['logits'], data['true_labels'], data['class_names'].tolist())
            np.testing.assert_array_equal(r['probabilities'], data['probabilities'])
            for metric, value in r['summary'].items():
                require(value == row[metric], f'Saved summary mismatch: {key}, {metric}')
            results[key] = r
    return report, results


def legend(fig, y=0.96):
    handles = [Line2D([], [], color=COLORS[i], linestyle=LINESTYLES[i], marker='o' if i=='random' else 's',
                      markersize=3, label=LABELS[i]) for i in INITIALIZATIONS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,y), ncol=2, frameon=False)


def save(fig, suffix, caption, entries, report, report_path):
    # Matplotlib rasterizes continuous color bars by default; keep them vector.
    for axis in fig.axes:
        for collection in axis.collections:
            collection.set_rasterized(False)
    check_figure_text(fig)
    stem = 'pet_test_epoch025_' + suffix
    entry = dict(id=stem, checkpoint_epoch=25, figure_size_inches=fig.get_size_inches().tolist(),
                 caption_english=caption,
                 evaluation_report=str(report_path), evaluation_report_sha256=sha256_file(report_path),
                 generator='scripts/plot_pet_test_25.py', generator_sha256=sha256_file(__file__),
                 style_module='src/figure_style.py', style_sha256=sha256_file(ROOT/'src/figure_style.py'),
                 font_family='Times New Roman', font_size_pt=8, language='English', png_dpi=600)
    for extension in ('svg', 'png'):
        path = OUT/f'{stem}.{extension}'
        fig.savefig(path, dpi=600)
        entry[extension] = str(path)
        entry[extension+'_relative'] = path.relative_to(ROOT).as_posix()
        entry[extension+'_sha256'] = sha256_file(path)
    svg = ET.parse(entry['svg']).getroot()
    require(not svg.findall('.//{http://www.w3.org/2000/svg}image'), 'SVG contains raster elements')
    require('TimesNewRoman' in Path(entry['svg']).read_text(encoding='utf-8'), 'Times New Roman glyphs absent')
    plt.close(fig)
    entries.append(entry)
    print(f'[SAVED] {stem}.svg / .png', flush=True)


def main(directory):
    font_path = apply_paper_style()
    report, results = load_results(directory)
    report_path = directory/'evaluation.json'
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    prefix = ('Oxford-IIIT Pet official test set (3669 images, 37 classes), fixed epoch 25 and training seed 11. '
              'The original 300-epoch learning-rate schedule with 50 warmup epochs was retained. '
              'This is an exploratory follow-up after earlier test-set inspection; it does not quantify training-seed variability. ')

    def metric_panels(specs, suffix, caption, percentage=False):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6))
        fig.subplots_adjust(left=.09, right=.98, bottom=.11, top=.84, wspace=.32, hspace=.48)
        for j, (ax, (metric, title, ylabel)) in enumerate(zip(axes.flat, specs)):
            maximum = 0
            for initialization in INITIALIZATIONS:
                rows = [results[(noise,initialization)]['summary'] for noise in NOISES]
                values = np.array([r[metric] for r in rows])
                style = dict(color=COLORS[initialization], linestyle=LINESTYLES[initialization],
                             marker='o' if initialization=='random' else 's', markersize=3)
                if metric == 'accuracy_percent':
                    lows = np.array([r['accuracy_wilson_low_percent'] for r in rows])
                    highs = np.array([r['accuracy_wilson_high_percent'] for r in rows])
                    ax.errorbar(NOISES, values, yerr=[values-lows, highs-values], capsize=2, **style)
                    maximum = max(maximum, float(highs.max()))
                else:
                    ax.plot(NOISES, values, **style)
                    maximum = max(maximum, float(values.max()))
            ax.set_title(f'({chr(97+j)}) {title}')
            ax.set_xlabel('Training label noise (%)')
            ax.set_ylabel(ylabel)
            ax.set_xticks(NOISES)
            ax.set_xlim(-1,31)
            upper = maximum*1.15
            if percentage:
                upper = min(100, max(10, np.ceil(upper/10)*10))
            ax.set_ylim(0, upper)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis='y', color='#dddddd', linewidth=.45)
            ax.set_axisbelow(True)
        legend(fig)
        save(fig, suffix, prefix+caption, entries, report, report_path)

    metric_panels([
        ('accuracy_percent','Top-one accuracy','Test accuracy (%)'),
        ('top5_accuracy_percent','Top-five accuracy','Top-five accuracy (%)'),
        ('macro_f1_percent','Macro F1 score','Macro F1 score (%)'),
        ('balanced_accuracy_percent','Balanced accuracy','Balanced accuracy (%)')],
        'classification', 'Classification metrics by training label-noise rate. Top-one accuracy bars show pointwise 95% Wilson intervals. '
        'Balanced accuracy equals macro recall; macro metrics weight all 37 breeds equally. Lines connect the four measured conditions.', True)

    metric_panels([
        ('negative_log_likelihood','Negative log-likelihood','Negative log-likelihood'),
        ('brier_score','Multiclass Brier score','Brier score'),
        ('expected_calibration_error_percent','Expected calibration error','Expected calibration error (%)'),
        ('mean_confidence_percent','Mean predicted confidence','Mean predicted confidence (%)')],
        'probability_quality', 'Probability-quality metrics: lower is better for negative log-likelihood, Brier score and expected calibration error. '
        'The Brier score sums squared errors over all 37 classes (range 0 to 2). Calibration uses 15 equal-width confidence bins. '
        'Mean confidence alone is not a quality score and should be read with accuracy and calibration.')

    fig, ax = plt.subplots(figsize=(7.16,3.1))
    fig.subplots_adjust(left=.13, right=.97, bottom=.19, top=.88)
    pairs = report['comparisons']
    values = np.array([r['accuracy_difference_percentage_points'] for r in pairs])
    lo = np.array([r['paired_bootstrap_low_percentage_points'] for r in pairs])
    hi = np.array([r['paired_bootstrap_high_percentage_points'] for r in pairs])
    ax.axhline(0, color='#666666', linewidth=.7)
    ax.errorbar(NOISES, values, yerr=[values-lo,hi-values], fmt='o', color=COLORS['procedural'], capsize=4, markersize=4)
    span = max(float(hi.max()),0) - min(float(lo.min()),0)
    ax.set_ylim(min(float(lo.min()),0)-span*.15, max(float(hi.max()),0)+span*.22)
    ax.set_yticks(np.arange(np.ceil(ax.get_ylim()[0]), np.floor(ax.get_ylim()[1])+1))
    for noise, value, high in zip(NOISES, values, hi):
        ax.annotate(f'{value:+.2f}', (noise,high), xytext=(0,5), textcoords='offset points', ha='center')
    ax.set_xlim(-3,33)
    ax.set_xticks(NOISES)
    ax.set_xlabel('Training label noise (%)')
    ax.set_ylabel('Accuracy difference (percentage points)')
    ax.set_title('Procedural initialization minus random initialization')
    ax.grid(axis='y', color='#dddddd', linewidth=.45)
    save(fig, 'paired_accuracy_difference', prefix+'Accuracy differences use the same test images for both initializations. '
         'Bars are pointwise 95% class-stratified paired bootstrap percentile intervals (10000 replicates). '
         'Positive values favor procedural initialization. Intervals condition on the fitted models and the observed class counts; they are not simultaneous intervals.',
         entries, report, report_path)

    keys = [(n,i) for n in NOISES for i in INITIALIZATIONS]
    values = np.array([[row['recall_percent'] for row in results[key]['per_class']] for key in keys]).T
    fig, ax = plt.subplots(figsize=(7.16,6.6))
    fig.subplots_adjust(left=.28, right=.88, bottom=.18, top=.93)
    mesh = ax.pcolormesh(np.arange(9),np.arange(38),values, vmin=0,vmax=100,cmap='Blues', linewidth=.2,edgecolor='white')
    ax.set_ylim(37,0)
    ax.set_yticks(np.arange(37)+.5, report['class_names'])
    ax.set_xticks(np.arange(8)+.5, [f'{n}% {i.capitalize()}' for n,i in keys], rotation=45,ha='right')
    ax.set_xlabel('Training label noise and initialization')
    ax.set_title('Recall for each pet breed')
    for x in (2,4,6):
        ax.axvline(x, color='white',linewidth=1.5)
    for y in range(37):
        for x in range(8):
            ax.text(x+.5,y+.5,f'{values[y,x]:.0f}', ha='center',va='center',color='white' if values[y,x]>=60 else '#202020')
    cbax = fig.add_axes([.91,.31,.018,.43])
    fig.colorbar(mesh,cax=cbax).set_label('Class recall (%)')
    save(fig,'class_recall',prefix+'Per-breed recall on clean test labels; each row is a breed, and adjacent columns pair random and procedural initialization at the same noise rate. '
         'Cell labels round to whole percentages; exact values and test support are saved in per_class.csv.',entries,report,report_path)

    fig, axes = plt.subplots(2,2,figsize=(7.16,4.6))
    fig.subplots_adjust(left=.09,right=.98,bottom=.11,top=.84,wspace=.32,hspace=.48)
    for j,(ax,noise) in enumerate(zip(axes.flat,NOISES)):
        ax.plot([0,100],[0,100],color='#999999',linestyle=':',linewidth=.8)
        for initialization in INITIALIZATIONS:
            populated = [r for r in results[(noise,initialization)]['calibration'] if r['count']]
            ax.plot([100*r['mean_confidence'] for r in populated],[100*r['accuracy'] for r in populated],
                    color=COLORS[initialization],linestyle=LINESTYLES[initialization],
                    marker='o' if initialization=='random' else 's', markersize=3)
        ax.set(xlim=(0,100),ylim=(0,100),xlabel='Mean predicted confidence (%)',ylabel='Observed accuracy (%)',
               title=f'({chr(97+j)}) Training label noise: {noise}%')
        ax.set_xticks([0,25,50,75,100])
        ax.set_yticks([0,25,50,75,100])
        ax.grid(color='#dddddd',linewidth=.45)
    legend(fig)
    save(fig,'reliability',prefix+'Reliability diagrams with 15 equal-width confidence bins. The dotted diagonal indicates perfect calibration. '
         'Only nonempty bins are connected; each marker is a bin mean, not a single image. Bin counts are available in calibration_bins.csv; sparse bins may be noisy.',
         entries,report,report_path)

    for noise in NOISES:
        fig, axes = plt.subplots(1,2,figsize=(7.16,3.5))
        fig.subplots_adjust(left=.07,right=.88,bottom=.20,top=.83,wspace=.25)
        for ax, initialization in zip(axes, INITIALIZATIONS):
            cm = results[(noise,initialization)]['confusion']
            normalized = 100*cm/cm.sum(axis=1,keepdims=True)
            mesh = ax.pcolormesh(np.arange(38),np.arange(38),normalized,cmap='Blues',vmin=0,vmax=100)
            ax.set_aspect('equal')
            ax.set_ylim(37,0)
            ax.set_xticks(np.arange(0,37,6)+.5, np.arange(0,37,6))
            ax.set_yticks(np.arange(0,37,6)+.5, np.arange(0,37,6))
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('True class')
            ax.set_title(LABELS[initialization])
        fig.suptitle(f'Confusion matrices at {noise}% training label noise',y=.97)
        cbax=fig.add_axes([.91,.23,.018,.53])
        fig.colorbar(mesh,cax=cbax).set_label('Percentage within true class (%)')
        save(fig,f'confusion_noise{noise:02d}',prefix+f'Row-normalized confusion matrices for {noise}% training label noise. '
             'Rows are true classes; columns are predicted classes. The common color scale spans 0 to 100%. '
             'Class identifiers 0 through 36 are mapped to breed names in class_mapping.csv and follow the row order in the class-recall figure. '
             'All raw counts are saved in confusion_matrices.csv.',entries,report,report_path)

    registry = OUT/'figure_manifest.json'
    manifest = json.loads(registry.read_text(encoding='utf-8')) if registry.exists() else {'schema_version':1,'figures':[]}
    replaced = {row['id'] for row in entries}
    manifest['figures'] = [row for row in manifest['figures'] if row['id'] not in replaced] + entries
    manifest['pet_epoch025_evaluation'] = dict(report_path=str(report_path),report_sha256=sha256_file(report_path),
                                              font_path=font_path,figure_count=len(entries),formats=['svg','png'])
    write_json(registry,manifest)
    write_json(directory/'figures.json',dict(status='PASS',figures=entries,registry=str(registry)))
    print(f'PET_25_FIGURES_COMPLETED\nstatus=PASS\nfigure_pairs={len(entries)}\nOutput: {OUT}',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'runs/evaluation/oxford_pet_epoch025')
    main(parser.parse_args().results.resolve())
