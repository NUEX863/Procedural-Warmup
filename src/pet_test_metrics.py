"""Metrics computed from saved, aligned single-label classification logits."""

import numpy as np
from scipy.special import logsumexp


def wilson_interval(correct, count):
    z = 1.959963984540054
    p = correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / denominator
    return 100 * (center - radius), 100 * (center + radius)


def classification_metrics(logits, labels, classes, bins=15):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n, k = logits.shape
    assert n == len(labels) and k == len(classes)
    assert np.isfinite(logits).all() and ((labels >= 0) & (labels < k)).all()
    log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
    probabilities = np.exp(log_probabilities)
    predicted = logits.argmax(axis=1)
    correct = predicted == labels
    confidence = probabilities.max(axis=1)
    ranking = np.argsort(-logits, axis=1, kind="stable")
    top5 = (ranking[:, :min(5, k)] == labels[:, None]).any(axis=1)
    confusion = np.bincount(k * labels + predicted, minlength=k*k).reshape(k, k)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = confusion.diagonal()
    precision = np.divide(true_positive, predicted_count, out=np.zeros(k), where=predicted_count > 0)
    recall = np.divide(true_positive, support, out=np.zeros(k), where=support > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(k), where=(precision+recall) > 0)
    low, high = wilson_interval(int(correct.sum()), n)
    calibration = []
    assignment = np.minimum((confidence*bins).astype(int), bins-1)
    ece = 0.0
    for b in range(bins):
        mask = assignment == b
        count = int(mask.sum())
        average_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count/n * abs(accuracy-average_confidence)
        calibration.append(dict(bin_index=b, lower_bound=b/bins, upper_bound=(b+1)/bins,
                                count=count, mean_confidence=average_confidence, accuracy=accuracy))
    nll = -log_probabilities[np.arange(n), labels]
    brier = (probabilities**2).sum(axis=1) - 2*probabilities[np.arange(n), labels] + 1
    summary = dict(test_samples=n, correct=int(correct.sum()),
                   accuracy_percent=100*float(correct.mean()),
                   accuracy_wilson_low_percent=low, accuracy_wilson_high_percent=high,
                   error_percent=100*float((~correct).mean()),
                   top5_accuracy_percent=100*float(top5.mean()),
                   macro_precision_percent=100*float(precision.mean()),
                   macro_recall_percent=100*float(recall.mean()),
                   balanced_accuracy_percent=100*float(recall.mean()),
                   macro_f1_percent=100*float(f1.mean()),
                   weighted_f1_percent=100*float(np.dot(f1, support)/n),
                   negative_log_likelihood=float(nll.mean()),
                   brier_score=float(brier.mean()),
                   expected_calibration_error_percent=100*ece,
                   mean_confidence_percent=100*float(confidence.mean()))
    per_class = [dict(class_id=c, class_name=classes[c], support=int(support[c]),
                      predicted_count=int(predicted_count[c]), true_positive=int(true_positive[c]),
                      precision_percent=100*float(precision[c]), recall_percent=100*float(recall[c]),
                      f1_percent=100*float(f1[c])) for c in range(k)]
    return dict(summary=summary, per_class=per_class, calibration=calibration,
                confusion=confusion, probabilities=probabilities, predicted=predicted,
                correct=correct, confidence=confidence, top5=top5, nll=nll)


def paired_accuracy(random_correct, procedural_correct, labels, noise, repeats=10000):
    # Class-stratified paired bootstrap. The four outcome counts are sufficient
    # statistics for resampling paired correctness, without allocating image arrays.
    rng = np.random.default_rng(np.random.SeedSequence([11, 25, noise, 20261027]))
    r = np.asarray(random_correct, dtype=bool)
    p = np.asarray(procedural_correct, dtype=bool)
    outcome = r.astype(int)*2 + p.astype(int)
    differences = np.zeros(repeats, dtype=np.int64)
    for c in np.unique(labels):
        counts = np.bincount(outcome[labels == c], minlength=4)
        draws = rng.multinomial(int(counts.sum()), counts/counts.sum(), size=repeats)
        differences += draws[:, 1] - draws[:, 2]
    low, high = np.quantile(100*differences/len(labels), [0.025, 0.975])
    return dict(noise_percent=noise, both_correct=int((r & p).sum()),
                random_only_correct=int((r & ~p).sum()),
                procedural_only_correct=int((~r & p).sum()), both_wrong=int((~r & ~p).sum()),
                accuracy_difference_percentage_points=100*float(p.mean()-r.mean()),
                paired_bootstrap_low_percentage_points=float(low),
                paired_bootstrap_high_percentage_points=float(high), bootstrap_replicates=repeats)


def self_check():
    # Independently known counts: confusion [[1,1,0],[0,2,0],[1,0,1]].
    y = np.array([0, 0, 1, 1, 2, 2])
    prediction = np.array([0, 1, 1, 1, 0, 2])
    logits = np.zeros((6, 3))
    logits[np.arange(6), prediction] = np.log(2)
    result = classification_metrics(logits, y, ['a', 'b', 'c'])
    s = result['summary']
    np.testing.assert_allclose(result['confusion'], [[1,1,0],[0,2,0],[1,0,1]])
    np.testing.assert_allclose(s['accuracy_percent'], 100*4/6)
    np.testing.assert_allclose(s['macro_precision_percent'], 100*(.5+2/3+1)/3)
    np.testing.assert_allclose(s['macro_f1_percent'], 100*(.5+.8+2/3)/3)
    np.testing.assert_allclose(s['negative_log_likelihood'], (4*np.log(2)+2*np.log(4))/6)
    np.testing.assert_allclose(s['brier_score'], (4*.375+2*.875)/6)
    np.testing.assert_allclose(s['expected_calibration_error_percent'], 100/6)
    assert sum(row['count'] for row in result['calibration']) == 6
    pair = paired_accuracy(result['correct'], result['correct'], y, 0, repeats=100)
    assert pair['accuracy_difference_percentage_points'] == 0
    assert pair['paired_bootstrap_low_percentage_points'] == 0
    assert pair['paired_bootstrap_high_percentage_points'] == 0
    uniform = classification_metrics(np.zeros((6, 3)), y, ['a', 'b', 'c'])
    np.testing.assert_allclose(uniform['summary']['negative_log_likelihood'], np.log(3))
    assert np.isfinite(list(uniform['summary'].values())).all()
    print('[PASS] Metrics: known confusion, precision, F1, likelihood, Brier, calibration and paired bootstrap')


if __name__ == '__main__':
    self_check()
