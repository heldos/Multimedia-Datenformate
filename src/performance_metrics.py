import csv
import ast
import numpy as np

def compute_metrics_from_csv(csv_path):
    total_tp = total_fn = total_fp = total_tn = 0
    total_far = []
    total_frr = []
    total_eer = []
    total_erm = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            genuine_scores = ast.literal_eval(row[1])
            fake_scores = ast.literal_eval(row[2])
            all_scores = sorted(set(genuine_scores + fake_scores))

            min_erm = float('inf')
            best_threshold = None
            best_far = best_frr = None
            eer = None

            for threshold in all_scores:
                tp = sum(s >= threshold for s in genuine_scores)
                fn = sum(s < threshold for s in genuine_scores)
                fp = sum(s >= threshold for s in fake_scores)
                tn = sum(s < threshold for s in fake_scores)

                far = fp / (fp + tn) if (fp + tn) > 0 else 0
                frr = fn / (tp + fn) if (tp + fn) > 0 else 0
                erm = far + frr

                if eer is None or abs(far - frr) < abs(best_far - best_frr):
                    eer = (far + frr) / 2
                    best_far = far
                    best_frr = frr
                    best_threshold = threshold

                if erm < min_erm:
                    min_erm = erm

            # Confusion matrix at best threshold for this ID
            tp = sum(s >= best_threshold for s in genuine_scores)
            fn = sum(s < best_threshold for s in genuine_scores)
            fp = sum(s >= best_threshold for s in fake_scores)
            tn = sum(s < best_threshold for s in fake_scores)

            total_tp += tp
            total_fn += fn
            total_fp += fp
            total_tn += tn
            total_far.append(best_far)
            total_frr.append(best_frr)
            total_eer.append(eer)
            total_erm.append(min_erm)

            #print(f"ID: {row[0]}, Threshold: {best_threshold}, FAR: {best_far}, FRR: {best_frr}, EER: {eer}, ERM: {min_erm}")

    # Aggregate metrics
    confusion_matrix = {
        'TP': total_tp,
        'FN': total_fn,
        'FP': total_fp,
        'TN': total_tn
    }
    metrics = {
        'mean_FAR': np.mean(total_far),
        'mean_FRR': np.mean(total_frr),
        'mean_EER': np.mean(total_eer),
        'mean_ERM': np.mean(total_erm),
        'confusion_matrix': confusion_matrix
    }
    print(f"Aggregated Confusion Matrix: {confusion_matrix}")
    print(f"Mean FAR: {metrics['mean_FAR']}, Mean FRR: {metrics['mean_FRR']}, Mean EER: {metrics['mean_EER']}, Mean ERM: {metrics['mean_ERM']}")
    return metrics

# Example usage:
# metrics = compute_metrics_per_id_from_csv('output/baseline.csv')