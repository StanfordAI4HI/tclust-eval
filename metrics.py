from munkres import Munkres, make_cost_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
from utils import *


def munkres_score(gt, pred):
    """
    :param gt: a list of lists, each containing ints
    :param pred: a list of lists, each containing ints
    :return: accuracy
    """

    # Combine all the sequences into one long sequence for both gt and pred
    gt_combined = np.concatenate(gt)
    pred_combined = np.concatenate(pred)

    # Make sure we're comparing the right shapes
    assert(gt_combined.shape == pred_combined.shape)

    # Build out the contingency matrix
    # This follows the methodology suggested by Zhou, De la Torre & Hodgkins, PAMI 2013.
    mat = contingency_matrix(gt_combined, pred_combined)

    # Make the cost matrix
    # Use the fact that no entry can exceed the total length of the sequence
    cost_mat = make_cost_matrix(mat, lambda x: gt_combined.shape[0] - x)

    # Apply the Munkres method (also called the Hungarian method) to find the optimal cluster correspondence
    m = Munkres()
    indexes = m.compute(cost_mat)

    # Pull out the associated 'costs' i.e. the cluster overlaps for the correspondences found
    cluster_overlaps = mat[list(zip(*indexes))]

    # Now compute the accuracy
    accuracy = np.sum(cluster_overlaps)/float(np.sum(mat))

    return accuracy


def purity_weights(gt, pred):
    # Build the contingency matrix
    cmat = contingency_matrix(gt, pred)

    # Find assignments based on a purity criteria
    # Maps clusters to gt-labels
    pure_assignments = np.argmax(cmat, axis=0)

    # A weight for each time-step (= 1 if cluster matches assigned gt-label otherwise = 0)
    return (gt == pure_assignments[pred]).astype(int)


def repeated_structure_score(gt, pred, similarity_fn, weight_fn=None, by_cluster=False):
    assert(len(gt) == len(pred))

    # First make sure we perform a label encoding to keep everything 0-indexed and contiguous
    gt = LabelEncoder().fit_transform(gt)
    pred = LabelEncoder().fit_transform(pred)

    # Determine the unique labels
    gt_clusters = np.unique(gt)
    pred_clusters = np.unique(pred)

    # Determine the length of the temporal clustering
    m = len(gt)

    # Find weights for each time-step based on the weight function
    weight_vec = weight_fn(gt, pred) if weight_fn is not None else np.ones(m)

    # Find the segment dictionary -- splits the predictions along the boundaries of the gt segments
    segment_dict = get_segment_dict(gt, pred, weight_vec)

    # Compute the metric
    metric = 0
    normalizer = 0
    per_gt_label_metrics, per_gt_label_normalizers = [], []
    for gt_cluster in gt_clusters:
        normalizer += len(segment_dict[gt_cluster]) * np.sum([b - a + 1 for a, b in segment_dict[gt_cluster]])
        per_gt_label_normalizers.append(len(segment_dict[gt_cluster]) * np.sum([b - a + 1 for a, b in segment_dict[gt_cluster]]))
        this_metric = 0.
        # Compare every pair of segments with the similarity function
        for s1, w1, _ in segment_dict[gt_cluster].values():
            for s2, w2, _ in segment_dict[gt_cluster].values():
                score = similarity_fn(s1, s2, w1, w2)
                metric += score
                this_metric += score
        per_gt_label_metrics.append(this_metric)

    normalizer *= 2.
    metric = metric / normalizer

    per_gt_label_metrics, per_gt_label_normalizers = np.array(per_gt_label_metrics), 2*np.array(per_gt_label_normalizers)
    per_gt_label_metrics /= per_gt_label_normalizers

    if not by_cluster:
        return metric
    else:
        return {gt_cluster: val for gt_cluster, val in zip(gt_clusters, per_gt_label_metrics)}


def compute_HSC_given_SG(gt, pred):
    segment_dict = get_segment_dict(gt, pred)

    unnormalized_score = 0.
    for cg in segment_dict:
        for a, b in segment_dict[cg]:
            segment_length = (b - a + 1)
            segment_prob = segment_length / float(len(gt))
            _, _, segment = segment_dict[cg][(a, b)]
            H_SC_given_SG = entropy(relabel_clustering(segment))

            unnormalized_score += segment_prob * H_SC_given_SG

    return unnormalized_score


def compute_HC_given_SG(gt, pred):
    segment_dict = get_segment_dict(gt, pred)

    unnormalized_score = 0.
    for cg in segment_dict:
        for a, b in segment_dict[cg]:
            segment_length = (b - a + 1)
            segment_prob = segment_length / float(len(gt))
            _, _, segment = segment_dict[cg][(a, b)]
            H_C_given_SG = entropy(segment)

            unnormalized_score += segment_prob * H_C_given_SG

    return unnormalized_score


def label_agnostic_oversegmentation_score(gt, pred):
    H_SC = entropy(relabel_clustering(pred))
    H_SC_given_SG = compute_HSC_given_SG(gt, pred)

    metric = H_SC_given_SG / H_SC if H_SC != 0 else 0.

    return max(1 - metric, 0.)


def label_agnostic_undersegmentation_score(gt, pred):
    return label_agnostic_oversegmentation_score(pred, gt)


def label_agnostic_segmentation_score(gt, pred):
    H_SC = entropy(relabel_clustering(pred))
    H_SG = entropy(relabel_clustering(gt))
    H_SC_given_SG = compute_HSC_given_SG(gt, pred)
    H_SG_given_SC = compute_HSC_given_SG(pred, gt)

    metric = (H_SC_given_SG + H_SG_given_SC) / (H_SG + H_SC) if (H_SC + H_SG) != 0 else 0.

    return max(1 - metric, 0.)


def segment_completeness_score(gt, pred):
    H_C = entropy(pred)
    H_C_given_SG = compute_HC_given_SG(gt, pred)

    metric = H_C_given_SG / (H_C) if H_C != 0 else 0.

    return max(1 - metric, 0.)


def segment_homogeneity_score(gt, pred):
    return segment_completeness_score(pred, gt)


def segment_structure_score(gt, pred):
    H_SC = entropy(relabel_clustering(pred))
    H_SG = entropy(relabel_clustering(gt))
    H_SC_given_SG = compute_HSC_given_SG(gt, pred)
    H_SG_given_SC = compute_HSC_given_SG(pred, gt)

    H_C = entropy(pred)
    H_C_given_SG = compute_HC_given_SG(gt, pred)
    H_G = entropy(gt)
    H_G_given_SC = compute_HC_given_SG(pred, gt)

    metric = (H_SC_given_SG + H_SG_given_SC + H_C_given_SG + H_G_given_SC) / \
             (H_SG + H_SC + H_C + H_G) if (H_SC + H_SG + H_C + H_G) != 0 else 0.

    return max(1 - metric, 0.)


def temporal_structure_score(gt, pred, similarity_fn, weight_fn=None, beta=1.0):
    sss = segment_structure_score(gt, pred)
    rss = repeated_structure_score(gt, pred, similarity_fn=similarity_fn, weight_fn=weight_fn)
    return ((1 + beta) * rss * sss)/(beta * rss + sss)


