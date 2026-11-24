import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from tools.comprehensive.statistics import auc_by_draw, paired_intervals, sampling_weights


GOOD = ("Ring", "001", 0)
BAD = ("Ring", "001", 1)


def view(obj, label, score, view_id):
    return {"identity": obj, "view": view_id, "label": label, "score": score}


class BootstrapTests(unittest.TestCase):
    def test_normal_ng_same_name_and_view_cluster_with_mixed_mask_labels(self):
        rows = [view(GOOD, 0, .2, "000"), view(GOOD, 0, .3, "030"),
                view(BAD, 0, .2, "000"), view(BAD, 1, .3, "030")]
        draws = np.array([[1, 1], [3, 2]])
        output = auc_by_draw(rows, [GOOD, BAD], draws)
        for numbers, value in zip(draws, output, strict=True):
            expanded = [row for row in rows for _ in range(numbers[row["identity"][2]])]
            self.assertAlmostEqual(value, roc_auc_score([x["label"] for x in expanded], [x["score"] for x in expanded]))

    def test_strata_counts_and_seed_repeatability(self):
        positions = [GOOD, ("Ring", "004", 0), BAD, ("Ring", "005", 1), ("Ring", "006", 1)]
        draws = sampling_weights(positions, 2000, 42005)
        self.assertTrue(np.array_equal(draws, sampling_weights(positions, 2000, 42005)))
        self.assertTrue(np.all(draws[:, :2].sum(1) == 2))
        self.assertTrue(np.all(draws[:, 2:].sum(1) == 3))

    def test_same_method_reordered_has_zero_delta_and_interval(self):
        vals = [view(GOOD, 0, .2, "000"), view(BAD, 1, .6, "000")]
        data = {"views": vals, "objects": vals}
        reverse = {"views": vals[::-1], "objects": vals[::-1]}
        for info in paired_intervals(data, reverse, 100, 42005, 80, .95).values():
            self.assertEqual(info["delta_pp"], 0)
            self.assertEqual(info["lower_95_pp"], 0)
            self.assertEqual(info["upper_95_pp"], 0)

    def test_conflicting_mask_labels_are_rejected(self):
        vals = [view(GOOD, 0, .2, "000"), view(BAD, 1, .6, "000")]
        with self.assertRaisesRegex(ValueError, "identities differ"):
            paired_intervals({"views": vals, "objects": vals},
                             {"views": [vals[0], {**vals[1], "label": 0}], "objects": vals}, 100, 42005, 80, .95)
