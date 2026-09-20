import math
import unittest
from jev.calibration import TemperatureScaler, metrics, evaluate


class CalibrationTests(unittest.TestCase):
    def test_heldout_fit_reduces_overconfidence_without_changing_choices(self):
        fit = [[.9, .1]] * 10
        truth = [0] * 7 + [1] * 3
        scaler = TemperatureScaler().fit(fit, truth)
        scaled = scaler.transform(fit)
        self.assertGreater(scaler.temperature, 1.)
        self.assertLess(metrics(scaled, truth)['nll'], metrics(fit, truth)['nll'])
        self.assertLess(metrics(scaled, truth)['ece'], metrics(fit, truth)['ece'])
        self.assertTrue(all(p[0] > p[1] and math.isclose(sum(p), 1.) for p in scaled))

    def test_extreme_probabilities_stay_finite(self):
        out = TemperatureScaler(4.).transform([[0., 1.], [1e-300, 1.]])
        self.assertTrue(all(math.isfinite(v) for p in out for v in p))

    def test_invalid_data_and_temperature_fail(self):
        for t in [0, -1, math.inf, math.nan, True]:
            with self.subTest(t=t), self.assertRaises(ValueError):
                TemperatureScaler(t)
        for rows, labels in [([], []), ([[.5, .5]], []), ([[.5, .5]], [2]),
                             ([[.5, .5]], [True]), ([[.2, .2]], [0]),
                             ([[math.nan, 1.]], [0]), ([[-.1, 1.1]], [0])]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                TemperatureScaler().fit(rows, labels)
        with self.assertRaises(ValueError):
            TemperatureScaler().fit([[.5, .5]], [0], grid=[])

    def test_report_uses_disjoint_context_groups(self):
        fit = [dict(group='train', probabilities=[.9, .1], correct=i % 2) for i in range(10)]
        heldout = [dict(group='test', probabilities=[.9, .1], correct=i % 2) for i in range(10)]
        report = evaluate(fit, heldout)
        self.assertEqual(report['evaluation_count'], 10)
        self.assertFalse(report['applied_to_server'])
        with self.assertRaises(ValueError):
            evaluate(fit, fit)

    def test_metrics_match_known_distribution(self):
        m = metrics([[.75, .25], [.75, .25]], [0, 1])
        self.assertEqual(m['accuracy'], .5)
        self.assertAlmostEqual(m['ece'], .25)
        self.assertAlmostEqual(m['nll'], -(math.log(.75) + math.log(.25)) / 2)
