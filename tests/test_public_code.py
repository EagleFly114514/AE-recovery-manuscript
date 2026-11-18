# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Check source-only packaging and patient-grouped resampling without clinical data."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'code'))
from functional_baseline import strict_splits
from clinical import transition_table

_entry_spec = importlib.util.spec_from_file_location('ae_reproduce', ROOT / 'reproduce.py')
_entry = importlib.util.module_from_spec(_entry_spec)
_entry_spec.loader.exec_module(_entry)

class PublicCodeTests(unittest.TestCase):
    @staticmethod
    def input_bundle(path, collection=None):
        path.mkdir()
        names = ['cohort.csv', 'cohort_multiple_episode_sensitivity.csv',
                 'cohort_icu_sensitivity.csv', 'legacy_history.csv', 'clinical_spec.json']
        for name in names[:-1]:
            (path / name).write_text('synthetic\n', encoding='utf-8')
        (path / names[-1]).write_text(json.dumps({'screening': {}}), encoding='utf-8')
        if collection is not None:
            name = 'multicentre_collection.json'
            names.append(name)
            (path / name).write_text(json.dumps(collection), encoding='utf-8')
        manifest = {'files': [{'file': name, 'sha256': hashlib.sha256((path / name).read_bytes()).hexdigest()}
                              for name in names]}
        (path / 'input_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return path

    def test_distribution_contains_software_only(self):
        allowed_suffixes = {'.py', '.ps1', '.md', '.json', '.txt', '.sha256'}
        special = {'LICENSE', '.gitignore', '.gitattributes'}
        forbidden_folders = {'data', 'datasets', 'raw', 'results', 'reference_results', 'reference_figures',
                             'qa', 'manuscript', 'manuscript_source', 'figures', 'provenance'}
        for path in ROOT.rglob('*'):
            relative = path.relative_to(ROOT)
            if any(part in {'.git', '__pycache__', '.venv', '.pytest_cache'} for part in relative.parts):
                continue
            if path.is_file():
                self.assertTrue(relative.as_posix().isascii(), str(relative))
                self.assertFalse(forbidden_folders.intersection(relative.parts), str(relative))
                self.assertTrue(path.name in special or path.suffix in allowed_suffixes, str(relative))
                text = path.read_text(encoding='utf-8-sig')
                if path.suffix == '.json':
                    self.assertEqual(relative.parts[0], 'configs')
                if path.suffix in {'.py', '.ps1'}:
                    self.assertIn('SPDX-License-Identifier: MIT', text)
                if path.suffix == '.py':
                    ast.parse(text)
                self.assertNotRegex(text, r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
                self.assertNotRegex(text, r'gh[pousr]_[A-Za-z0-9]{30,}')
        self.assertTrue((ROOT / 'LICENSE').read_text(encoding='utf-8').startswith('MIT License'))

    def test_grouped_splits_keep_copies_of_a_person_together(self):
        labels = np.tile(np.arange(2), 24)
        groups = np.repeat(np.arange(labels.size), 2)
        y = np.repeat(labels, 2)
        splits = strict_splits(y, groups, folds=4, repeats=2, seed=13)
        coverage = np.zeros(y.size, dtype=int)
        for train, test in splits:
            self.assertEqual(np.intersect1d(groups[train], groups[test]).size, 0)
            self.assertEqual(set(y[train]), {0, 1})
            coverage[test] += 1
        np.testing.assert_array_equal(coverage, np.full(y.size, 2))

    def test_inconsistent_duplicates_are_rejected(self):
        labels = np.tile(np.arange(2), 24)
        groups = np.repeat(np.arange(labels.size), 2)
        y = np.repeat(labels, 2)
        y[1] = 1 - y[0]
        with self.assertRaises(ValueError):
            strict_splits(y, groups, folds=4, repeats=1, seed=13)

    def test_transition_denominator_is_calculated_from_input(self):
        rng = np.random.default_rng(31)
        frame = pd.DataFrame({name: rng.integers(0, 7, size=97).astype(float)
                              for name in ['mrs6', 'mrs12', 'mrs24']})
        frame.loc[::9, 'mrs24'] = np.nan
        matrix = transition_table(frame)
        total = matrix.loc[matrix['12-month status'].eq('Total'), 'Total'].iloc[0]
        self.assertEqual(total, frame.mrs6.between(3, 5).sum())

    def test_reusing_output_without_optional_input_removes_old_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            with_collection = self.input_bundle(base / 'with_collection', {'synthetic': 'first'})
            without_collection = self.input_bundle(base / 'without_collection')
            run = base / 'run'
            _entry.prepare(run, with_collection)
            staged = run / 'data/multicentre_collection.json'
            self.assertTrue(staged.is_file())
            unrelated = run / 'data/user_notes.txt'
            unrelated.write_text('preserve', encoding='utf-8')

            _entry.prepare(run, without_collection)
            self.assertFalse(staged.exists())
            self.assertEqual(unrelated.read_text(encoding='utf-8'), 'preserve')
            self.assertEqual((run / 'data/cohort.csv').read_bytes(), (without_collection / 'cohort.csv').read_bytes())

    def test_reusing_output_refreshes_optional_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            first = self.input_bundle(base / 'first', {'synthetic': 'first'})
            second = self.input_bundle(base / 'second', {'synthetic': 'second'})
            run = base / 'run'
            _entry.prepare(run, first)
            _entry.prepare(run, second)
            self.assertEqual((run / 'data/multicentre_collection.json').read_bytes(),
                             (second / 'multicentre_collection.json').read_bytes())

    def test_invalid_optional_input_is_rejected_before_replacing_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            first = self.input_bundle(base / 'first', {'synthetic': 'first'})
            second = self.input_bundle(base / 'second', {'synthetic': 'second'})
            run = base / 'run'
            _entry.prepare(run, first)
            (second / 'multicentre_collection.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Input manifest mismatch: multicentre_collection.json'):
                _entry.prepare(run, second)
            self.assertEqual((run / 'data/multicentre_collection.json').read_bytes(),
                             (first / 'multicentre_collection.json').read_bytes())

if __name__ == '__main__':
    unittest.main()
