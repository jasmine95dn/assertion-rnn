# --*-- coding: utf-8 --*--
"""
Source code based on `Alsentzer's preprocessing code <https://github.com/EmilyAlsentzer/clinicalBERT/blob/master/downstream_tasks/i2b2_preprocessing/i2b2_2010_relations/Reformat.ipynb>`__
"""
import os
import pickle
import re

import numpy as np


class Reformat:
    """
    Class Reformat combines raw text and annotated data together for later use
    Args:
        train_dir (str): directory of training data (labeled and raw text)
        test_dir (str): directory of labeled test data
        output_dir (str): directory of output reformatted data
        test_txt_dir (str): directory of raw text test data
        split_ratio (float): splitting ratio between training and development data

    Attributes:
        train_sub_dirs (list[str]): list of subdirectories in a training data directory
        train_dirs (list[str]): list of fullpath of subdirectories in a training data directory
        test_dir (str): directory of test data
        output_dir (str): directory of output data
        test_txt_dir (str): directory of raw text test data
        assertion_labels (set[str]): set of assertion labels
        split_ratio (float): splitting ratio between training and development data

    """
    def __init__(self, train_dir: str, test_dir: str, output_dir: str, test_txt_dir=None, split_ratio=0.9):
        
        train_dir = train_dir if train_dir.endswith('/') else train_dir + '/'
        self.train_sub_dirs = [subdir for subdir in os.listdir(train_dir) if os.path.isdir(subdir)]
        self.train_dirs = [os.path.join(train_dir, subdir) for subdir in self.train_sub_dirs]

        self.test_dir = test_dir if test_dir.endswith('/') else test_dir + '/'
        self.output_dir = output_dir

        self.test_txt_dir = test_txt_dir if test_txt_dir.endswith('/') else test_txt_dir + '/'

        self.assertion_labels = set()
        self.split_ratio = split_ratio

    def _assign_concept(self, concept_str: str) -> dict:
        """
        Takes string like
        'c="asymptomatic" 16:2 16:2||t="problem"'
        and returns dictionary like
        {'t': 'problem', 'start_line': 16, 'start_pos': 2, 'end_line': 16, 'end_pos': 2}
        
        Args:
            concept_str (str): concept annotation following format of 2010 i2b2/VA challenge

        Returns: 
            dict(str:str): a dictionary with attributes for a concept
        """
        try:
            position_bit, problem_bit = concept_str.split('||')
            t = problem_bit[3:-1]

            start_and_end_span = next(re.finditer(r'\s\d+:\d+\s\d+:\d+', concept_str)).span()
            c = concept_str[3:start_and_end_span[0] - 1]
            c = [y for y in c.split(' ') if y.strip() != '']
            c = ' '.join(c)

            start_and_end = concept_str[start_and_end_span[0] + 1: start_and_end_span[1]]
            start, end = start_and_end.split(' ')
            start_line, start_pos = [int(x) for x in start.split(':')]
            end_line, end_pos = [int(x) for x in end.split(':')]

            # Stupid and hacky!!!! This particular example raised a bug in my code below.
        #         if c == 'folate' and start_line == 43 and start_pos == 3 and end_line == 43 and end_pos == 3:
        #             start_pos, end_pos = 2, 2

        except:
            print(concept_str)
            raise

        return {
            't': t, 'start_line': start_line, 'start_pos': start_pos, 'end_line': end_line, 'end_pos': end_pos,
            'c': c,
        }

    def _assign_assertion(self, assertion_str: str) -> dict:
        """
        Takes string like
        'c="r thumb injury" 21:8 21:10||t="problem"||a="present"'
        and returns dictionary like
        {'a':'present', 't': 'problem', 'start_line': 21, 'start_pos': 8, 'end_line': 21, 'end_pos': 10}

        Args:
            assertion_str (str): assertion annotation following format of 2010 i2b2/VA challenge
        
        Returns: 
            dict(str:str): a dictionary with attributes for a problem concept
        """
        try:
            position_bit, problem_bit, assertion_bit = assertion_str.split('||')
            t = re.search(r'[a-z]{2,}', problem_bit).group()
            a = re.search(r'[a-z_]{2,}', assertion_bit).group()
            self.assertion_labels.add(a)

            start_and_end_span = next(re.finditer(r'\s\d+:\d+\s\d+:\d+', assertion_str)).span()
            c = assertion_str[3:start_and_end_span[0] - 1]
            c = [y for y in c.split(' ') if y.strip() != '']
            c = ' '.join(c)

            start_and_end = assertion_str[start_and_end_span[0] + 1: start_and_end_span[1]]
            start, end = start_and_end.split(' ')
            start_line, start_pos = [int(x) for x in start.split(':')]
            end_line, end_pos = [int(x) for x in end.split(':')]

        except:
            print(assertion_str)
            raise

        return {
            'a': a, 't': t, 'start_line': start_line, 'start_pos': start_pos, 'end_line': end_line, 'end_pos': end_pos,
            'c': c,
        }

    def _build_label_vocab(self, base_dirs: list) -> tuple:
        """
        Build label vocabulary for concept annotation

        Args:
            base_dirs (list[str]): list of base directories with data for building label vocabulary
        
        Returns:
            label_vocab (dict(str:str)): label vocabulary
            label_vocab_size (int): size of label vocabulary
        """
        seen, label_vocab, label_vocab_size = {'O'}, {'O': 'O'}, 0

        for base_dir in base_dirs:
            concept_dir = os.path.join(base_dir, 'concept')

            assert os.path.isdir(concept_dir), "Directory structure doesn't match!"

            ids = set([x[:-4] for x in os.listdir(concept_dir) if x.endswith('.con')])

            for i in ids:
                with open(os.path.join(concept_dir, f'{i:str}.con')) as f:
                    concepts = [self._assign_concept(x.strip()) for x in f.readlines()]
                for c in concepts:
                    if c['t'] not in seen:
                        label_vocab_size += 1
                        label_vocab['B-%s' % c['t']] = 'B-%s' % c['t']  # label_vocab_size
                        label_vocab_size += 1
                        label_vocab['I-%s' % c['t']] = 'I-%s' % c['t']  # label_vocab_size
                        seen.update([c['t']])
        return label_vocab, label_vocab_size

    def reformat(self, base_dirs, label_vocab, txt_dir=None, concept_dir=None, assertion_dir=None) -> dict:
        """
        Reformat the text data with annotated concepts and assertions

        Args:
            base_dirs (str/list[str]): directory/directories of data
            label_vocab (str): vocabulary for concept labels
            txt_dir (str): directory of records in raw text
            concept_dir (str): directory of concept annotation of records
            assertion_dir (str): directory of assertion annotation of records

        Returns: 
            dict(str:str): reformatted records with annotated concepts and assertions

        """
        txt_dir = os.path.join(base_dirs, 'txt') if not txt_dir else txt_dir
        concept_dir = os.path.join(base_dirs, 'concept') if not concept_dir else concept_dir
        assertion_dir = os.path.join(base_dirs, 'ast') if not assertion_dir else assertion_dir

        assert os.path.isdir(txt_dir) and os.path.isdir(concept_dir) and os.path.isdir(
            assertion_dir), "Directory structure doesn't match!"

        txt_ids = set([x[:-4] for x in os.listdir(txt_dir) if x.endswith('.txt')])
        concept_ids = set([x[:-4] for x in os.listdir(concept_dir) if x.endswith('.con')])
        assertion_ids = set([x[:-4] for x in os.listdir(assertion_dir) if x.endswith('.ast')])

        assert txt_ids == concept_ids, (
            f"id set doesn't match: txt - concept = {txt_ids - concept_ids}, concept - txt = {concept_ids - txt_ids}"
        )

        ids = txt_ids

        reprocessed_texts = {}
        for i in ids:
            with open(os.path.join(txt_dir, f'{i}.txt')) as f:
                lines = f.readlines()
                txt = [[y for y in x.strip().split(' ') if y.strip() != ''] for x in lines]

                line_starts_with_space = [x.startswith(' ') for x in lines]

            with open(os.path.join(concept_dir, f'{i}.con')) as f:
                concepts = [self._assign_concept(x.strip()) for x in f.readlines()]

            with open(os.path.join(assertion_dir, f'{i}.ast')) as f:
                assertion = (self._assign_assertion(x.strip()) for x in f.readlines())

            labels = [['O' for _ in line] for line in txt]
            for c in concepts:
                ast = None
                # extract label from annotated assertion
                if c['t'] == 'problem':
                    ast = next(assertion)
                    assert ast
                    assert c['c'] == ast['c']
                    assert c['start_line'] == ast['start_line']
                    assert c['end_line'] == ast['end_line']

                if c['start_line'] == c['end_line']:
                    line = c['start_line'] - 1
                    p_modifier = -1 if line_starts_with_space[line] else 0
                    text = (' '.join(txt[line][c['start_pos'] + p_modifier:c['end_pos'] + 1 + p_modifier])).lower()
                    assert text == c['c'], (
                            "Text mismatch! %s vs. %s (id: %s, line: %d)\nFull line: %s"
                            "" % (c['c'], text, i, line, txt[line])
                    )

                # assign label of concept and assertion (for concept 'problem' only) to entity
                for line in range(c['start_line'] - 1, c['end_line']):
                    p_modifier = -1 if line_starts_with_space[line] else 0
                    start_pos = c['start_pos'] + p_modifier if line == c['start_line'] - 1 else 0
                    end_pos = c['end_pos'] + 1 + p_modifier if line == c['end_line'] - 1 else len(txt[line])

                    if line == c['end_line'] - 1:
                        labels[line][end_pos - 1] = label_vocab['I-%s' % c['t']]
                    if line == c['start_line'] - 1:
                        labels[line][start_pos] = label_vocab[
                            'B-%s' % c['t']] if not ast else '%s\t%s' % (label_vocab['B-%s' % c['t']], ast['a'])
                    for j in range(start_pos + 1, end_pos - 1):
                        labels[line][j] = label_vocab['I-%s' % c['t']]

            joined_words_and_labels = [zip(txt_line, label_line) for txt_line, label_line in zip(txt, labels)]

            reprocessed_texts[i] = '\n\n'.join(
                ['\n'.join(['\t'.join(p) for p in joined_line]) for joined_line in joined_words_and_labels]
            )

        return reprocessed_texts

    @staticmethod
    def _split_train_dev(name: str, reprocessed_texts: dict, split_ratio=0.9) -> dict:
        """
        Split data ids into 2 parts for training and development
        
        Args:
            name (str): name of subdirectory
            reprocessed_texts (dict(str:str)): reprocessed data after reformatting
            split_ratio (float): splitting ratio for training and development data
        
        Returns:
            train_ids (list[str]): list of ids chosen for training data
            dev_ids (list[str]): list of ids chosen for development data
        """
        all_train_ids = np.random.permutation(list(reprocessed_texts.keys()))
        num_data = len(all_train_ids)
        num_data_train = int(split_ratio * num_data)

        train_ids = all_train_ids[:num_data_train]
        dev_ids = all_train_ids[num_data_train:]

        print(f"{name.upper()} # Patients: Train: {len(train_ids)}, Dev: {len(dev_ids)}")

        return train_ids, dev_ids

    def run(self):
        """
        Runs reformatting processor
        """

        label_vocab, label_vocab_size = self._build_label_vocab(self.train_dirs)

        reprocessed_texts = {'train': {subdir_name: self.reformat(base_dirs=subdir, label_vocab=label_vocab)
                                       for subdir_name, subdir in zip(self.train_sub_dirs, self.train_dirs)},
                             'test': self.reformat(base_dirs=self.test_dir,
                                                   label_vocab=label_vocab,
                                                   txt_dir=self.test_txt_dir,
                                                   concept_dir=os.path.join(self.test_dir, 'concepts'))
                             }
        np.random.seed(1)

        print('Start merging')
        merged_train, merged_dev = [], []
        for subdir in reprocessed_texts['train'].keys():
            subdir_train_ids, subdir_dev_ids = self._split_train_dev(name=subdir,
                                                                     reprocessed_texts=reprocessed_texts['train'][
                                                                         subdir],
                                                                     split_ratio=self.split_ratio)
            merged_train.extend([reprocessed_texts[subdir][i] for i in subdir_train_ids])
            merged_dev.extend([reprocessed_texts[subdir][i] for i in subdir_dev_ids])
        merged_test = list(reprocessed_texts['test'].values())
        print(f"Merged # Samples: Train: {len(merged_train)}, Dev: {len(merged_dev)}, Test: {len(merged_test)}")

        txt = {'train': '\n\n'.join(np.random.permutation(merged_train)),
               'dev': '\n\n'.join(np.random.permutation(merged_dev)),
               'test': '\n\n'.join(np.random.permutation(merged_test))
               }

        print(f"Check if {self.output_dir} exists")
        # check whether output directory exists
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        print("Start writing files")
        # write each txt to output directory
        if not self.output_dir.endswith('/'):
            self.output_dir += '/'
        for filename, data_txt in txt.items():
            print(f"Write {filename.upper()} in {self.output_dir}")
            with open(f'{self.output_dir}{filename}.tsv', 'w') as f:
                f.write(data_txt)

        print("Start writing pickle files")
        print("\tWrite LABEL VOCAB pickle file")
        with open(f'{self.output_dir}label_vocab.pkl', 'wb') as f:
            pickle.dump(label_vocab, f)

        print("\tWrite IDS TO ASSERTION LABELS pickle file")
        with open(f'{self.output_dir}ids_ast_labels.pkl', 'wb') as f:
            pickle.dump({i: ast_label for i, ast_label in enumerate(self.assertion_labels)}, f)

        print("\tWrite ASSERTION LABELS TO IDS pickle file")
        with open(f'{self.output_dir}ast_labels_ids.pkl', 'wb') as f:
            pickle.dump({i: ast_label for i, ast_label in enumerate(self.assertion_labels)}, f)

        print("Done.")
