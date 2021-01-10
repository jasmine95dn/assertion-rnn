# --*-- coding: utf-8 --*--
"""
This module defines some elements for data
"""
import re
import os
import torch


class Entity:
    """
    Class Entity contains info of an entity, including an entity, its pre-entity (left part)
    and post-entity (right part) in a sentence

    Args:
        entity (str): entity sequence in sentence
        left (str): left context sequence of this entity in sentence
        right (str): right context sequence of this entity in sentence
        ast (str): assertion label for this medical entity

    Attributes:
        entity (str): entity sequence in sentence
        entity_start (int): start position of entity in sentence
        entity_end (int): end position of entity in sentence
        left (str): left context sequence of this entity in sentence
        right (str): right context sequence of this entity in sentence
        ast_label (str): assertion label for this medical entity
        entity_embedding (torch.Tensor): embedding representing entity in sentence
    """
    def __init__(self, entity: str, left: str, right: str, ast: str):

        self.entity = entity
        self.entity_start = len(left)
        self.entity_end = self.entity_start + len(entity) - 1
        
        self.left = left
        self.right = right
        self.ast_label = ast
        self.entity_embedding = None

    def set_entity_embedding(self, left_emb: torch.Tensor, entity_emb: torch.Tensor, right_emb: torch.Tensor):
        """
        Set embedding for 3 parts of an entity and combine them together as representation for an entity in sentence

        Args:
            left_emb (torch.Tensor): phrase embedding for left context sequence in sentence
            entity_emb (torch.Tensor): phrase embedding for entity in sentence
            right_emb (torch.Tensor): phrase embedding for right context sequence in sentence

        """
        assert isinstance(left_emb, torch.Tensor)
        assert isinstance(entity_emb, torch.Tensor)
        assert isinstance(right_emb, torch.Tensor)
        self.entity_embedding = torch.cat((left_emb, entity_emb, right_emb), 0)


class Sentence:
    """
    Class Sentence contains information of elements in sentence

    Args:
        line (str): line representing sentence
        sent_id (int): sentence id defined in data

    Attributes:
        replace (str): annotation part to be replaced
        sent_id (int): sentence id defined in data
        entities (dict(str:str)): all entities in a sentence
        sentence (str): sentence in raw form without annotation
    """
    def __init__(self, line: str, sent_id: int):

        self.replace = r'\t(O|[BI]-[a-z\t]+)'
        self.sent_id = sent_id

        self.entities = {}
        self.__parse_entity(line)

        self.sentence = ''
        self.__parse_sent(line)

    def __parse_entity(self, line: str):
        """
        Parse data used for training in GRU
        3 parts: pre-entity, entity, post-entity

        Args:
            line (str): line to parse entity
        """

        entities = re.findall(r'([^\n]+\tB-problem\t[a-z]+\n([^\n]+\tI-problem\n)*)', line)
        for entity, _ in entities:
            left, right = line.split(f'{entity}')
            entity = re.sub(self.replace, '', entity)
            self.entities[entity] = Entity(entity=entity, left=re.sub(self.replace, '', left),
                                           right=re.sub(self.replace, '', right),
                                           ast=re.search(r'B-problem\t([a-z]+)\n', entity).group(1))

    def __parse_sent(self, line: str):
        """
        Parse sentence used for embedding as input

        Args:
            line (str): line to parse sentence
        """
        # currently: each sentence is split with "\n"
        self.sentence = re.sub(self.replace, '', line)


class Data:
    """
    Class Data contains all sentences with annotations.

    Args:
        filename (str): path to preprocessed file with annotations
        type_data (str): name of data type (train/test)

    Attributes:
        type_data (str): name of data type (train/test)
        sentences (list[str]): list of all sentences that contain entity of medical problem concept and its assertion
                                annotation
    """
    def __init__(self, filename: str, type_data='train'):

        assert os.path.isfile(filename), f'{filename} does not exist!'

        self.type_data = type_data
        self.sentences = []

        self.__read_file(filename)

    def __read_file(self, filename: str):
        """
        Read a given file and return the related sentences for use

        Args:
            filename (str): name of data file
        """

        with open(filename) as f2r:
            data = [sent for sent in f2r.read().split('\n\n') if 'B-problem' in sent]

        self.sentences = [Sentence(line, i) for i, line in enumerate(data)]
