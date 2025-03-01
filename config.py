import copy
import json
import os

from transformers import AutoConfig


class Config(object):
    def __init__(self, **kwargs):
        self.coref = kwargs.pop('coref', False)
        # bert
        self.bert_model_name = kwargs.pop('bert_model_name', 'bert-large-cased')
        self.bert_cache_dir = kwargs.pop('bert_cache_dir', None)

        # mapping supervision
        self.per_declaration = kwargs.pop('per_declaration', False)

        # decoding
        self.max_position_embeddings = kwargs.pop('max_position_embeddings', 2048)

        # files
        self.train_file = kwargs.pop('train_file', None)
        self.dev_file = kwargs.pop('dev_file', None)
        self.log_path = kwargs.pop('log_path', './log')
        self.output_path = kwargs.pop('output_path', './output')

        # training
        self.accumulate_step = kwargs.pop('accumulate_step', 1)
        self.batch_size = kwargs.pop('batch_size', 10)
        self.eval_batch_size = kwargs.pop('eval_batch_size', 5)
        self.eval_period = kwargs.pop('eval_period', 1)
        self.max_epoch = kwargs.pop('max_epoch', 50)
        self.max_length = kwargs.pop('max_length', 128)
        self.learning_rate = kwargs.pop('learning_rate', 5e-5)
        self.weight_decay = kwargs.pop('weight_decay', 1e-5)
        self.warmup_epoch = kwargs.pop('warmup_epoch', 5)
        self.grad_clipping = kwargs.pop('grad_clipping', 5.0)
        self.SOT_weights = kwargs.pop('SOT_weights', 100)

        # others
        self.use_gpu = kwargs.pop('use_gpu', True)
        self.gpu_device = kwargs.pop('gpu_device', 0)
        self.use_copy = kwargs.pop('use_copy', False)
        self.k = kwargs.pop('k', 12)
        self.enrich_ner = kwargs.pop('enrich_ner', False)
        self.seed = None # Taken from command line args

    @classmethod
    def from_dict(cls, dict_obj):
        """Creates a Config object from a dictionary.
        Args:
            dict_obj (Dict[str, Any]): a dict where keys are
        """
        config = cls()
        for k, v in dict_obj.items():
            setattr(config, k, v)
        return config

    @classmethod
    def from_json_file(cls, path):
        with open(path, 'r', encoding='utf-8') as r:
            return cls.from_dict(json.load(r))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def save_config(self, path):
        """Save a configuration object to a file.
        :param path (str): path to the output file or its parent directory.
        """
        if os.path.isdir(path):
            path = os.path.join(path, 'config.json')
        print('Save config to {}'.format(path))
        with open(path, 'w', encoding='utf-8') as w:
            w.write(json.dumps(self.to_dict(), indent=2,
                               sort_keys=True))


