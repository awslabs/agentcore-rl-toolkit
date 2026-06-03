"""Dataset class for AgentCore rollout mode.

In AgentCore mode, input data is passed through to ACR containers as-is
(no tokenization needed on the trainer side). This dataset skips prompt
length filtering when there is no prompt column and returns raw dict items.
"""

import datasets
from verl.utils.dataset.rl_dataset import RLHFDataset


class AgentCoreDataset(RLHFDataset):
    """Dataset for AgentCore rollout mode.

    Set in config:
        data.custom_cls.path=agentcore_rl_toolkit.backends.verl.dataset
        data.custom_cls.name=AgentCoreDataset
    """

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        if self.prompt_key not in dataframe.column_names:
            return dataframe
        return super().maybe_filter_out_long_prompts(dataframe)

    def __getitem__(self, item):
        return self.dataframe[item]
