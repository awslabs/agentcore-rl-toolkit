"""``PayloadDataset`` — verl dataset for payload-first ACR training data.

The dataset contract for the AgentCore backend is a single ``payload`` column
holding each row's exact ACR invoke payload, authored against the agent's own
API. verl's machinery, however, needs a chat-format ``prompt`` column (its
dataloader fabricates ``raw_prompt`` from it, and length filtering renders it).
Rather than making users author that ceremony column themselves, this dataset
synthesizes it at load time from a designated payload field:

    data:
      custom_cls:
        path: <this file or pkg://agentcore_rl_toolkit.backends.experimental.verl.dataset>
        name: PayloadDataset
      payload_prompt_field: prompt   # which payload field is "the prompt" (default: prompt)

Rows that already have a ``prompt`` column are left untouched, so datasets
authored the explicit two-column way keep working.
"""

import logging

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class PayloadDataset(RLHFDataset):
    """RLHFDataset that synthesizes the chat-format ``prompt`` column verl
    requires from the ``payload`` column's designated prompt field."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def maybe_filter_out_long_prompts(self, dataframe=None):
        # Runs after load + max_samples selection, before length filtering —
        # the earliest seam where the full dataframe is in hand.
        dataframe = self._synthesize_prompt_column(dataframe)
        return super().maybe_filter_out_long_prompts(dataframe)

    def _synthesize_prompt_column(self, dataframe):
        if dataframe is None or "payload" not in dataframe.column_names:
            return dataframe
        if self.prompt_key in dataframe.column_names:
            return dataframe  # explicit prompt column wins; nothing to do

        field = self.config.get("payload_prompt_field", "prompt")
        prompt_key = self.prompt_key

        def add_prompt(row):
            payload = row["payload"]
            text = payload.get(field) if isinstance(payload, dict) else None
            if not isinstance(text, str):
                raise ValueError(
                    f"PayloadDataset: payload[{field!r}] must be a string to synthesize the "
                    f"chat prompt column; got {type(text).__name__}. Set data.payload_prompt_field "
                    "to the payload field holding the prompt text, or author an explicit "
                    f"chat-format {prompt_key!r} column."
                )
            return {prompt_key: [{"role": "user", "content": text}]}

        logger.info(
            "PayloadDataset: synthesizing chat %r column from payload[%r] for %d rows",
            prompt_key,
            field,
            len(dataframe),
        )
        return dataframe.map(add_prompt)


__all__ = ["PayloadDataset"]
