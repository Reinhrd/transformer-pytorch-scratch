import typing as t

import torch
import numpy as np
import pydantic as pyd
from transformers import PreTrainedTokenizer

from transformer.decoding.base import BaseDecoder
from transformer.params import TemperatureSamplingParams
from transformer.models import CausalLM, Seq2SeqLM


__all__ = ["TemperatureSamplingDecoder"]


class TemperatureSamplingDecoder(BaseDecoder):
    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: t.Self,
        params: TemperatureSamplingParams,
        model: CausalLM | Seq2SeqLM,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(params=params, model=model, random_state=random_state)

    def _generate(
        self: t.Self,
        context: str | None,
        /,
        *,
        tokenizer: PreTrainedTokenizer,
        forward: t.Callable[[torch.LongTensor, torch.LongTensor], torch.FloatTensor],
    ) -> str:
        # tokenisasi konteks (kalau ada)
        output, output_ids, output_masks, length = super()._generate(
            context, tokenizer=tokenizer, forward=forward
        )

        # ID yang valid buat di-sample
        valid_ids = self._valid_ids(tokenizer)

        while (
            length < self.params.max_length
            and output[length - 1] != tokenizer.eos_token_id
        ):
            # bikin prediksi
            pred = forward(output_ids, output_masks)

            # ambil log-prob buat token berikutnya
            idx = -1 if length >= self.model.params.context_length else length - 1
            log_probs = pred[0, idx, valid_ids]

            # sesuaikan logit dengan temperature - pakai trik log-sum-exp biar tetap di domain log
            adj_log_probs = log_probs / self.params.temperature
            adj_log_probs -= adj_log_probs.logsumexp(dim=0)

            if self.params.k is not None:
                # ambil k token teratas paling mungkin
                top_k = adj_log_probs.topk(k=self.params.k)
                # sampling dari distribusi kategorikal pakai trik Gumbel-max
                gumbels = self.random_state.gumbel(size=self.params.k)
                idx = top_k.indices[np.argmax(top_k.values + gumbels)]
            else:
                # pilih token paling mungkin
                idx = adj_log_probs.argmax()
            next_token_id = valid_ids[idx]

            # tambahkan ke output
            output[length] = next_token_id

            # geser context window & mask (kalau perlu) lalu update
            if length >= self.model.params.context_length:
                output_ids[0, :-1] = output_ids[0, 1:].clone()
                output_ids[0, -1] = next_token_id
            else:
                output_masks[0, length] = 1
                output_ids[0, length] = next_token_id

            length += 1

        # decode id, balikin token jadi string
        output_string = tokenizer.decode(
            output.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # kalau belum berhenti, tambahkan elipsis di akhir sebagai penanda
        if output[length - 1] != tokenizer.eos_token_id:
            output_string = f"{output_string}..."

        return output_string
