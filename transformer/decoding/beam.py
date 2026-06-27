import heapq
import typing as t

import torch
import numpy as np
import pydantic as pyd
from transformers import PreTrainedTokenizer

from transformer.decoding.base import BaseDecoder
from transformer.params import BeamSearchParams
from transformer.models import CausalLM, Seq2SeqLM

__all__ = ["BeamSearchDecoder"]


class Path(t.NamedTuple):
    log_prob: float
    window: torch.LongTensor
    mask: torch.LongTensor
    output: torch.LongTensor
    length: int
    terminated: bool

    def __lt__(self: t.Self, other: t.Any) -> bool:
        return self.log_prob < other.log_prob


class BeamSearchDecoder(BaseDecoder):
    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: t.Self,
        params: BeamSearchParams,
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

        # bikin prediksi
        pred = forward(output_ids, output_masks)

        # inisialisasi heap pakai k token dengan skor log-softmax tertinggi
        paths: list[Path] = []
        log_probs = pred[0, length - 1, valid_ids].topk(k=self.params.beam_width)
        for i in range(self.params.beam_width):
            log_prob: torch.FloatTensor = log_probs.values[i].item()
            token_id: torch.LongTensor = valid_ids[log_probs.indices][i]
            path_output: torch.LongTensor = output.clone()
            path_output[length] = token_id
            window: torch.LongTensor = output_ids.clone()
            window[0, length] = token_id
            mask: torch.LongTensor = output_masks.clone()
            mask[0, length] = 1
            heapq.heappush(
                paths, Path(log_prob, window, mask, path_output, length + 1, False)
            )

        while paths:
            # kalau jalur paling mungkin sudah berhenti, kembalikan
            if paths[0].terminated:
                output = paths[0].output
                break

            new_paths: list[Path] = []
            while paths:
                # pilih jalur paling kecil kemungkinannya sekarang
                path = heapq.heappop(paths)

                # jangan perluas jalur yang sudah berhenti
                if path.terminated:
                    if len(new_paths) < self.params.beam_width:
                        # push jalur baru
                        heapq.heappush(new_paths, path)
                    else:
                        # ganti jalur paling kecil kemungkinannya kalau yang baru lebih mungkin
                        if path.log_prob >= new_paths[0].log_prob:
                            heapq.heapreplace(new_paths, path)
                    continue

                # bikin prediksi
                pred = forward(path.window, path.mask)

                # ambil token teratas paling mungkin
                idx = (
                    -1
                    if path.length >= self.model.params.context_length
                    else path.length - 1
                )
                log_probs = pred[0, idx, valid_ids].topk(k=self.params.beam_width)

                # perluas jalur
                for i in range(self.params.beam_width):
                    next_log_prob: torch.FloatTensor = log_probs.values[i].item()

                    # jangan diperluas kalau log-prob baru lebih rendah dari yang terendah sekarang
                    if (
                        len(new_paths) == self.params.beam_width
                        and next_log_prob < new_paths[0].log_prob
                    ):
                        continue

                    # copy output & set id token berikutnya
                    next_token_id: torch.LongTensor = valid_ids[log_probs.indices][i]
                    path_output: torch.LongTensor = path.output.clone()
                    path_output[path.length] = next_token_id
                    # geser context window & mask (kalau perlu) lalu update
                    window: torch.LongTensor = path.window.clone()
                    mask: torch.LongTensor = path.mask.clone()
                    if path.length >= self.model.params.context_length:
                        window[0, :-1] = window[0, 1:].clone()
                        window[0, -1] = next_token_id
                    else:
                        mask[0, path.length] = 1
                        window[0, path.length] = next_token_id
                    # cek: sudah mentok panjang maksimum atau ketemu token <eos>?
                    terminated = (
                        next_token_id == tokenizer.eos_token_id
                        or path.length + 1 == self.params.max_length
                    )

                    # bikin jalur baru
                    new_path = Path(
                        log_prob + next_log_prob,
                        window,
                        mask,
                        path_output,
                        path.length + 1,
                        terminated,
                    )

                    if len(new_paths) < self.params.beam_width:
                        # push jalur baru
                        heapq.heappush(new_paths, new_path)
                    else:
                        # ganti jalur paling kecil kemungkinannya kalau yang baru lebih mungkin
                        # catatan: gak bakal sampai sini kalau ada jalur yang lebih mungkin
                        heapq.heapreplace(new_paths, new_path)

            # update kandidat sekarang
            paths = new_paths

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
