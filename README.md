# Transformer dari Nol (PyTorch) — Terjemahan Inggris → Italia

Implementasi arsitektur Transformer dari paper
[Attention Is All You Need](https://arxiv.org/abs/1706.03762), ditulis dari nol
pakai PyTorch. Fokusnya bikin arsitektur sequence-to-sequence buat menerjemahkan
teks dari bahasa Inggris ke Italia — tapi modulnya disusun rapi, jadi bisa
dipakai juga buat encoder-only dan decoder-only.

> **Catatan kepemilikan.** Kode inti repo ini **karya Edwin Onuonga**
> (`ed@eonu.net`), berlisensi **MIT**. Salinan ini cuma **kupelajari, kurapikan,
> dan kuanotasi ulang** buat belajar pribadi. Detail lengkap ada di `CREDITS.md`
> dan `LICENSE`. — Reinhard

## Gambaran singkat

Transformer mengubah cara kerja tugas sequence-to-sequence: alih-alih pakai
jaringan rekuren (RNN/LSTM) yang memproses kata satu per satu, dia pakai
mekanisme **attention** supaya tiap token bisa langsung "melihat" semua token
lain sekaligus. Repo ini memanfaatkan arsitektur itu buat menerjemahkan kalimat
Inggris ke Italia.

Yang menarik: hampir semua komponen ditulis dari nol — attention, multi-head,
positional encoding, block encoder/decoder, sampai strategi decoding (greedy,
beam, top-k, nucleus). Satu-satunya yang dipinjam dari luar adalah **tokenizer**
Hugging Face, murni buat pra-pemrosesan data.

## Data

Model dilatih dan dievaluasi pakai dataset
[g8a9/europarl_en-it](https://huggingface.co/datasets/g8a9/europarl_en-it) —
pasangan kalimat Inggris–Italia.

Idealnya dilatih di seluruh dataset (~1,9 juta pasangan kalimat), tapi karena
keterbatasan hardware, di implementasi aslinya dipakai 100K untuk pelatihan dan
20K untuk evaluasi.

## Hasil (dari pelatihan implementasi asli)

| Epoch | Loss (train) | Loss (val) | F1 (train) | F1 (val) | Waktu (menit) | Device |
|-------|--------------|------------|------------|----------|---------------|--------|
| 5 | 0.116378 | 0.091345 | 0.262342 | 0.273207 | 170 | 1 × GPU T4 |

## Struktur kode

```
transformer/
├─ modules/        # blok inti: attention, embedding, block, varian transformer
├─ models/         # model siap-pakai: seq2seq, causal, classifier, regressor
├─ dataloaders/    # penyiapan data: seq2seq, causal, inference
├─ decoding/       # strategi decoding: greedy, beam, top-k, nucleus, temperature
├─ params/         # konfigurasi (pydantic) buat transformer & decoding
└─ utils/          # konstanta + layer/fungsi bantuan
```

## Ucapan terima kasih

Arsitektur dan konfigurasi mengikuti paper
[Attention Is All You Need](https://arxiv.org/abs/1706.03762). Implementasi inti
sepenuhnya karya **Edwin Onuonga** (lisensi MIT) — terima kasih atas kode yang
rapi dan enak dipelajari.
