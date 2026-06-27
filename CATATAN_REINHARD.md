# Catatan Belajar 

Catatan santai buat diriku sendiri pas bedah repo Transformer ini. Bukan dokumen
resmi, cuma biar paham alurnya tiap buka lagi.

## Inti yang harus kupegang

Transformer itu intinya satu ide: **attention**. Tiap token "nanya" ke semua
token lain  mana yang relevan buatku? Caranya lewat tiga vektor: Query (Q),
Key (K), Value (V). Q dicocokkan ke semua K → dapat bobot → ambil campuran
berbobot dari V.

Rumus yang wajib diingat:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √dₖ) · V
```

`√dₖ` itu biar angkanya gak meledak pas dimensi besar (scaling).

## Peta file ke konsep

- `modules/embedding.py` → ubah token jadi vektor + positional encoding
  (karena attention buta urutan, posisi harus disuntik manual).
- `modules/attention.py` → ini jantungnya. Ada `SelfAttention` (satu kepala) dan
  `MultiHeadSelfAttention` (banyak kepala paralel, tiap kepala nangkap pola beda).
- `modules/block.py` → satu blok = attention → add & norm → FFN → add & norm.
  Add (residual) biar gradien ngalir lancar, norm (layer norm) biar stabil.
- `modules/transformers/` → tinggal numpuk blok jadi tiga bentuk:
  - `encoder_only.py` → buat klasifikasi/regresi (paham teks).
  - `decoder_only.py` → gaya GPT (prediksi token berikutnya).
  - `encoder_decoder.py` → buat translasi (paper aslinya bentuk ini).
- `decoding/` → cara milih token output: greedy (paling aman), beam (cari jalur
  terbaik), top-k & nucleus (lebih variatif), temperature (atur "berani"-nya).

## Yang sempat bikin aku mikir

- **Masked attention** di decoder: token gak boleh ngintip masa depan, jadi
  dikasih masker segitiga-atas sebelum softmax. Penting buat training.
- **Cross-attention** di encoder-decoder: Q dari decoder, tapi K & V dari output
  encoder, di sinilah decoder "membaca" kalimat sumber.
- Bedanya tiga arsitektur cuma soal blok apa yang dipakai dan masking-nya,
  bukan mekanisme attention-nya. Mekanismenya sama.

## Sumber

Kode inti: Edwin Onuonga (MIT). Paper: Attention Is All You Need (2017).
Catatan ini hasil belajarku sendiri di atas kode beliau.
