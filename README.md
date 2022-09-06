# Prediksi Harga Saham - Husnan

## Domain Proyek

Saham adalah sebuah bukti kepemilikan nilai sebuah perusahaan, artinya pemilik saham adalah pemilik perusahaan. Semakin besar saham yang dimiliki, maka semakin besar kekuasaannya di perusahaan tersebut. Pada kesempatan ini saya membangun model LSTM untuk memprediksi harga saham. LSTM sangat berguna dalam memprediksi harga saham. Ini karena prediksi harga saham di masa depan bergantung pada harga sebelumnya.

## Business Understanding

### Problem Statement
- Apa yang dapat membantu proses analisis fundamenal/pasar untuk mendukung pengambilan keputusan investasi saham?

### Goals
- Membangun model LSTM untuk memprediksi harga saham.

## Data Understanding
Pada proyek ini saya menggunakan data dari yFinance. yFinance ialah sumber terbuka yang menggunakan API Yahoo yang tersedia untuk umum, dan ditujukan untuk tujuan penelitian dan pendidikan. Bisa diakses di: [yfinance](https://pypi.org/project/yfinance/#description). 

### Variabel-variabel pada yFinance adalah sebagai berikut:
- Date : Tanggal
- Open : Harga Pembukaan
- High : Harga tertinggi
- Low : Harga terendah
- Close : Harga penutupan
- Adj Close : Harga penutupan yang disesuaikan dengan aksi korporasi seperti right issue, stock split atau stock reverse
- Volume : Volume transaksi biasanya dalam jumlah lembar

## Data Preparation
Teknik data preparation yang dilakukan pada proyek ini adalah sebagai berikut:
- Memisahkan data harga saham menjadi data training dan dan data testing.
- Menggunakan Scikit-Learn MinMaxScaler untuk menormalkan semua data mulai dari 0 hingga 1.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
