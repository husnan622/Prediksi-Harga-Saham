# Prediksi Harga Saham - Husnan

## Domain Proyek

Saham adalah sebuah bukti kepemilikan nilai sebuah perusahaan, artinya pemilik saham adalah pemilik perusahaan. Semakin besar saham yang dimiliki, maka semakin besar kekuasaannya di perusahaan tersebut. Pada kesempatan ini saya membangun model LSTM untuk memprediksi harga saham. LSTM sangat berguna dalam memprediksi harga saham. Ini karena prediksi harga saham di masa depan bergantung pada harga sebelumnya.

## Business Understanding

### Problem Statement
- Apa yang dapat membantu proses analisis fundamental/pasar untuk mendukung pengambilan keputusan investasi saham?

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
Pada proyek ini menggunakan Tensorflow dan LSTM.
- Menentukan model Sequential.
- Menambahkan LSTM layer dengan memberikan beberapa network units.
- Mengatur return sequence
- Menambahkan Dense layer dengan memberikan beberapa network units.
- Menggunakan optimizer "adam" dan loss function "mean square error".
- Melatih model serta mengatur batch size dan epoch.  

## Evaluation

- Metrik yang digunakan adalah root mean square error (RMSE) untuk memeriksa kinerja model.
- Hasil proyek berdasarkan metrik adalah `1.3004825948431418` yang mana model bekerja dengan baik

![Prediksi Saham](https://user-images.githubusercontent.com/57633103/188735701-75e8beab-ebbc-477f-be9c-68f186f907dc.png)
