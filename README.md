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

|Date       |Open      |High      |Low       |Close     |Adj Clos  |Volume
|-----------|----------|----------|----------|----------|----------|----------|
|2016-01-04 |25.652500 |26.342501 |25.500000 |26.337500 |24.151499 |270597600 |
|2016-01-05 |26.437500 |26.462500 |25.602501 |25.677500 |23.546272 |223164000 |
|2016-01-06 |25.139999 |25.592501 |24.967501 |25.174999 |23.085484 |273829600 |
|2016-01-07 |24.670000 |25.032499 |24.107500 |24.112499 |22.111172 |324377600 |
|2016-01-08	|24.637501 |24.777500 |24.190001 |24.240000 |22.228086 |283192000 |

### Variabel-variabel pada yFinance adalah sebagai berikut:
- Date : Tanggal
- Open : Harga Pembukaan
- High : Harga tertinggi
- Low : Harga terendah
- Close : Harga penutupan
- Adj Close : Harga penutupan yang disesuaikan dengan aksi korporasi seperti right issue, stock split atau stock reverse
- Volume : Volume transaksi biasanya dalam jumlah lembar

## Data Preparation
Data preparation yang dilakukan pada proyek ini adalah sebagai berikut:

### Mempersiapkan training set
- Mengekstrak closing prices dari data saham dan mengubah menjadi number series.
- Menghitung ukuran data untuk 80% dari kumpulan data. Metode math.ceil adalah untuk memastikan ukuran data dibulatkan menjadi bilangan bulat.
- Menggunakan Scikit-Learn MinMaxScaler untuk menormalkan semua data mulai dari 0 hingga 1.
- Memisahkan 80% pertama dari data sebagai training set.
- Membuat urutan data fitur (x_train) dan urutan data label (y_train).
- Memuat harga historis 60 hari sebagai data fitur (x_train) dan 60 hari berikutnya sebagai data label (y_train).
- Mengkonversi data fitur (x_train) dan data label (y_train) menjadi array Numpy. kemudian membentuk kembali x_train dan y_train menjadi array tiga dimensi sebagai bagian dari persyaratan untuk melatih model LSTM.

### Mempersiapkan test set
- Mengekstrak closing prices dari dataset yang dinormalisasi (20% terakhir dari dataset).
- Membuat data fitur (x_test) dan data label (y_test) dari dari test set.
- Mengkonversikan data fitur (x_test) dan data label (y_test) menjadi array Numpy. Kemudian membentuk kembali x_test dan y_test menjadi array tiga dimensi.

## Modeling
Modelling yang dilakukan pada proyek ini adalah sebagai berikut:
- Menentukan model Sequential yang terdiri dari linear stack of layers.
- Menambahkan lapisan LSTM dengan memberi 100 network units. Mengatur return_sequence menjadi true sehingga output dari layer akan menjadi urutan yang lain dengan panjang yang sama.
- Menambahkan lapisan LSTM lain dengan 100 network units. Mengatur return_sequence ke false.
- Menambahkan dense layer dengan 25 network units.
- Menambahkan dense layer yang menentukan output dari 1 network unit.
- Menampilkan ringkasan arsitektur LSTM.

## Evaluation
Evaluation yang dilakukan pada proyek ini adalah sebagai berikut:
- Menerapkan model untuk memprediksi harga saham berdasarkan test set.
- Menggunakan metode inverse_transform untuk mendenormalisasi prediksi harga saham.
- Menerapkan RMSE untuk menghitung tingkat perbedaan antara harga yang diprediksi dan harga sebenarnya (y_test) dan menampilkan hasilnya.
- Hasilnya menunjukkan bahwa RMSE hanya sekitar `1.3004825948431418`. Modelnya bekerja dengan baik.

## Conclusion
Dari grafik yang dihasilkan, kita dapat melihat prediksi harga saham mengikuti tren harga saham. Hal ini menunjukkan efektivitas LSTM bekerja dengan time series atau data sekuensial seperti harga saham.

![Prediksi Saham](https://user-images.githubusercontent.com/57633103/188735701-75e8beab-ebbc-477f-be9c-68f186f907dc.png)
