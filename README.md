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

```python
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

stock_data = yf.download('AAPL', start='2016-01-01', end='2021-10-01')
stock_data.head()
```

**Baris 1-9:** 
> Mengimport semua library yang diperlukan.

**Baris 11–12:**
> Melakukan Pengunduhan dataset untuk memperoleh data saham, berikut pratinjau datanya.

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
Teknik data preparation yang dilakukan pada proyek ini adalah sebagai berikut:
- Memisahkan data harga saham menjadi data training dan dan data testing.
- Menggunakan Scikit-Learn MinMaxScaler untuk menormalkan semua data mulai dari 0 hingga 1.

### Mempersiapkan training set
```python
close_prices = stock_data['Close']
values = close_prices.values
training_data_len = math.ceil(len(values)* 0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))
train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
```
**Baris 1-2:** 
> Mengekstrak closing prices dari data saham dan mengubah menjadi number series.

**Baris 3:**
> Menghitung ukuran data untuk 80% dari kumpulan data. Metode math.ceil adalah untuk memastikan ukuran data dibulatkan menjadi bilangan bulat.

**Baris 5-6:**
> Menggunakan Scikit-Learn MinMaxScaler untuk menormalkan semua data mulai dari 0 hingga 1.

**Baris 7:**
> Memisahkan 80% pertama dari data sebagai training set.

**Baris 9-10:**
> Membuat urutan data fitur (x_train) dan urutan data label (y_train).

**Baris 12-14:**
> Memuat harga historis 60 hari sebagai data fitur (x_train) dan 60 hari berikutnya sebagai data label (y_train).

**Baris 16-17:**
> Mengkonversi data fitur (x_train) dan data label (y_train) menjadi array Numpy. kemudian membentuk kembali x_train dan y_train menjadi array tiga dimensi sebagai bagian dari persyaratan untuk melatih model LSTM.

### Mempersiapkan test set

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
