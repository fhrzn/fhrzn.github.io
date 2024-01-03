---
title: 'Bag of Words vs TF-IDF — Penjelasan dan Perbedaannya'
date: 2021-11-01T23:53:18+07:00
tags: ["textprocessing", "tfidf", "preprocessing"]
draft: false
description: "Bag of Words dan TF-IDF adalah 2 metode transformasi teks yang cukup populer. Mari kita bahas cara kerja serta perbedaannya!"
disableShare: true
hideSummary: false
ShowReadingTime: true
ShowWordCount: true
cover:
    image: "cover.jpg" # image path/url
    alt: "Cover Post" # alt text
    caption: "Photo by [Alfons Morales](https://unsplash.com/@alfonsmc10) on [Unsplash](https://unsplash.com/photos/YLSwjSy7stw)" # display caption under cover
    relative: true # when using page bundles set this to true
keywords: ["text processing", "preprocessing", "tfidf", "bag of words"]
summary: "Ketika kita berhubungan dengan data teks seperti klasifikasi teks misalnya, kita tentunya harus melakukan transformasi data teks menjadi sekumpulan angka (vektor) terlebih dahulu sebelum melakukan modelling. Nah, 2 metode yang cukup populer diantaranya adalah Bag of Words dan TF-IDF. Mari kita bahas bagaimana mereka bekerja serta apa perbedaannya!"
---

> Ketika kita berhubungan dengan data teks seperti klasifikasi teks misalnya, kita tentunya harus melakukan transformasi data teks menjadi sekumpulan angka (vektor) terlebih dahulu sebelum melakukan modelling. Nah, 2 metode yang cukup populer diantaranya adalah Bag of Words dan TF-IDF. Mari kita bahas bagaimana mereka bekerja serta apa perbedaannya!

# The Story

Bayangkan saja kita adalah pemilik restoran. Setiap pengunjung selesai makan, kita meminta mereka untuk menuliskan review dari segi apapun sebagai bahan evaluasi restoran. Dan setiap akhir bulan kita melakukan evaluasi berdasarkan review pengunjung. Kebetulan, bulan ini kita mendapat 3 review yang isinya seperti berikut:

**_Review 1:_** _Makanan disini gurih dan enak!_

**_Review 2:_** _Makanan disini biasa saja._

**_Review 3_**_: Makanan disini hambar dan tidak enak!_

Sebagai pemilik restoran yang melek IT, kita ingin seluruh review nantinya diproses menggunakan komputer. Sayangnya oh sayangnya, komputer tidak mengerti bahasa manusia. Mereka hanya memahami angka. Oleh karena itu, kita perlu melakukan transformasi terhadap data kita dari teks menjadi sekumpulan angka yang biasa disebut vektor. Yuk, mari kita lakukan!

# Bag of Words

Bag of Words (BoW) merupakan salah satu metode paling sederhana dalam mengubah data teks menjadi vektor yang dapat dipahami oleh komputer. Metode ini sejatinya hanya menghitung frekuensi kemunculan kata pada seluruh dokumen.

Mari kita ingat kembali contoh yang sudah kita baca sebelumnya.

**_Review 1:_** _Makanan disini gurih dan enak!_

**_Review 2:_** _Makanan disini biasa saja._

**_Review 3_**_: Makanan disini hambar dan tidak enak!_

Pertama, kita abaikan tanda baca serta huruf kapital dari ketiga review tersebut. Kemudian kita bisa membentuk sebuah korpus / kamus kata seperti berikut.

“makanan”

“disini”

“gurih”

“dan”

“enak”

“biasa”

“saja”

“hambar”

“tidak”

_Perlu diperhatikan sebelumnya, bahwa dalam membentuk korpus, kita hanya menghitung kata secara unik. Artinya, setiap kata yang berulang hanya akan ditulis sekali._

Berikutnya, mari kita hitung frekuensi kemunculan kata di korpus tersebut kepada ketiga review sebelumnya. Kita beri nilai **1** jika kata tersebut muncul pada sebuah review dan **0** jika tidak muncul.

Agar lebih mudah dalam memahminya, mari kita perhatikan tabel berikut.
![Bag of words](https://miro.medium.com/v2/resize:fit:720/format:webp/1*Kq-sqbWpvxn_cbm4e_vkBw.png)
*Perhitungan Bag of Words (BoW)*

Dari tabel tersebut, akhirnya kita dapatkan vektor dari setiap review seperti berikut.

**Vektor Review 1** = \[1, 1, 1, 1, 1, 0, 0, 0, 0\]

**Vektor Review 2** = \[1, 1, 0, 0, 0, 1, 1, 0, 0\]

**Vektor Review 3** = \[1, 1, 0, 1, 1, 0, 0, 1, 1\]

Itulah konsep dari Bag of Words, cukup mudah bukan? Namun, meski demikian metode ini ternyata memiliki beberapa kekurangan. Yuk mari kita ulas.

## Kekurangan Bag of Words

1.  Ukuran korpus Bag of Words mengikuti jumlah kata unik dari seluruh dokumen. Artinya, jika nantinya terdapat berbagai kata unik baru maka ukuran korpus juga akan semakin membesar. Tentunya hal ini akan berpengaruh pada komputasi yang dibutuhkan pada saat kita melatih model machine learning.
2.  Seperti yang kita lihat pada tabel diatas, ada banyak angka 0 dalam vektor kita. Kondisi ini biasa juga disebut dengan _sparse matrix_. Hal tersebut harusnya kita hindari karena model harus menemukan informasi yang sedikit dalam ukuran data yang besar, yang tentunya juga akan membutuhkan proses komputasi lebih tinggi.
3.  Bag of Words menghilangkan konteks kalimat akibat tidak memperhatikan urutan kata.

# TF-IDF

TF-IDF merupakan singkatan dari _Term Frequency — Inverse Document Frequency_. Sejatinya, TF-IDF merupakan gabungan dari 2 proses yaitu **_Term Frequency_ (TF)** dan **_Inverse Document Frequency_ (IDF)**.

TF-IDF biasa digunakan ketika kita ingin mengubah data teks menjadi vektor namun dengan memperhatikan apakah sebuah kata tersebut cukup informatif atau tidak. Mudahnya, TF-IDF membuat kata yang sering muncul memiliki nilai yang cenderung kecil, sedangkan untuk kata yang semakin jarang muncul akan memiliki nilai yang cenderung besar. Kata yang sering muncul disebut juga **_Stopwords_** biasanya dianggap kurang penting, salah satu contohnya adalah kata hubung (yang, di, akan, dengan, dll).

Sekarang, mari kita coba aplikasikan TF-IDF terhadap 3 review yang telah kita miliki sebelumnya.

## Term Frequency (TF)

Term Frequency (TF) menghitung frekuensi jumlah kemunculan kata pada sebuah dokumen. Karena panjang dari setiap dokumen bisa berbeda-beda, maka umumnya nilai TF ini dibagi dengan panjang dokumen (jumlah seluruh kata pada dokumen).

![Term Frequency](https://miro.medium.com/v2/resize:fit:720/format:webp/0*QgForh1rYt4_qJyp.png)
*Rumus Term Frequency (TF)*

**Keterangan**

tf = frekuensi kemunculan kata pada sebuah dokumen

Mari kita ambil contoh kalimat Review 1 untuk dihitung nilai TF nya.

**_Review 1:_** _Makanan disini gurih dan enak!_

*   Korpus = \[“makanan”, “disini”, “gurih”, “dan”, “enak”\]
*   Panjang kalimat = 5

Sehingga perhitungan untuk nilai TF nya menjadi:

*   TF(“**makanan**”) = 1/5 ≈ 0.2
*   TF(“**disini**”) = 1/5 ≈ 0.2
*   TF(“**gurih**”) = 1/5 ≈ 0.2
*   TF(“**dan**”) = 1/5 ≈ 0.2
*   TF(“**enak**”) = 1/5 ≈ 0.2

Berikutnya, mari kita coba terapkan pada seluruh review dan kita formulasikan ke dalam bentuk tabel seperti berikut.

![Perhitungan Term Frequency](https://miro.medium.com/v2/resize:fit:720/format:webp/0*y-flD_EXBtTKjSrd)
*Perhitungan Term Frequency (TF)*

**R1, R2, R3** merupakan notasi untuk setiap **_Review 1, Review 2,_** dan **_Review 3_**. Sedangkan **TF1, TF2, TF3** merupakan notasi untuk nilai **_Term Frequency_** setiap Review.

## Inverse Document Frequency (IDF)

Setelah kita berhasil menghitung nilai Term Frequency, selanjutnya kita hitung nilai **Inverse Document Frequency** (IDF), yang merupakan nilai untuk mengukur seberapa penting sebuah kata. IDF akan menilai kata yang sering muncul sebagai kata yang kurang penting berdasarkan kemunculan kata tersebut pada seluruh dokumen. Semakin kecil nilai IDF maka akan dianggap semakin tidak penting kata tersebut, begitu pula sebaliknya.

![Inverse Document Frequency](https://miro.medium.com/v2/resize:fit:720/format:webp/0*nLUqaPGaf7ISsMST.png)
*Rumus Inverse Document Frequency (IDF)*

Setiap review yang diberikan oleh pelanggan merupakan sebuah dokumen. Karena pada tulisan ini kita mempunyai 3 review, maka artinya kita mempunyai 3 dokumen.

Mari kita coba hitung nilai IDF untuk masing-masing kata pada Review 1.

**_Review 1:_** _Makanan disini gurih dan enak!_

*   Korpus = \[“makanan”, “disini”, “gurih”, “dan”, “enak”\]
*   Jumlah dokumen = 3

Sehingga perhitungan untuk nilai IDF nya menjadi:

*   IDF(“**makanan**”) = $log(\\frac{3} {3})$ ≈ 0
*   IDF(“**disini**”) = $log(\\frac{3} {3})$ ≈ 0
*   IDF(“**gurih**”) = $log(\\frac{3} {1})$ ≈ 0.48
*   IDF(“**dan**”) = $log(\\frac{3} {2})$ ≈ 0.18
*   IDF(“**enak**”) = $log(\\frac{3} {2})$ ≈ 0.18

Sekarang, mari kita coba terapkan pada seluruh kata dan kita lengkapi tabel TF sebelumnya seperti berikut.

![Perhitungan Inverse Document Frequency](https://miro.medium.com/v2/resize:fit:720/format:webp/0*NcBUdfPD4dBwZ6hD)
*Perhitungan Inverse Document Frequency (IDF)*

## Term Frequency — Inverse Document Frequency (TF-IDF)

Setelah kita punya TF dan IDF, berikutnya kita bisa menghitung nilai TF-IDF yang merupakan hasil perkalian dari TF dan IDF.

![Rumus TF-IDF](https://miro.medium.com/v2/resize:fit:640/format:webp/0*gVV1W6_AjuXUmnF8.png)
*Rumus TF-IDF*

Karena kita sudah memiliki nilai TF dan IDF untuk setiap kata, maka mari kita coba hitung nilai TF-IDF untuk setiap kata pada Review 1.

**_Review 1:_** _Makanan disini gurih dan enak!_

**makanan**

*   TF(“**makanan**”) = 1/5 ≈ 0.2
*   IDF(“**makanan**”) = $log(\\frac{3} {3})$ ≈ 0
*   TFIDF(**“makanan”**) = $0.2 \\times0=0$

**disini**

*   TF(“**disini**”) = 1/5 ≈ 0.2
*   IDF(“**disini**”) = $log(\\frac{3} {3})$ ≈ 0
*   TFIDF(**“disini”**) = $0.2 \\times0=0$

**gurih**

*   TF(“**gurih**”) = 1/5 ≈ 0.2
*   IDF(“**gurih**”) = $log(\\frac{3} {1})$ ≈ 0.48
*   TFIDF(**“gurih”**) = $0.2 \\times0.48=0.095$

**dan**

*   TF(“**dan**”) = 1/5 ≈ 0.2
*   IDF(“**dan**”) = $log(\\frac{3} {2})$ ≈ 0.18
*   TFIDF(**“makanan”**) = $0.2 \\times0.18=0.035$

**enak**

*   TF(“**enak**”) = 1/5 ≈ 0.2
*   IDF(“**enak**”) = $log(\\frac{3} {2})$ ≈ 0.18
*   TFIDF(**“makanan”**) = $0.2 \\times0.18=0.035$

Sekarang, mari kita coba lengkapi tabel sebelumnya dengan nilai TF-IDF pada seluruh kata seperti berikut.

![Perhitungan TF](https://miro.medium.com/v2/resize:fit:720/format:webp/0*zB5yKRC0KmsSGvbO)
*Perhitungan Term Frequency — Inverse Document Frequency (TF-IDF)*

> N**ote**: Mungkin untuk sebagian perhitungan, angkanya tidak presisi dikarenakan tools yang saya gunakan. Semoga bisa dimaklumi dan tetap bisa diambil konsepnya 🙂

Dari tabel tersebut, akhirnya kita dapatkan vektor dari setiap review yang dinotasikan oleh **_TFIDF1, TFIDF2,_** dan **_TFIDF3_** seperti berikut.

**Vektor Review 1** = \[0, 0, 0.095, 0.035, 0.035, 0, 0, 0, 0\]

**Vektor Review 2** = \[0, 0, 0, 0, 0, 0.119, 0.119, 0, 0\]

**Vektor Review 3** = \[0, 0, 0, 0.0293, 0.0293, 0, 0, 0.080, 0.080\]

## Kekurangan TF-IDF

1.  TF-IDF sejatinya berdasar pada Bag of Words (BoW), sehingga TF-IDF pun tidak bisa menangkap posisi teks dan semantiknya.
2.  TF-IDF hanya berguna sebagai fitur di level leksikal.

So, itulah perbedaan antara Bag of Words (BoW) dan TF-IDF sebagai metode untuk transformasi teks menjadi vektor. Jika ada pertanyaan, diskusi, sanggahan, kritik, maupun saran jangan pernah ragu untuk menuliskannya di kolom komentar 🙂

Sekian tulisan saya kali ini, mohon maaf apabila ada kekurangan dan salah kata, semoga bermanfaat. Terima kasih!

Yuk, belajar dan diskusi lebih lanjut tentang seputar Data Science, Artificial Intelligence, dan Machine Learning dengan gabung di discord [Jakarta AI Research](https://discord.gg/6v28dq8dRE). Dan jangan lupa follow medium [Data Folks Indonesia](https://medium.com/data-folks-indonesia) biar nggak ketinggalan update terbaru dari kami.

---

### Referensi

1.  [https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/](https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/)
2.  [https://machinelearningmastery.com/gentle-introduction-bag-words-model/](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
3.  [http://www.tfidf.com/](http://www.tfidf.com/)
4.  [https://www.quora.com/What-are-the-advantages-and-disadvantages-of-TF-IDF](https://www.quora.com/What-are-the-advantages-and-disadvantages-of-TF-IDF)
