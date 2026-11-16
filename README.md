# uts-klasifikasi-kanker-paru
## Implementasi Machine Learning Dalam Memprediksi Level Penyakit Kangker Paru-Paru Menggunakan Algoritma Random Forest

nama kelompok : <br>
Dewa Ngakan Made Rasta Baladika AK (2501010074) <br>
I Gede Wahyu Candra Deanova (2501010079) <br>
Wayan Andhika Darma Putra (2501010070) <br>

# PENDAHULUAN

## LATAR BELAKANG

Kanker merupakan salah satu penyakit yang bisa menyebabkan kematian pada manusia. Penyakit kanker umumnya ada banyak seperti kanker payudara, kanker paru-paru, kanker hati, kanker kolorektal, dsb. Di antara semua jenis kanker, kanker paru-paru termasuk 10 besar penyakit yang menyebabkan kematian terbanyak di dunia. Penyakit ini umumnya disebabkan oleh kebiasaan merokok atau paparan asap rokok, faktor lingkungan, dan genetik. Menurut data World Health Organization (WHO), kanker paru-paru menyebabkan sekitar 1,8 juta hingga 2,5 juta kematian per tahun secara global.

Diagnosis kanker paru-paru sangat penting untuk diketahui sedari dini. Hal ini bertujuan untuk meningkatkan kesembuhan dan mengurangi tingkat resiko kematian. Diagnosis kanker paru-paru saat ini menghadapi tantangan akibat variasi gejala dari setiap pasien berbeda-beda. Oleh karena itu, pengembangan diagnosis yang akurat dan efisien sangat diperlukan.

Artificial Intelligence (AI) memberikan banyak manfaat yang mempermudah kehidupan manusia, salah satunya pada bidang kesehatan. Artificial Intelligence memberikan manfaat yang signifikan untuk meningkatkan efisiensi dan akurasi medis. Salah satu cabang  Artificial Intelligence yaitu machine learning. Machine learning adalah mesin yang berguna untuk memprediksi sesuatu dalam bentuk angka dan juga dalam bentuk klasifikasi. Contoh metode dalam memprediksi dalam bentuk angka adalah Linear Regression dan Random Forest Regressor. Sedangkan contoh metode dalam memprediksi dalam bentuk klasifikasi adalah Logistic Regression, Random Forest, Super Vector Machine (SVM).

## RUMUSAN MASALAH

Berdasarkan latar belakang di atas, rumusan masalah yang akan dibahas dalam laporan ini adalah: <BR>
1. Apakah Penggunaan metode Random Forest pada machine learning bisa mengklasifikasi kanker paru-paru? <BR>
2. Bagaimana hasil pengujian dalam menggunakan metode Random Forest dan apa saja faktor yang paling mempengaruhi? <BR>

## TUJUAN PENULISAN

Tujuan dari penulisan laporan teknis ini adalah:



## DATASET
https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link 

Jika berbicara analisis pasti tidak jauh jauh dari yang namanya dataset. Dataset merupakan kumpulan data yang yang disusun secara rapi sehingga mudah dibaca. Data yang diambil merupakan data dari kaggle yang merupakan data bersifat open source. Dataset tersebut berjumlah 1000 baris dan memiliki 26 atribut atau kolom. salah satu kolom yang paling penting adalah kolom target atau label, yaitu Level yang berisi nilai "Low" atau “Medium” atau "High" yang menunjukkan apakah pasien tersebut mengidap kanker paru-paru yang Low, Medium, atau High.

## RANDOMFOREST

Random forest (RF) didefinisikan sebagai algoritma pembelajaran mesin yang ampuh yang membangun sekelompok pohon keputusan  untuk membuat prediksi yang lebih baik melalui pemungutan suara (untuk klasifikasi).Implementasi RandomForest mudah dan sangat cepat karena tidak membutuhkan banyak memori, menjadikannya model yang sering digunakan oleh para peneliti untuk mengembangkan machine learning

# TAHAPAN/LANGKAH

## UPLOAD FILE
pertama-tama upload file pada google colab di bagian kiri yang ada gambar file lalu uploud data di sana.lalu jalankan kodingan ini agar bisa di baca oleh google colab.<br>
<img width="670" height="128" alt="Screenshot 2026-11-15 214346" src="https://github.com/user-attachments/assets/0376a3d7-040d-4418-835c-14096216f276" /> <br>




## DATA PROCESING

Langkah awal yang dilakukan adalah dengan memeriksa data yang kosong (missing value). Setelah diperiksa, dataset ini tidak terdapat missing value, jadi tidak perlu menambahkan atau menghapus data. Lalu, tidak lupa untuk mengecek data duplikat dari dataset tersebut. Setelah dilakukan pengecekan, ternyata dataset ini juga tidak memiliki data yang duplikat sehingga tidak perlu untu menghapus data lagi. Berikut cara untuk mengecek apakah ada data duplikat dan apakah ada data yang kosong.<BR>
<img width="254" height="77" alt="Screenshot 2026-11-15 173610" src="https://github.com/user-attachments/assets/38c3e37f-6d02-4e9f-821e-1017dbcb846d" /> <BR>
Tahap Selanjutnya dengan menentukan fitur dan target. Target atau label (y) yang digunakan merupakan kolom “level” . Kolom Level dipilih sebagai target karena proses ini digunakan untuk menandai apakah seseorang dalam dataset terindikasi menderita kanker paru-paru yang low, medium, atau high. Lalu dilakukan penghapusan beberapa fitur yaitu pada fitur “patience id” dan “index”. Hal ini dilakukan untuk mencegah agar data tidak terlalu kompleks untuk dipelajari model. Selain Itu, Fitur tersebut juga bukan fitur penting untuk mengklasifikasi Level dari penyakit kanker Paru-Paru. <br>
<img width="520" height="157" alt="Screenshot 2026-11-15 215206" src="https://github.com/user-attachments/assets/06ce4764-b065-4ad4-b19b-e856fe5a9745" />

Selanjutnya, data dibagi menjadi dua bagian menggunakan fungsi train_test_split dari pustaka Scikit-Learn, dengan proporsi 80% sebagai data latih dan 20% sebagai data uji. <BR>

<img width="653" height="256" alt="Screenshot 2026-11-15 220655" src="https://github.com/user-attachments/assets/e8858176-f78f-4cae-ad64-1439a38b810c" />

Selanjutnya dilakukan proses standarisasi. Proses ini menyamakan perbandingan angka pada data agar memiliki nilai yang seimbang.Hal ini dilakukan untuk mengantipasi mesin learning tidak terlalu menggunakan fitur yang memiliki nilai angka besar.

<img width="373" height="164" alt="Screenshot 2026-11-15 220543" src="https://github.com/user-attachments/assets/a11306c8-4356-4779-abaf-125d106d4d23" />

## MEMBANGUN MODEL
<img width="659" height="185" alt="Screenshot 2026-11-15 221706" src="https://github.com/user-attachments/assets/d4ce407b-7978-414d-9cf6-478ee726d27b" /> <br>
model yang digunakan adalah random forest yang merupakan salah satu model yang terkenal untuk mengklasifikasikan sesuatu.n_estimators = 200 itu berarti model tersebut menggunakan 200 pohon keputusan. random state berguna agar hasil percobaan dari pohon keputusan dapat diulang secara konsisten. Lalu fit() itu merupakan proses melatih model menggunakan data training.<br>
<img width="373" height="189" alt="Screenshot 2026-11-15 230103" src="https://github.com/user-attachments/assets/4810165b-8ea0-49a4-a3b8-33da1aede8b6" />

## HASIL EVALUASI

<img width="586" height="309" alt="Screenshot 2026-11-15 225328" src="https://github.com/user-attachments/assets/9144dccd-9a72-4a96-872b-6470ea4a259e" /> <br>

<img width="405" height="311" alt="Screenshot 2026-11-16 000445" src="https://github.com/user-attachments/assets/d7f3dba7-cb9e-444f-87a0-b14716524673" />


disini model random forest berhasil mendapat akurasi 100% yang berarti model berhasil menebak sesuai label atau target aslinya. 

## KESIMPULAN

berdasarkan hasil penelitian tersebut, dapat disimpulkan bahwa algoritma klasifikasi yang digunakan menunjukkan kinerja yang baik dalam mendeteksi kanker paru-paru. Random Forest dapat memprediksi level keparahan dari penyakit kanker paru paru. Dengan menggunakan data latih dan data uji sebesar 80:20 model mempengaruhi akurasi training 100% dan testing 100%. pada evaluasi confusion matriks menunjukan bahwa 61 data pada kelas low, 66 data kelas medium, 73 data kelas high diprediksi dengan benar tanpa ada kesalahan.






