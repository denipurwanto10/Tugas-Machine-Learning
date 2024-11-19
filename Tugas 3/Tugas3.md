**Nama : Deni Purwanto** 

**NPM : 41155050210017** 

**Kelas : INF A** 

**Mata Kuliah : Machine Learning** 

0. Lakukan praktik dari https://youtu.be/5wwXKtLkyqs?si=fn88eveu\_qbCC6b3 , buat screenshot dengan nama kalian pada coding, kumpulkan dalam bentuk pdf, dari kegiatan ini: 
1. Pengenalan komponen Decision Tree: root, node, leaf 

![image](https://github.com/user-attachments/assets/5420a643-2ba6-4732-bec2-83e5c214e8e7)

Decision Tree adalah salah satu metode dalam machine learning yang digunakan untuk membuat model prediksi atau klasifikasi. Struktur decision tree terdiri dari beberapa komponen utama, yaitu root, node, dan leaf. Root (akar) adalah titik awal atau node pertama dari sebuah decision tree yang menjadi tempat pertama data dipisahkan berdasarkan kriteria tertentu, umumnya menggunakan atribut yang paling signifikan. Selanjutnya, node adalah titik di dalam pohon tempat keputusan dibuat berdasarkan atribut tertentu. Node ini dapat berupa internal node, yang masih memiliki cabang untuk keputusan lebih lanjut, atau leaf node (daun), yang tidak memiliki cabang lagi. Leaf adalah node terakhir dalam decision tree yang memberikan hasil akhir dari keputusan atau klasifikasi, seperti kategori atau nilai prediksi. Sebagai contoh, jika decision tree digunakan untuk memprediksi kelayakan kredit, root mungkin memulai dengan pertanyaan "Apakah pendapatan > 50 juta?", node berikutnya bisa mempertimbangkan "Apakah umur < 30 tahun?", dan leaf memberikan hasil akhir seperti "Layak Kredit" atau "Tidak Layak Kredit." Struktur ini memungkinkan decision tree menjadi alat yang efektif dalam analisis data, baik untuk klasifikasi maupun regresi.

2. Pengenalan Gini Impurity 

![image](https://github.com/user-attachments/assets/5675d166-8931-448f-85d3-bde0fa019b98)

![image](https://github.com/user-attachments/assets/a46e638f-c110-40d5-9ceb-cff35b9177d8)

![image](https://github.com/user-attachments/assets/ed0b8e2b-2056-4008-84fd-74b81fe25526)

![image](https://github.com/user-attachments/assets/191f25e1-b43f-4fee-ae75-4ab92543efc9)

Gini Impurity adalah metode dalam algoritma Decision Tree untuk mengukur ketidakkemurnian (impurity) data, menunjukkan seberapa sering elemen akan salah klasifikasi jika diklasifikasikan berdasarkan distribusi label. Nilainya berkisar antara 0 (murni, semua data dalam satu kelas) hingga 1 (sangat heterogen). Rumusnya adalah G=1−∑i=1npi2G=1−∑i=1n​pi2​, di mana pipi​ adalah proporsi elemen dalam kelas ii. Misalnya, jika sebuah node memiliki 70% data di kelas A dan 30% di kelas B, Gini Impurity-nya adalah G=1−(0,72+0,32)=0,42G=1−(0,72+0,32)=0,42. Dalam algoritma seperti CART, fitur dengan Gini Impurity terkecil dipilih untuk menghasilkan node yang lebih murni dan pohon keputusan yang lebih akurat.

3. Pengenalan Information Gain 

![image](https://github.com/user-attachments/assets/767ff942-542b-45cd-8bec-2bcbe4f36e35)

Information Gain adalah konsep dalam algoritma Decision Tree yang digunakan untuk mengukur seberapa besar suatu atribut dapat mengurangi ketidakkemurnian data setelah pembagian (split). Semakin besar Information Gain, semakin baik atribut tersebut dalam memisahkan data ke kelompok yang lebih homogen. Perhitungan Information Gain didasarkan pada perbedaan antara entropi sebelum dan sesudah pembagian data. Entropi, yang mengukur tingkat ketidakpastian dalam data, dihitung dengan rumus E(S)=−∑i=1npilog⁡2(pi)E(S)=−∑i=1n​pi​log2​(pi​), di mana pipi​ adalah proporsi data dalam kelas ii. Setelah pembagian data, entropi dihitung ulang untuk setiap subset, dan Information Gain dihitung dengan rumus IG=E(S)−∑j=1m∣Sj∣∣S∣E(Sj)IG=E(S)−∑j=1m​∣S∣∣Sj​∣​E(Sj​), di mana SjSj​ adalah subset data hasil pembagian, dan ∣Sj∣/∣S∣∣Sj​∣/∣S∣ adalah proporsi data di subset tersebut. Sebagai contoh, jika entropi awal dataset adalah 0,940,94 dan entropi setelah pembagian data menjadi 0,40,4, maka Information Gain-nya adalah 0,540,54. Dalam algoritma seperti ID3, atribut dengan Information Gain terbesar dipilih untuk pembagian, menghasilkan pohon keputusan yang lebih efisien dan akurat.

4. Membangun Decision Tree 

![image](https://github.com/user-attachments/assets/846628d5-d281-455e-b4b1-f10ae6b9c5a6)

pendekatan Gini Impurity dan Information Gain. Dataset awal terdiri dari atribut Color (warna), Diameter (ukuran), dan Label (jenis buah seperti Apple, Grape, dan Lemon). Beberapa kemungkinan pembagian data (splitting) dievaluasi, seperti "Apakah Color = Green?", "Apakah Diameter < 3?", dan sebagainya, dengan tujuan memisahkan data ke dalam kelompok yang lebih homogen. Gini Impurity dihitung menggunakan rumus G=1−∑(Pi2)G=1−∑(Pi2​), di mana PiPi​ adalah proporsi elemen dalam setiap kelas. Pada contoh, probabilitas untuk masing-masing label adalah P(Apple)=2/5P(Apple)=2/5, P(Grape)=2/5P(Grape)=2/5, dan P(Lemon)=1/5P(Lemon)=1/5, sehingga G=1−(252+252+152)=0,63G=1−(52​2+52​2+51​2)=0,63. Setelah menghitung Gini Impurity untuk node awal dan setiap kemungkinan pembagian, Information Gain dihitung, dan pembagian dengan Information Gain tertinggi dipilih untuk memisahkan data. Proses ini dilakukan berulang kali hingga menghasilkan pohon keputusan dengan cabang-cabang yang optimal, di mana setiap leaf node merepresentasikan kelas akhi

5. Persiapan dataset: Iris Dataset 

![image](https://github.com/user-attachments/assets/23f9d851-3f60-4863-a4f7-ec6753987b2d)


6. Training model Decision Tree Classifier 

![image](https://github.com/user-attachments/assets/f8755628-030f-4e5c-b481-980e341f328f)


7. Visualisasi model Decision Tree 

![image](https://github.com/user-attachments/assets/faff4989-9be8-47b2-b304-15d308a8a9b3)


8. Evaluasi model Decision Tree 

![image](https://github.com/user-attachments/assets/db92d831-7e98-4d74-b90f-792a0f45e1f5)


0. Lakukan praktik dari https://youtu.be/yKovaQ6tyV8?si=HnHG6kcoCsDwvo\_0 , buat screenshot dengan nama kalian pada coding, kumpulkan dalam bentuk pdf, dari kegiatan ini: 
1. Proses training model Machine Learning secara umum 

![image](https://github.com/user-attachments/assets/44023d06-bfd0-40ed-af86-ca781891a74b)

Proses training model machine learning dimulai dengan pengumpulan data yang relevan, lalu dilakukan preprocessing untuk membersihkan dan menormalkan data. Data kemudian dibagi menjadi training set dan test set. Model dipilih sesuai dengan jenis masalah, seperti regresi atau klasifikasi, dan dilatih menggunakan data tersebut untuk meminimalkan kesalahan. Setelah itu, model dievaluasi menggunakan data yang belum dilihat sebelumnya untuk mengukur kinerjanya. Jika perlu, model dituning dengan mengubah hyperparameters untuk meningkatkan performa. Setelah diuji dan menunjukkan hasil yang baik, model siap diimplementasikan dan perlu dipantau serta diperbarui secara berkala.

2. Pengenalan Ensemble Learning 

![image](https://github.com/user-attachments/assets/ac4b6f31-f065-42db-9940-6d9e8accc0ab)

Ensemble Learning adalah pendekatan dalam machine learning yang menggabungkan beberapa model untuk meningkatkan akurasi dibandingkan model tunggal. Metode utama dalam ensemble learning meliputi Bagging (seperti Random Forest) yang menggabungkan model dari subset data acak, Boosting (seperti AdaBoost dan XGBoost) yang melatih model bertahap untuk memperbaiki kesalahan sebelumnya, dan Stacking, yang menggunakan model lain untuk menggabungkan prediksi dari berbagai model. Pendekatan ini dapat mengurangi overfitting dan meningkatkan kinerja secara keseluruhan.

3. Pengenalan Bootstrap Aggregating | Bagging 

![image](https://github.com/user-attachments/assets/965152ce-d12d-47a7-9b6c-14ea91d439d6)

Bootstrap Aggregating, atau yang lebih dikenal dengan sebutan Bagging, adalah teknik ensemble learning yang bertujuan untuk meningkatkan akurasi model dengan mengurangi varians dan menghindari overfitting. Pada metode ini, beberapa model dilatih secara paralel pada subset data yang diambil secara acak dengan penggantian (bootstrap), yang berarti beberapa data mungkin muncul lebih dari sekali dalam setiap subset. Setelah model-model tersebut dilatih, prediksi akhir dihasilkan dengan menggabungkan hasil prediksi dari semua model, biasanya dengan cara rata-rata untuk regresi atau voting mayoritas untuk klasifikasi. Salah satu contoh paling terkenal dari teknik Bagging adalah Random Forest, yang menggabungkan banyak pohon keputusan untuk membuat prediksi yang lebih stabil dan akurat. Bagging sangat efektif dalam meningkatkan kinerja model yang cenderung memiliki varians tinggi, seperti pohon keputusan

4. Pengenalan Random Forest | Hutan Acak 

![image](https://github.com/user-attachments/assets/96b33aaf-d887-44ee-ba99-a8459d67c072)

Random Forest (Hutan Acak) adalah algoritma ensemble learning yang menggunakan teknik bagging untuk meningkatkan akurasi dengan membangun banyak pohon keputusan secara acak. Setiap pohon dilatih pada subset data yang berbeda, dan hanya subset acak dari fitur yang dipertimbangkan pada setiap pemisahan. Prediksi akhir diambil dengan suara mayoritas (klasifikasi) atau rata-rata (regresi) dari semua pohon. Random Forest efektif dalam menangani data besar dan kompleks, serta mengurangi overfitting yang sering terjadi pada pohon keputusan tunggal.

5. Persiapan dataset | Iris Flower Dataset 

![image](https://github.com/user-attachments/assets/4621b7a9-1102-48f1-b5d5-b503067a1203)


6. Implementasi Random Forest Classifier dengan Scikit Learn 

![image](https://github.com/user-attachments/assets/7663b7b0-eda3-4fdb-bb00-94b09608d6f5)


7. Evaluasi model  dengan Classification Report 

 ![image](https://github.com/user-attachments/assets/67c345f3-f07c-4218-a9d5-7bcf387de24b)

