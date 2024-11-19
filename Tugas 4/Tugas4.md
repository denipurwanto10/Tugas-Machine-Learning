**Nama : Deni Purwanto** 

**NPM : 41155050210017** 

**Kelas : INF A** 

**Mata Kuliah : Machine Learning** 

1\.0.  Lakukan praktik dari https://youtu.be/Sj1ybuDDf9I?si=hCajHe1zasTQ9HGY , buat screenshot dengan nama kalian pada coding, kumpulkan dalam bentuk pdf, dari kegiatan ini: 

1.1.Pengenalan Bayes Theorem | Teori Bayes | Conditional Probability 

![image](https://github.com/user-attachments/assets/7d8d37c8-c6e3-458e-8698-008ca12b5581)

eori Bayes (Bayes' Theorem) adalah konsep dasar dalam statistik dan probabilitas yang digunakan untuk menghitung probabilitas suatu peristiwa berdasarkan informasi baru atau bukti yang tersedia. Teori ini menggabungkan probabilitas bersyarat (conditional probability) untuk memperbarui keyakinan kita tentang suatu peristiwa setelah memperoleh data atau bukti baru. Secara matematis, Bayes' Theorem dinyatakan sebagai P(A∣B)=P(B∣A)⋅P(A)P(B)P(A∣B)=P(B)P(B∣A)⋅P(A)​, di mana P(A∣B)P(A∣B) adalah probabilitas terjadinya peristiwa A setelah mengetahui peristiwa B, P(B∣A)P(B∣A) adalah probabilitas terjadinya peristiwa B jika peristiwa A sudah terjadi, P(A)P(A) adalah probabilitas awal peristiwa A, dan P(B)P(B) adalah probabilitas terjadinya peristiwa B. Teori Bayes sering digunakan dalam aplikasi seperti klasifikasi, diagnosis medis, dan machine learning (terutama dalam algoritma Naive Bayes), karena memungkinkan model untuk memperbarui prediksi mereka berdasarkan data baru yang masuk.

1\.2.Pengenalan Naive Bayes Classification 

![image](https://github.com/user-attachments/assets/0c542fd1-72f9-4699-8ebe-b2c84f0c335b)

Naive Bayes Classification adalah algoritma klasifikasi berbasis probabilitas yang menggunakan Teori Bayes dengan asumsi bahwa semua fitur dalam data bersifat independen satu sama lain, yang disebut sebagai "naive" atau asumsi sederhana. Meskipun asumsi ini sering kali tidak realistis dalam banyak kasus, Naive Bayes tetap efektif dalam berbagai masalah klasifikasi, terutama ketika jumlah data besar dan hubungan antar fitur tidak terlalu kompleks. Algoritma ini bekerja dengan menghitung probabilitas posterior dari setiap kelas berdasarkan fitur-fitur yang ada, menggunakan rumus Bayes. Kemudian, kelas dengan probabilitas tertinggi akan dipilih sebagai prediksi. Naive Bayes sering digunakan dalam aplikasi seperti klasifikasi email (spam atau bukan spam), analisis sentimen, dan pengenalan teks, karena kemampuannya untuk bekerja dengan cepat dan efisien pada dataset besar.

1\.3.Pengenalan Prior Probability 

![image](https://github.com/user-attachments/assets/6e55a30c-337f-4377-877f-e6ebe8c44509)

Prior Probability adalah probabilitas awal suatu peristiwa atau hipotesis sebelum mempertimbangkan bukti atau informasi baru. Dalam konteks Teori Bayes, prior probability menggambarkan keyakinan kita terhadap suatu peristiwa berdasarkan pengetahuan yang ada sebelumnya, tanpa memperhitungkan data baru yang mungkin masuk. Prior probability sering kali diperoleh dari data historis atau informasi yang sudah ada, dan merupakan komponen penting dalam perhitungan probabilitas posterior setelah bukti baru dipertimbangkan. Dalam banyak aplikasi machine learning dan statistik, prior probability digunakan untuk memulai proses inferensi atau klasifikasi, yang kemudian akan diperbarui dengan informasi lebih lanjut menggunakan probabilitas bersyarat dan aturan Bayes.

1\.4.Pengenalan Likelihood 

![image](https://github.com/user-attachments/assets/624a90b2-4659-4fd1-9b92-609044740e86)

Likelihood dalam statistik merujuk pada probabilitas atau kemungkinan data yang diamati terjadi, mengingat model atau parameter tertentu. Dalam konteks Teori Bayes, likelihood digunakan untuk mengukur seberapa baik suatu model atau hipotesis dapat menjelaskan data yang diberikan. Ini dihitung dengan menggunakan rumus P(D∣θ)P(D∣θ), yang menyatakan probabilitas data DD diberikan parameter θθ. Likelihood sangat penting dalam proses estimasi parameter, di mana tujuan utamanya adalah menemukan nilai parameter yang memaksimalkan likelihood, suatu pendekatan yang dikenal sebagai Maximum Likelihood Estimation (MLE). Dalam machine learning, likelihood sering digunakan dalam berbagai algoritma, seperti dalam klasifikasi dan regresi, untuk memperbarui atau mempelajari parameter model berdasarkan data yang tersedia.

1\.5.Pengenalan Evidence | Normalizer 

![image](https://github.com/user-attachments/assets/0bf6d703-3f9b-431f-a96a-0efa27a458fb)

Evidence atau Normalizer dalam konteks Teori Bayes merujuk pada probabilitas dari data yang diamati, yang berfungsi untuk memastikan bahwa probabilitas posterior dari berbagai hipotesis atau kelas akan dijumlahkan menjadi satu. Dalam rumus Bayes, evidence sering kali diwakili dengan P(D)P(D), yang merupakan probabilitas total dari data DD tanpa mempertimbangkan kelas atau parameter tertentu. Evidence ini bertindak sebagai faktor normalisasi yang memastikan bahwa nilai-nilai probabilitas yang dihitung tetap valid dan terukur, sehingga semua probabilitas posterior berada dalam rentang yang benar (antara 0 dan 1). Dalam aplikasi nyata, nilai evidence sering kali tidak dihitung secara eksplisit karena sering kali merupakan konstanta yang sama untuk semua kelas atau hipotesis, namun perannya sangat penting dalam memastikan konsistensi perhitungan probabilitas dalam model probabilistik seperti klasifikasi Bayes.

1\.6.Pengenalan Posterior Probability 

![image](https://github.com/user-attachments/assets/0b71aa79-ca44-45f9-8740-f884dd48c0a2)

Posterior Probability adalah probabilitas suatu peristiwa atau hipotesis setelah mempertimbangkan bukti atau data baru. Dalam Teori Bayes, posterior probability dihitung menggunakan rumus Bayes, yaitu P(A∣B)=P(B∣A)⋅P(A)P(B)P(A∣B)=P(B)P(B∣A)⋅P(A)​, di mana P(A∣B)P(A∣B) adalah probabilitas posterior peristiwa AA setelah mengetahui bukti BB, P(B∣A)P(B∣A) adalah likelihood atau probabilitas data BB diberikan peristiwa AA, P(A)P(A) adalah prior probability, dan P(B)P(B) adalah evidence atau probabilitas total data. Posterior probability memberikan pembaruan pada keyakinan kita terhadap suatu hipotesis setelah data baru diperoleh, dan sering digunakan dalam aplikasi seperti klasifikasi, prediksi, dan inferensi statistika untuk membuat keputusan berdasarkan informasi yang lebih lengkap dan akurat.

1\.7.Studi kasus dan implementasi Naive Bayes 

![image](https://github.com/user-attachments/assets/3fe6e639-6da9-4fd7-920a-378eb4369283)

![image](https://github.com/user-attachments/assets/bb4d5bc6-592b-4ab2-b30e-3b53e58a8431)

![image](https://github.com/user-attachments/assets/c998fc2d-856a-40e6-8a7f-fa0d85179d58)

![image](https://github.com/user-attachments/assets/3f2a82af-b48e-4e66-b1f0-a4a62d345282)

![image](https://github.com/user-attachments/assets/2c3ca5c6-cb5c-46f3-b0b2-6bc808d3c251)

![image](https://github.com/user-attachments/assets/d75ed0d8-64aa-48a3-aaf2-293d7570cbec)




