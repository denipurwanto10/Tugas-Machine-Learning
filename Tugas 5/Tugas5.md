Nama : Deni Purwanto            
NPM : 41155050210017   
Kelas : INF A  
Mata Kuliah : Machine Learning  



1.0.	K-Nearest Neighbours (KNN). Lakukan praktik dari https://youtu.be/4zARMcgc7hA?si=x6RoHQXFF4NY76X8 , buat screenshot dengan nama kalian pada coding, kumpulkan dalam bentuk pdf, dari kegiatan ini:
1.1.	Persiapan sample dataset
 ![Screenshot 2024-11-08 185104](https://github.com/user-attachments/assets/68cf41c5-d73b-498b-9fa0-30f3e231aaa8)
        
1.2.	Visualisasi dataset
![Screenshot 2024-11-08 185428](https://github.com/user-attachments/assets/a50611fb-be7f-4b0a-84a3-2e83c243fde7)

1.3.	Pengantar classification dengan K-Nearest Neighbours | KNN
Algoritma K-Nearest Neighbor (KNN) adalah algoritma machine learning yang bersifat non-parametrik dan lazy learning. Sebagai algoritma non-parametrik, KNN tidak membuat asumsi apapun tentang distribusi data yang mendasarinya. Ini berarti, model KNN tidak memiliki jumlah parameter tetap atau estimasi parameter tertentu, baik itu untuk data yang kecil maupun besar. Dalam algoritma non-parametrik seperti KNN, jumlah parameter bisa bervariasi dan cenderung meningkat seiring dengan bertambahnya jumlah data. Meskipun secara komputasi lebih lambat, algoritma non-parametrik seperti KNN memiliki keuntungan karena lebih sedikit membuat asumsi tentang struktur data.

Selain itu, KNN termasuk dalam kategori lazy learning. Artinya, algoritma ini tidak secara eksplisit membangun model pada fase pelatihan (training), melainkan hanya menyimpan data pelatihan dan melakukan perhitungan saat data baru (data uji) dimasukkan. Dengan kata lain, tidak ada fase pelatihan yang kompleks, dan proses perhitungan baru dilakukan saat prediksi harus dibuat.

1.4.	Preprocessing dataset dengan Label Binarizer
![Screenshot 2024-11-08 185744](https://github.com/user-attachments/assets/8c65b6cf-2878-427b-a311-3e298a83c95c)

1.5.	Training KNN Classification Model
![Screenshot 2024-11-08 191238](https://github.com/user-attachments/assets/b07ed4fc-2c93-4e1e-917c-42a9eb2af8f0)
![Screenshot 2024-11-08 191423](https://github.com/user-attachments/assets/d65c15d5-e992-4a00-9f8a-f69fa01e94a9)

1.6.	Prediksi dengan KNN Classification Model
![Screenshot 2024-11-08 191651](https://github.com/user-attachments/assets/bbad4996-64ca-4bb6-9e74-de4e3f77e57a)

1.7.	Visualisasi Nearest Neighbours
![Screenshot 2024-11-08 191915](https://github.com/user-attachments/assets/2634db74-8886-4d7e-8cd1-83cd42d4a14e)
![Screenshot 2024-11-08 191926](https://github.com/user-attachments/assets/1447d32e-3a22-4fc3-b2bd-9bf6b394bc1b)
 
1.8.	Kalkulasi jarak dengan Euclidean Distance
![Screenshot 2024-11-08 192733](https://github.com/user-attachments/assets/d1f36d42-a453-4874-9168-59cd5f7054ec)
![Screenshot 2024-11-08 192741](https://github.com/user-attachments/assets/e0944122-6a8b-4979-9916-7fc9ffa0e5ea)
 
1.9.	Evaluasi KNN Classification Model | Persiapan testing set
![Screenshot 2024-11-08 193022](https://github.com/user-attachments/assets/da7f0f52-8ef3-4ae2-8d4d-010c782c8851)

1.10.	Evaluasi model dengan accuracy score
![Screenshot 2024-11-08 193118](https://github.com/user-attachments/assets/81fce5b0-1ecd-4295-94d6-88329d581db1)
 
1.11.	Evaluasi model dengan precision score
![Screenshot 2024-11-08 193235](https://github.com/user-attachments/assets/100a1d51-e3c1-4223-883c-06c9981fd262)
 
1.12.	Evaluasi model dengan recall score
![Screenshot 2024-11-08 193312](https://github.com/user-attachments/assets/db2cd781-59a2-4716-9b42-39332ef711fe)
 
1.13.	Evaluasi model dengan F1 score
![Screenshot 2024-11-08 193347](https://github.com/user-attachments/assets/e1b26cdd-a7d6-4a74-9a27-3d16b8ac54c7)
 
1.14.	Evaluasi model dengan classification report
![Screenshot 2024-11-08 193648](https://github.com/user-attachments/assets/bff45094-c4ee-4ac6-9c52-baadb7bfd6a1)
 
1.15.	Evaluasi model dengan Mathews Correlation Coefficient
![Screenshot 2024-11-08 193834](https://github.com/user-attachments/assets/24f34aa4-8aa1-4a12-8aaf-4be8bd0a90b5)


2.0.	Support Vector Machine (SVM). Lakukan praktik dari https://youtu.be/z69XYXpvVrE?si=KR_hDSlwjGIMcT0w , buat screenshot dengan nama kalian pada coding, kumpulkan dalam bentuk pdf, dari kegiatan ini:

2.1.	Pengenalan Decision Boundary & Hyperplane
![Screenshot 2024-11-08 201019](https://github.com/user-attachments/assets/62ae3c33-599b-4d9d-87d9-c64c187d0971)
Decision boundary adalah garis atau permukaan yang memisahkan ruang fitur menjadi dua wilayah, masing-masing untuk kelas yang berbeda. Hyperplane adalah istilah yang digunakan untuk menyebut bidang atau batas yang memisahkan kedua kelas dalam ruang fitur berdimensi lebih tinggi. Sementara itu, margin merujuk pada jarak antara hyperplane dan titik data terdekat dari masing-masing kelas, yang berfungsi sebagai "lebar jalan" yang membatasi kedua kelas tersebut.
 
2.2.	Pengenalan Support Vector & Maximum Margin
![Screenshot 2024-11-08 201055](https://github.com/user-attachments/assets/6c27c2cd-6882-4556-8025-719d6589df7c)
Maximum margin adalah pendekatan dalam pemodelan linier yang memisahkan kelas dalam suatu set data dengan jarak terjauh dari cangkang cembung (convex hull) setiap kelas. Pemisahan ini memastikan bahwa margin, yaitu jarak antara hyperplane dan titik data terdekat dari masing-masing kelas, maksimal. Selain itu, garis pemisah (hyperplane) tersebut harus tegak lurus terhadap garis terpendek yang menghubungkan kedua kelas.
 
2.3.	Pengenalan kondisi Linearly Inseparable dan Kernel Tricks
![Screenshot 2024-11-08 201109](https://github.com/user-attachments/assets/56f4d222-7b07-4d6f-b07f-7f15f7bd62cc)
Dua set titik data dalam ruang dua dimensi dikatakan dapat dipisahkan secara linier jika keduanya dapat dipisahkan sepenuhnya oleh satu garis lurus. Secara umum, dua kelompok titik data dalam ruang n-dimensi dapat dipisahkan secara linier jika keduanya dapat dipisahkan oleh sebuah hyperplane berdimensi n-1. Teknik kernel adalah metode yang memungkinkan SVM untuk menangani masalah klasifikasi non-linier dengan memetakan data input ke ruang fitur berdimensi lebih tinggi secara implisit. Dengan cara ini, SVM dapat menemukan hyperplane yang memisahkan kelas-kelas data yang berbeda di ruang dimensi yang lebih tinggi.
 
2.4.	Pengenalan MNIST Handwritten Digits Dataset
![Screenshot 2024-11-08 194742](https://github.com/user-attachments/assets/0d2c11d2-0de1-4ed7-8372-d6805c69a5e8)
![Screenshot 2024-11-08 194752](https://github.com/user-attachments/assets/b500eeff-3160-4e36-972e-ed2175b7e1d1)
![Screenshot 2024-11-08 194818](https://github.com/user-attachments/assets/85c17e1c-02b3-4ea9-95c4-d744c7c67714)
 
2.5.	Klasifikasi dengan Support Vector Classifier | SVC
![Screenshot 2024-11-08 195728](https://github.com/user-attachments/assets/5bc9fef9-5e21-497f-b743-fd72cf8a5957)
![Screenshot 2024-11-08 195739](https://github.com/user-attachments/assets/23f3ad33-4105-40b2-81d3-fb81e73f391c)
 
2.6.	Hyperparameter Tuning dengan Grid Search
![Screenshot 2024-11-08 200636](https://github.com/user-attachments/assets/dace907d-fdf0-4978-a37d-3606febf664d)
![Screenshot 2024-11-08 200838](https://github.com/user-attachments/assets/150ffc16-c903-4537-ae46-74dba32c5c03)
 
2.7.	Evaluasi Model
![Screenshot 2024-11-08 200954](https://github.com/user-attachments/assets/52d999b6-0762-422f-8a4f-4fe1d9a367fb)

 

