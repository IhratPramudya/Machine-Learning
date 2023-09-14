import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

# Fashion MNIST dirancang untuk memiliki 60.000 gambar pelatihan dan 10.000 gambar uji. 
# Jadi, hasil dari data.load_dataakan memberi Anda larik berisi 60.000 larik berukuran 
# 28 × 28 piksel yang disebut training_images, dan larik berisi 60.000 nilai (0–9) yang 
# disebut training_labels. Demikian pula, test_imagesarray akan berisi 10.000 array 28 × 
# 28 piksel, dan array test_labelsakan berisi 10.000 nilai antara 0 dan 9.

(training_images, training_labels), (test_images, test_labels) = data.load_data()


# ular pitonmemungkinkan Anda melakukan operasi di seluruh array dengan notasi ini. 
# Ingatlah bahwa semua piksel dalam gambar kita adalah skala abu-abu, dengan nilai antara 0 dan 255. 
# Membaginya dengan 255 memastikan bahwa setiap piksel diwakili oleh angka antara 0 dan 1. 
# Proses ini disebut normalisasi gambar.
# Penjelasan mengapa data yang dinormalisasi lebih baik untuk melatih jaringan neural berada 
# di luar cakupan buku ini, namun perlu diingat saat melatih jaringan neural di TensorFlow 
# bahwa normalisasi akan meningkatkan performa.
training_images = training_images / 255.0
test_images = test_images / 255.0


# Itupertama, Flatten, bukanlah lapisan neuron, melainkan spesifikasi lapisan masukan. 
# Masukan kita adalah gambar berukuran 28 × 28, namun kita ingin agar gambar tersebut 
# diperlakukan sebagai rangkaian nilai numerik, seperti kotak abu-abu di bagian atas Gambar 2-5 . 
# Flattenmengambil nilai "persegi" (array 2D) dan mengubahnya menjadi garis (array 1D).

# Ituberikutnya, Dense, adalah lapisan neuron, dan kita menentukan bahwa kita menginginkan 128 lapisan neuron. 
# Ini adalah lapisan tengah yang ditunjukkan pada Gambar 2-5 . Anda akan sering mendengar lapisan seperti itu 
# digambarkan sebagai lapisan tersembunyi . Lapisan yang berada di antara masukan dan keluaran tidak terlihat 
# oleh pemanggil, sehingga istilah “tersembunyi” digunakan untuk menggambarkannya. Kami meminta 128 neuron 
# agar parameter internalnya diinisialisasi secara acak. Seringkali pertanyaan yang saya tanyakan saat ini 
# adalah “Mengapa 128?” Hal ini sepenuhnya arbitrer—tidak ada aturan tetap mengenai jumlah neuron yang akan 
# digunakan. Saat Anda mendesain lapisan, Anda ingin memilih jumlah nilai yang sesuai agar model 
# Anda benar-benar belajar. Lebih banyak neuron berarti ia akan berjalan lebih lambat, karena harus 
# mempelajari lebih banyak parameter. Lagineuron juga dapat menghasilkan jaringan yang hebat dalam mengenali 
# data pelatihan, namun tidak begitu baik dalam mengenali data yang belum pernah dilihat sebelumnya 
# (ini dikenal sebagai overfitting , dan kita akan membahasnya nanti di bab ini). Di sisi lain, lebih 
# sedikit neuron berarti model tersebut mungkin tidak memiliki parameter yang memadai untuk dipelajari.

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50)


classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])