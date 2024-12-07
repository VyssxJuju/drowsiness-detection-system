import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Đường dẫn dữ liệu
original_train_dir = r'D:\code\btl_xlyanh\.venv\drowsiness-detection\Open_Closed_Eyes_Dataset\train'
val_dir = r'D:\code\btl_xlyanh\.venv\drowsiness-detection\Open_Closed_Eyes_Dataset\val'

# Thông số mô hình
img_height, img_width = 48, 48
batch_size = 16  # Giảm batch size để tăng tốc trên tập dữ liệu nhỏ
epochs = 5       # Giảm số epoch để thử nghiệm nhanh

# Tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Tạo bộ dữ liệu
train_generator = train_datagen.flow_from_directory(
    original_train_dir,  # Sử dụng toàn bộ dữ liệu gốc
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Đầu ra nhị phân
])

# Compile mô hình
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    verbose=1  # Hiển thị tiến trình chi tiết
)

# Lưu mô hình
model.save('my_model.keras')
