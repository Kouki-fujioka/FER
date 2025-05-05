from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('model.h5')  # モデルのロード

img_path = './test/neutral/PrivateTest_59059.jpg' # 画像のパス
img = image.load_img(
    img_path,   # 画像のパス
    target_size = (224, 224)    # model.h5 に適した入力サイズ (高さ, 幅) に変換
)
img_array = image.img_to_array(img) # img という画像オブジェクトを numpy 配列 (高さ, 幅, カラーチャンネル) に変換
img_array = np.expand_dims(img_array, axis = 0) # img_array 配列の先頭に次元 (値 = 1) を追加
img_array = img_array / 255.0   # 画像のピクセル値 (0 ~ 255) を再スケーリング (0 ~ 1)
"""
画像を自動的に RGB(カラーチャンネル = 3) に設定
画像を手動的に numpy 配列に変換 (224, 224, 3)
バッチサイズ用の次元 (値 = 1) を追加して4次元配列を作成 (1, 224, 224, 3)
"""

# ラベルを手動で設定 (モデル学習時と同等)
class_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

predict = model.predict(img_array)  # 確率分布を取得
predict = [f'{p:.10f}' for p in predict[0]] # 確率分布を指数表記から少数表記に変換
predict_class = np.argmax(predict)  # 確率分布で最も高い確率値を持つクラスを取得
predict_label = class_labels[predict_class] # 確率分布で最も高い確率値を持つクラスのラベルを取得
if predict_label == 'neutral':  # ラベルが neutral の場合
    predict_min = 0 # 最小値
    print(f'Predict label:{predict_label}') # ラベルを表示
    print(f'Predict min:{predict_min}') # 最小値を表示
else:   # ラベルが neutral 以外の場合
    predict_max = predict[predict_class]    # 確率分布で最も高い確率値を取得
    print(f'Predict label:{predict_label}') # ラベルを表示
    print(f'Predict max:{predict_max}') # 確率分布で最も高い確率値を表示
