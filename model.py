from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model

train_dir = './train'  # トレーニングデータが格納されているディレクトリへのパス
train_datagen = ImageDataGenerator(rescale = 1.0 / 255.0) # 画像のピクセル値 (0 ~ 255) を再スケーリング (0 ~ 1)
train_generator = train_datagen.flow_from_directory(
    train_dir,  # ディレクトリ (ラベル名はディレクトリの構造に基づいて自動的に取得)
    target_size = (224, 224),   # InceptionV3 に適した入力サイズ (高さ, 幅) に変換
    batch_size = 32,    # 一度にモデルに入力されるデータのサンプル数
    class_mode = 'categorical'  # 出力ラベルをカテゴリカル (多クラス分類) に設定
)
test_dir = './test'  # テストデータが格納されているディレクトリへのパス
test_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)    # 画像のピクセル値 (0 ~ 255) を再スケーリング (0 ~ 1)
test_generator = test_datagen.flow_from_directory(
    test_dir,   # ディレクトリ (ラベル名はディレクトリの構造に基づいて自動的に取得)
    target_size = (224, 224),   # InceptionV3 に適した入力サイズ (高さ, 幅) に変換
    batch_size = 32,    # 一度にモデルに入力されるデータのサンプル数
    class_mode = 'categorical'  # 出力ラベルをカテゴリカル (多クラス分類) に設定
)
"""
画像を自動的に RGB(カラーチャンネル = 3) に設定
画像を自動的に numpy 配列に変換 (224, 224, 3)
バッチサイズを設定して4次元配列を作成 (32, 224, 224, 3)
"""

base_model = InceptionV3(
    weights = 'imagenet',   # ImageNet データセット (画像認識の基本的な特徴を学習済み) で訓練された重みを使用
    include_top = False,    # 全結合層を除外し, 特徴抽出器として使用
    input_shape = (224, 224, 3) # InceptionV3 の入力層が受け取る単一画像の形状 (高さ, 幅, カラーチャンネル) を設定
)
"""
InceptionV3 の全結合層は ImageNet 1000クラス分類に対応
全結合層を除外すると畳み込み層の出力 (特徴マップ) がモデルの出力に対応
タスクに合わせた全結合層を追加する必要性
"""

x = base_model.output   # 畳み込み層 (特徴マップ) の出力を取得
x = GlobalAveragePooling2D()(x) # 各特徴マップの平均を計算
x = Dense(
    1024,   # 全結合層 (中間層) に含まれるニューロンの数 (出力ベクトルの次元)
    activation = 'relu' # ニューロンの出力を計算するために使用する非線形関数 (入力 (前層の出力) が0以下の場合は0, 0超の場合はそのまま出力)
)(x) # 全結合層 (中間層) を追加
x = Dense(
    train_generator.num_classes,    # 全結合層 (出力層) に含まれるニューロンの数 (出力ベクトルの次元)
    activation = 'softmax'  # ニューロンの出力を計算するために使用する非線形関数 (入力 (前層の出力) から分類タスク用の確率分布を出力)
)(x) # 全結合層 (出力層) を追加
"""
非線形関数を用いると線形関数では表現できない複雑な関係を学習可能
畳み込み層の出力:(バッチサイズ, 特徴マップの高さ, 特徴マップの幅, 特徴マップの数) https://zenn.dev/nekoallergy/articles/dl-advanced-featuremap-anchor
グローバル平均プーリング層の出力:(バッチサイズ, 特徴マップの高さ (値 = 1), 特徴マップの幅 (値 = 1), 特徴マップの数) https://cvml-expertguide.net/terms/dl/layers/pooling-layer/global-average-pooling/ https://qiita.com/mine820/items/1e49bca6d215ce88594a
全結合層 (中間層) の出力:(バッチサイズ, ニューロンの数) 特徴マップを1024次元に変換
全結合層 (出力層) の出力:(バッチサイズ, ラベル数) 

例) 3クラス (A, B, C) の分類タスク
出力層サイズ:3
Softmax 出力:[0.7, 0.2, 0.1]
モデルは入力データが A クラスに該当する確率が70%, B クラスに該当する確率が20%, C クラスに該当する確率が10%であると予測
最も高い確率値を持つクラス (A クラス) を予測ラベルとして選択
"""

model = Model(
    inputs = base_model.input,  # モデルの入力を InceptionV3 の入力に設定
    outputs = x # モデルの出力を分類タスク用の確率分布に設定
)
model.compile(
    optimizer = Adam(lr = 0.0001),  # Adam オプティマイザーを使用し, 学習率0.0001で重みを更新
    loss = 'categorical_crossentropy',  # 多クラス分類問題用の損失関数を使用
    metrics = ['accuracy']  # 正解ラベルと予測ラベルが一致する割合を計算する指標
)
history = model.fit(
    train_generator,    # トレーニングデータのジェネレーターを学習に使用
    epochs = 10,    # トレーニングデータ全体を10回繰り返し, 訓練を実行
)
loss, accuracy = model.evaluate(test_generator) # テストデータでモデルの性能を評価
print(f'Validation Loss:{loss}')   # テストデータに対する損失値を表示
print(f'Validation Accuracy:{accuracy}')   # テストデータに対する精度を表示
model.save('model.h5')  # 訓練後のモデルを Keras モデルの保存形式で保存
