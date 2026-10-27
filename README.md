# MirrorFace

ブラウザだけで顔の中心位置を検知し、左右それぞれを軸に反転した対称画像2枚を生成する静的Webアプリです。

## 使い方

1. `index.html` をブラウザで開きます（推奨: GitHub Pages / HTTPで配信）。
2. 顔写真を選択すると、MediaPipeが顔を自動検知します。
3. 「A 左側からつくる顔」「B 右側からつくる顔」が表示されます。
4. 必要なら元画像上の縦線をドラッグして軸を微調整します。
5. 個別の「PNGで保存」からダウンロードします。

複数の顔がある場合は検出したうち最大の顔が初期選択されます。「顔の選択」から別の顔も選べます。検知失敗時にも中心線の手動調整を使えます。

## GitHub Pagesへの公開方法

1. GitHubリポジトリを作成し、このフォルダの `index.html`・`style.css`・`app.js` をリポジトリのルートに置きます。
2. リポジトリの **Settings → Pages** に移動します。
3. **Build and deployment → Source → Deploy from a branch** を選びます。
4. **Branch → main / (root)** を選択して **Save** します。
5. 表示される `https://ユーザー名.github.io/リポジトリ名/` のURLにアクセスします。

ビルド、Node.js、サーバーAPI、APIキーは必要ありません。

## 技術仕様

- AI顔検知: MediaPipe Tasks Vision `@mediapipe/tasks-vision@0.10.32` (CDN)
- モデル: BlazeFace short range（Googleのモデル配信URLからダウンロード）
- 画像処理: HTML Canvas API
- 画像の長辺は最大3200ピクセルに縮小（ブラウザ負荷を抑えるため）
- 顔の中心座標: 顔検知バウンディングボックスの水平方向中央
- 対称出力: 顔中心X軸から左右同じ距離だけ画像を切り出し、片方を反転コピー
- 出力形式: PNG
- 画像のアップロードは行わず、処理はブラウザ内で完了します。AIライブラリとモデルの初回取得にはネット接続が必要です。MediaPipe自身による利用状況の計測についてはMediaPipeのプライバシー告知に従います。

## 注意

- 顔が傾いていたり、輪郭がはっきり見えない場合は、検出された中心が見た目の中心からずれることがあります。手動調整をご利用ください。
- HEIC/HEIFは一部のブラウザで読み込めません。読み込めない場合はJPEGかPNGに変換してください。
- カメラのEXIF回転補正はブラウザの画像デコード挙動に依存します。
- 顔画像は本人の許可を得て使用してください。

## ファイル

- `index.html` : UI
- `style.css` : レイアウト・スマホ対応
- `app.js` : 顔検知・画像処理・ダウンロード

CDN: https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/+esm
モデル: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
