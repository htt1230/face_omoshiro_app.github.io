# MirrorFace

ブラウザだけで顔の中心位置を検知し、左右それぞれを軸に反転した対称画像2枚を生成する静的Webアプリです。

## 使い方

1. `index.html` をブラウザで開きます（推奨: GitHub Pages / HTTPで配信）。
2. 顔写真を選択すると、MediaPipeが顔を自動検知します。
3. 写真選択パネルが消え、左右それぞれから作った対称画像が全幅に表示されます。
4. 結果中央の↔を左右にドラッグして反転位置を調整します。下のスライダーから角度も変更できます。
5. 各画像の保存アイコンからダウンロードします。「写真を変更」で別の写真を選べます。

複数の顔がある場合も、最初だけ最大の顔の中心に自動で合わせます。その後の位置調整は中央の↔、角度調整はスライダーで行います。検知中に手動で調整した場合は、その操作を優先します。↶ボタンで最初の位置と角度に戻せます。

## GitHub Pagesへの公開方法

1. GitHubリポジトリを作成し、このフォルダの `index.html`・`style.css`・`app.js` をリポジトリのルートに置きます。
2. リポジトリの **Settings → Pages** に移動します。
3. **Build and deployment → Source → Deploy from a branch** を選びます。
4. **Branch → main / (root)** を選択して **Save** します。
5. 表示される `https://ユーザー名.github.io/リポジトリ名/` のURLにアクセスします。

ビルド、Node.js、サーバーAPI、APIキーは必要ありません。

### 更新時のキャッシュ対策

`app.js` と `style.css` を変更したら、公開前に `python3 scripts/update-asset-versions.py` を実行してください。ファイル内容のハッシュを読み込みURLの `?v=` に設定します。変更したファイルだけタグが変わり、ブラウザは新しいURLから取得します。更新した `index.html` も一緒に公開してください。

`python3 scripts/update-asset-versions.py --check` で更新漏れを確認できます。

## 技術仕様

- AI顔検知: MediaPipe Tasks Vision `@mediapipe/tasks-vision@0.10.32` (CDN)
- モデル: BlazeFace short range（Googleのモデル配信URLからダウンロード）
- 画像処理: HTML Canvas API
- 画像の長辺は最大3200ピクセルに縮小（ブラウザ負荷を抑えるため）
- 顔の中心座標: 顔検知バウンディングボックスの水平方向中央
- 対称出力: 軸の左側全体・右側全体をそれぞれ反転コピー。角度0°ではAの幅は中心X座標の2倍、Bの幅は（元画像幅−中心X座標）の2倍。角度付きでは軸で分割した各領域の回転後の範囲から個別に出力サイズを計算
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
