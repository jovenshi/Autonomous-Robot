# Sign model

`sign_svm.npz` is written by `tools/train_signs.py` on the laptop and read on
the robot by `perception/signs.load_sign_model` (path in `SIGN_MODEL_PATH`).
It holds the SVM `weights` (one row per class), `biases`, `classes`
(`SignType` names), `crop_px` and the `hog` settings it was trained with.
Retrain when the signs, the lighting or the front camera mount change.
