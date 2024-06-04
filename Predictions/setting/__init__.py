from Predictions.setting.setting import Setting

STANDARD = Setting(10, "train-test", True, False, 70, 10)
TAX = Setting(None, "train-test", False, True, 66) # original
CAMARGO = Setting(5, "test-train", False, True, 70)
DIMAURO = Setting(None, "k-fold", False, True, 80, train_k=3)
SDL = Setting(10, "train-test", True, False, 70)