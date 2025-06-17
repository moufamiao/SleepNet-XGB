import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pre_data.loadedf


# 1. 数据标准化工具
class EEGPreprocessor:
    def __init__(self, train_mean=None, train_std=None):
        self.train_mean = train_mean
        self.train_std = train_std

    def normalize(self, X, eps=1e-8):
        if self.train_mean is None:
            self.train_mean = np.mean(X, axis=(0, 1))
            self.train_std = np.std(X, axis=(0, 1)) + eps
        return (X - self.train_mean) / self.train_std

# 2. 数据集封装
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        signal = self.X[idx]
        return torch.FloatTensor(signal), torch.LongTensor([self.y[idx]]).squeeze()

# 3. 深度模型（无attention）
import torch.nn as nn
class EEGHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(3),
            nn.Conv1d(32, 64, kernel_size=15),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=False,
            dropout=0.6,
            batch_first=True
        )
        self.lstm_norm = nn.LayerNorm(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(128, 5)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param)
                    elif 'weight_hh' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        # 前向用于分类，不用于特征提取
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        # 用最后一帧输出作为表征
        last_output = lstm_out[:, -1, :]  # (batch, 128)
        return self.classifier(last_output)

# 4. 特征提取函数（直接取 LSTM 最后一帧输出）
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extract features"):
            inputs = inputs.to(device)
            x = inputs.permute(0, 2, 1)
            x = model.cnn(x)
            x = x.permute(0, 2, 1)
            lstm_out, (h_n, c_n) = model.lstm(x)
            lstm_out = model.lstm_norm(lstm_out)
            last_output = lstm_out[:, -1, :]  # (batch, 128)
            feats.append(last_output.cpu().numpy())
            labels.append(targets.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels

# 5. 主函数
def main():
    # ---- 数据准备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    X, y = pre_data.loadedf.loaddata()
    X = np.transpose(X, (0, 2, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, stratify=y, random_state=42
    )

    preprocessor = EEGPreprocessor()
    X_train_norm = preprocessor.normalize(X_train)
    X_test_norm = (X_test - preprocessor.train_mean) / preprocessor.train_std

    batch_size = 256
    train_dataset = EEGDataset(X_train_norm, y_train)
    test_dataset = EEGDataset(X_test_norm, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ---- 加载模型 ----
    model = EEGHybridModel().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # ---- 特征提取 ----
    train_feats, train_labels = extract_features(model, train_loader, device)
    test_feats, test_labels = extract_features(model, test_loader, device)

    # ---- Optuna贝叶斯优化 XGBoost ----
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 80, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 3.0),
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        model_xgb = xgb.XGBClassifier(**params)
        model_xgb.fit(train_feats, train_labels)
        preds = model_xgb.predict(test_feats)
        score = f1_score(test_labels, preds, average='macro')
        return score

    print("开始贝叶斯调参（Optuna）...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # 试验次数可加大

    print("\n最优F1-score:", study.best_trial.value)
    print("最优参数:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # ---- 用最优参数重新训练和评估 ----
    best_params = study.best_trial.params
    best_params['tree_method'] = 'hist'
    best_params['eval_metric'] = 'mlogloss'
    best_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_xgb = xgb.XGBClassifier(**best_params)
    best_xgb.fit(train_feats, train_labels)
    best_pred = best_xgb.predict(test_feats)

    print("\n最终准确率:", accuracy_score(test_labels, best_pred))
    print(classification_report(test_labels, best_pred, digits=4))

    cm = confusion_matrix(test_labels, best_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Optuna Bayes + XGBoost Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    main()
