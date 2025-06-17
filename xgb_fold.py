import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pre_data.loadedf
import pre_data.loadedf_78
import pre_data.loadedf_shhs
import xgboost as xgb
import optuna
from imblearn.metrics import geometric_mean_score

# -------------------- 预处理器 --------------------
class EEGPreprocessor:
    def __init__(self, train_mean=None, train_std=None):
        self.train_mean = train_mean
        self.train_std = train_std
    def normalize(self, X, eps=1e-8):
        if self.train_mean is None:
            self.train_mean = np.mean(X, axis=(0, 1))
            self.train_std = np.std(X, axis=(0, 1)) + eps
        return (X - self.train_mean) / self.train_std

# -------------------- 数据集 --------------------
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        signal = self.X[idx]
        return torch.FloatTensor(signal), torch.LongTensor([self.y[idx]]).squeeze()

# -------------------- 模型 --------------------
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
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        last_output = lstm_out[:, -1, :]  # (batch, 128)
        return self.classifier(last_output)

# -------------------- 特征提取函数 --------------------
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extract features", leave=False):
            inputs = inputs.to(device)
            x = inputs.permute(0, 2, 1)
            x = model.cnn(x)
            x = x.permute(0, 2, 1)
            lstm_out, (h_n, c_n) = model.lstm(x)
            lstm_out = model.lstm_norm(lstm_out)
            last_output = lstm_out[:, -1, :]
            feats.append(last_output.cpu().numpy())
            labels.append(targets.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels

# -------------------- 训练器 --------------------
class Trainer:
    def __init__(self, model, device, preprocessor):
        self.model = model.to(device)
        self.device = device
        self.preprocessor = preprocessor

    def train(self, train_loader, val_loader, epochs):
        optimizer = optim.AdamW([
            {'params': self.model.cnn.parameters(), 'lr': 1e-4},
            {'params': self.model.lstm.parameters(), 'lr': 1e-3},
            {'params': self.model.classifier.parameters(), 'lr': 5e-3}
        ], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
            for inputs, labels in progress:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.squeeze())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                progress.set_postfix({'loss': loss.item()})
            val_loss, val_acc, val_metrics, _, _ = self.evaluate(val_loader)
            scheduler.step(val_loss)
            if val_acc > best_acc:
                best_acc = val_acc
            # 这里加打印，每一轮都输出！
            print(
                f"  [Epoch {epoch + 1:2d}/{epochs}] "
                f"Train Loss: {train_loss / len(train_loader.dataset):.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Acc: {val_acc:.4f} | "
                f"Kappa: {val_metrics['k']:.4f} | "
                f"MF1: {val_metrics['mf1']:.4f} | "
                f"MGM: {val_metrics['mgm']:.4f}"
            )
        return best_acc

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        all_preds = []
        all_labels = []
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.squeeze())
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels.squeeze()).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / len(loader.dataset)
        mf1 = f1_score(all_labels, all_preds, average='macro')
        k = cohen_kappa_score(all_labels, all_preds)
        mgm = geometric_mean_score(all_labels, all_preds, average='macro')
        return avg_loss, accuracy, {'mf1': mf1, 'k': k, 'mgm': mgm}, all_labels, all_preds

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X, y = pre_data.loadedf.loaddata()
    X = np.transpose(X, (0, 2, 1))

    kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    all_acc, all_kappa, all_mf1, all_mgm = [], [], [], []
    all_true, all_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n========== Fold {fold+1}/20 ==========")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 数据标准化
        preprocessor = EEGPreprocessor()
        X_train_norm = preprocessor.normalize(X_train)
        X_val_norm = (X_val - preprocessor.train_mean) / preprocessor.train_std

        train_dataset = EEGDataset(X_train_norm, y_train)
        val_dataset = EEGDataset(X_val_norm, y_val)
        batch_size = 512
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

        # 训练深度模型
        model = EEGHybridModel()
        trainer = Trainer(model, device, preprocessor)
        trainer.train(train_loader, val_loader, epochs=65)  # 可调epochs

        # 特征提取
        train_feats, train_labels = extract_features(trainer.model, train_loader, device)
        val_feats, val_labels = extract_features(trainer.model, val_loader, device)

        #贝叶斯优化XGBoost
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 80, 200),
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
            preds = model_xgb.predict(val_feats)
            return accuracy_score(val_labels, preds)  # 用准确率做目标！


        # def objective(trial):
        #     params = {
        #         'n_estimators': trial.suggest_int('n_estimators', 80, 200),
        #         'max_depth': trial.suggest_int('max_depth', 3, 10),
        #         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        #         'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        #         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        #         'gamma': trial.suggest_float('gamma', 0, 1.0),
        #         'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        #         'reg_lambda': trial.suggest_float('reg_lambda', 0, 3.0),
        #         'tree_method': 'hist',
        #         'eval_metric': 'mlogloss',
        #         'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        #     }
        #     model_xgb = xgb.XGBClassifier(**params)
        #     model_xgb.fit(train_feats, train_labels)
        #     preds = model_xgb.predict(val_feats)
        #     return f1_score(val_labels, preds, average='macro')


        print("  Optuna贝叶斯优化XGBoost...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=40)  # 可调更大更慢

        best_params = study.best_trial.params
        best_params['tree_method'] = 'hist'
        best_params['eval_metric'] = 'mlogloss'
        best_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        best_xgb = xgb.XGBClassifier(**best_params)
        best_xgb.fit(train_feats, train_labels)
        val_pred = best_xgb.predict(val_feats)

        acc = accuracy_score(val_labels, val_pred)
        mf1 = f1_score(val_labels, val_pred, average='macro')
        kappa = cohen_kappa_score(val_labels, val_pred)
        mgm = geometric_mean_score(val_labels, val_pred, average='macro')

        print(f"Fold {fold+1} | Acc: {acc:.4f} | Kappa: {kappa:.4f} | MF1: {mf1:.4f} | MGM: {mgm:.4f}")

        all_acc.append(acc)
        all_kappa.append(kappa)
        all_mf1.append(mf1)
        all_mgm.append(mgm)
        all_true.extend(val_labels)
        all_pred.extend(val_pred)

    print("\n======= 20折交叉验证最终统计结果 =======")
    print(f"平均准确率  (acc): {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"平均Kappa    : {np.mean(all_kappa):.4f} ± {np.std(all_kappa):.4f}")
    print(f"平均Macro F1 : {np.mean(all_mf1):.4f} ± {np.std(all_mf1):.4f}")
    print(f"平均G-mean   : {np.mean(all_mgm):.4f} ± {np.std(all_mgm):.4f}")

    # 打印混淆矩阵
    cm = confusion_matrix(all_true, all_pred)
    print("\n最终整体混淆矩阵：\n", cm)
    # plt.figure(figsize=(8, 6))
    # plt.imshow(cm, cmap="Blues")
    # plt.title("20-Fold CV Overall Confusion Matrix")
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.colorbar()
    # plt.show()
