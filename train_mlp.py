"""
MLP (невронна мрежа) за ЕКГ класификация
==========================================
Ползва СЪЩИТЕ 239 признака като Random Forest pipeline-а -> чисто сравнение.
Разликата спрямо train_arrhythmia.py:
  * Класификатор: MLPClassifier (многослоен перцептрон) вместо Random Forest
  * Балансиране: претеглена загуба (sample_weight) вместо SMOTE - при
    невронни мрежи това обикновено дава по-добър recall за редките класове
    без да "залива" модела със синтетични примери.

Изисква вече обновените train_arrhythmia.py и utility.py (239 признака).
Импортира логиката за извличане директно от train_arrhythmia, за да няма
дублиране и разминаване.

Изпълнение:
    python train_mlp.py --data_dir /път/до/mit-bih --out_dir artifacts_mlp
"""

import os
import argparse
import numpy as np
import joblib
from collections import Counter

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Преизползваме готовата логика за извличане на признаци и AAMI групите,
# за да е ГАРАНТИРАНО същото като при Random Forest.
from train_arrhythmia import process_records, AAMI_GROUPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', default='artifacts_mlp')
    ap.add_argument('--lead', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    record_numbers = ['100','101','103','105','106','108','109','111','112','113',
                      '114','115','116','117','118','119','121','122','123','124',
                      '200','201','202','203','205','207','208','209','210','212',
                      '213','214','215','219','220','221','222','223','228','230',
                      '231','232','233','234']
    record_numbers = [r for r in record_numbers
                      if os.path.exists(os.path.join(args.data_dir, r + '.dat'))]
    if not record_numbers:
        raise FileNotFoundError(f"Няма намерени записи в {args.data_dir}")

    n_train = int(0.8 * len(record_numbers))
    train_records = record_numbers[:n_train]
    test_records = record_numbers[n_train:]
    print(f"Записи за трениране: {len(train_records)} | за тест: {len(test_records)}")

    print("Извличане на характеристики (train)...")
    X_train, y_train = process_records(train_records, args.data_dir, args.lead)
    print("Извличане на характеристики (test)...")
    X_test, y_test = process_records(test_records, args.data_dir, args.lead)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print("Разпределение (train):", Counter(y_train))
    print("Разпределение (test): ", Counter(y_test))

    # -- Кодиране на етикетите --
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    print("Класове:", list(le.classes_))

    # -- Скалиране (задължително за невронни мрежи!) --
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # -- Балансиране без SMOTE --
    # Вместо синтетични данни, oversample-ваме реалните примери на редките
    # класове чрез повтаряне (виж _balanced_indices). При невронни мрежи това
    # обикновено дава по-добър recall за S/F/Q без изкуствени артефакти.

    # -- MLP архитектура --
    # Два скрити слоя (128, 64). early_stopping спира, когато валидацията
    # спре да се подобрява -> предпазва от преобучение.
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,                 # L2 регуляризация
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=True,
    )

    print("Трениране на MLP...")
    # Балансиране чрез повтаряне на редките класове по индекси
    # (ръчен oversampling - без синтетични данни като SMOTE).
    idx = _balanced_indices(y_train_enc, random_state=42)
    clf.fit(X_train_s[idx], y_train_enc[idx])

    # -- Оценка --
    y_pred = clf.predict(X_test_s)
    print("\n===== Classification Report (MLP) =====")
    print(classification_report(y_test_enc, y_pred,
                                target_names=le.classes_, zero_division=0))
    print("===== Confusion Matrix =====")
    print("Редове=истина, колони=предсказано; ред/колона =", list(le.classes_))
    print(confusion_matrix(y_test_enc, y_pred))

    # -- Запазване --
    joblib.dump(clf, os.path.join(args.out_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(args.out_dir, 'label_encoder.pkl'))
    print(f"\nЗапазени: model.pkl, scaler.pkl, label_encoder.pkl в {args.out_dir}/")


def _balanced_indices(y, random_state=42):
    """
    Връща индекси, при които всеки клас е oversample-нат до размера
    на най-големия клас (чрез повтаряне на реални примери, не синтетични).
    По-леко от SMOTE и добре работи с невронни мрежи.
    """
    rng = np.random.default_rng(random_state)
    counts = Counter(y)
    max_n = max(counts.values())
    all_idx = []
    for cls in counts:
        cls_idx = np.where(y == cls)[0]
        # повтаряме с връщане до max_n
        chosen = rng.choice(cls_idx, size=max_n, replace=True)
        all_idx.append(chosen)
    out = np.concatenate(all_idx)
    rng.shuffle(out)
    return out


if __name__ == '__main__':
    main()