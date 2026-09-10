"""
Multi-class ECG Arrhythmia Classification (MIT-BIH)
====================================================
Клета, коректна версия на pipeline-а:
  * Сегментиране на всеки удар около R-пика (морфология)
  * Комбинирани характеристики: морфология на удара + локален RR контекст
  * Многокласова класификация по AAMI групи (не бинарна)
  * Правилно train/test разделяне по ЗАПИСИ (без изтичане между пациенти)
  * StandardScaler се тренира само на train и се ЗАПАЗВА
  * SMOTE се прилага само на train
  * Запазва: model, scaler, label encoder -> за deployment

Изпълнение:
    python train_arrhythmia.py --data_dir /път/до/mit-bih --out_dir artifacts
"""

import os
import argparse
import numpy as np
import wfdb
import joblib
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------------
# 1. AAMI групиране на анотациите
# ---------------------------------------------------------------------------
# MIT-BIH използва ~19 символа. Стандартът AAMI EC57 ги свежда до 5 клинично
# смислени супергрупи. Това запазва ВСИЧКИ патологии като отделни класове,
# вместо да ги слее в "не-нормален".
AAMI_GROUPS = {
    # N - Normal / bundle branch blocks (нормална проводимост)
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # S - Supraventricular ectopic (надкамерни)
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # V - Ventricular ectopic (камерни)
    'V': 'V', 'E': 'V',
    # F - Fusion (сливни удари)
    'F': 'F',
    # Q - Unknown / paced (неопределими / пейсмейкър)
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}

# Символи, които НЕ са удари (ритъм анотации, шум и т.н.) - пропускат се
NON_BEAT = set(['+', '~', '|', '"', '[', ']', '!', 'x', '(', ')', 'p', 't', 'u'])


# ---------------------------------------------------------------------------
# 2. Извличане на характеристики за един удар
# ---------------------------------------------------------------------------
def _p_wave_features(segment, fs, r_index):
    """
    Признаци за P-вълната - сегментът ПРЕДИ QRS комплекса.
    P-вълната отразява предсърдната деполяризация. При надкамерни
    екстрасистоли (S) тя често липсва, е с необичайна форма или
    сменена полярност - затова е силен разграничител S срещу N.
    P-областта е приблизително 50-220ms преди R-пика.
    """
    p_start = max(0, r_index - int(0.22 * fs))
    p_end = max(0, r_index - int(0.05 * fs))
    if p_end - p_start < 3:
        return [0.0, 0.0, 0.0, 0.0]
    p = segment[p_start:p_end]
    energy = float(np.sum(p ** 2))                       # има ли P-вълна изобщо
    p_ampl = float(np.max(p) - np.min(p))                # амплитуда
    p_max = float(np.max(p))                             # полярност/височина
    p_peak_pos = float(np.argmax(np.abs(p))) / len(p)    # позиция спрямо R
    return [energy, p_ampl, p_max, p_peak_pos]


def extract_beat_features(signal, r_peak, prev_r, next_r, fs,
                          local_mean_rr=None, win_before=120, win_after=110):
    """
    Връща вектор от характеристики за един удар:
      * морфология: сегмент от сигнала около R-пика (win_before+win_after проби)
      * RR контекст: предходен RR, следващ RR, съотношение
      * НОРМАЛИЗИРАН RR: отклонение на удара спрямо локалния ритъм.
        Ключов сигнал за надкамерни екстрасистоли (S) - идват "рано".
      * P-ВЪЛНА признаци: енергия/амплитуда/полярност на областта преди QRS.
        Разграничава надкамерни от нормални удари (P-вълната им се различава).
    signal: 1D масив (един лид)
    r_peak: индекс на R-пика
    prev_r, next_r: индекси на съседните R-пикове (за RR контекст)
    local_mean_rr: средният RR на последните ~10 удара (в секунди)
    win_before: разширен на 120 проби, за да улови P-вълната изцяло
    """
    start = r_peak - win_before
    end = r_peak + win_after
    # Отхвърляме удари твърде близо до краищата
    if start < 0 or end > len(signal):
        return None

    morphology = signal[start:end]

    # RR характеристики (в секунди)
    prev_rr = (r_peak - prev_r) / fs if prev_r is not None else 0.0
    next_rr = (next_r - r_peak) / fs if next_r is not None else 0.0
    rr_ratio = (prev_rr / next_rr) if next_rr > 0 else 0.0

    # --- Нормализирани RR признаци спрямо локалния ритъм ---
    if local_mean_rr and local_mean_rr > 0:
        prev_rr_norm = prev_rr / local_mean_rr   # <1 => дойде РАНО (типично за S)
        next_rr_norm = next_rr / local_mean_rr
    else:
        prev_rr_norm = 1.0
        next_rr_norm = 1.0

    # --- P-вълна признаци (R-пикът е на позиция win_before в сегмента) ---
    p_feats = _p_wave_features(morphology, fs, win_before)

    return np.concatenate([
        morphology,
        [prev_rr, next_rr, rr_ratio, prev_rr_norm, next_rr_norm],
        p_feats
    ])


# ---------------------------------------------------------------------------
# 3. Обработка на списък от записи -> X, y
# ---------------------------------------------------------------------------
def process_records(record_list, data_dir, lead=0):
    X, y = [], []
    for rec in record_list:
        rec_path = os.path.join(data_dir, rec)
        try:
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, 'atr')
        except Exception as e:
            print(f"  [!] Пропуснат запис {rec}: {e}")
            continue

        signal = record.p_signal[:, lead]
        fs = record.fs
        samples = annotation.sample
        symbols = annotation.symbol

        # Филтрираме само истинските удари с валиден AAMI клас
        beats = [(s, sym) for s, sym in zip(samples, symbols)
                 if sym in AAMI_GROUPS]

        # Предварително смятаме всички RR интервали (в секунди)
        peaks = [b[0] for b in beats]
        rr_all = [(peaks[i] - peaks[i - 1]) / fs for i in range(1, len(peaks))]

        WIN = 10  # брой предходни удари за локалната средна
        for i, (r_peak, sym) in enumerate(beats):
            prev_r = beats[i - 1][0] if i > 0 else None
            next_r = beats[i + 1][0] if i < len(beats) - 1 else None

            # Локална средна на RR от последните WIN удара (без текущия)
            lo = max(0, i - WIN)
            window_rr = rr_all[lo:i] if i > 0 else []
            local_mean_rr = float(np.mean(window_rr)) if window_rr else None

            feats = extract_beat_features(signal, r_peak, prev_r, next_r, fs,
                                          local_mean_rr=local_mean_rr)
            if feats is None:
                continue

            X.append(feats)
            y.append(AAMI_GROUPS[sym])

    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# 4. Основна тренировка
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='Папка с MIT-BIH .dat/.hea/.atr файлове')
    ap.add_argument('--out_dir', default='artifacts', help='Къде да се запазят модел/scaler/encoder')
    ap.add_argument('--lead', type=int, default=0, help='Кой лид да се ползва (0=MLII обикновено)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Пълен списък записи от MIT-BIH (без 102/104/107/217 - силно пейсирани, по избор)
    record_numbers = ['100','101','103','105','106','108','109','111','112','113',
                      '114','115','116','117','118','119','121','122','123','124',
                      '200','201','202','203','205','207','208','209','210','212',
                      '213','214','215','219','220','221','222','223','228','230',
                      '231','232','233','234']

    # Оставяме само реално наличните записи в папката
    record_numbers = [r for r in record_numbers
                      if os.path.exists(os.path.join(args.data_dir, r + '.dat'))]
    if not record_numbers:
        raise FileNotFoundError(f"Няма намерени записи в {args.data_dir}")

    # ВАЖНО: разделяме по ЗАПИСИ, за да няма изтичане на данни между пациенти.
    # (Не е коректно удари от един пациент да са и в train, и в test.)
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

    # -- Скалиране: fit САМО на train, после се запазва --
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)   # само transform!

    # -- SMOTE само на train (за баланс на редките класове) --
    # k_neighbors се сваля ако някой клас е много малък
    min_count = min(Counter(y_train_enc).values())
    k = min(5, max(1, min_count - 1))
    try:
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X_train_s, y_train_enc)
        print("След SMOTE:", Counter(y_res))
    except ValueError as e:
        print(f"  [!] SMOTE пропуснат ({e}); тренираме на оригинала")
        X_res, y_res = X_train_s, y_train_enc

    # -- Модел: Random Forest (устойчив, multi-class естествено) --
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    print("Трениране на Random Forest...")
    clf.fit(X_res, y_res)

    # -- Оценка --
    y_pred = clf.predict(X_test_s)
    print("\n===== Classification Report =====")
    print(classification_report(y_test_enc, y_pred,
                                target_names=le.classes_, zero_division=0))
    print("===== Confusion Matrix =====")
    print("Редове=истина, колони=предсказано; ред/колона =", list(le.classes_))
    print(confusion_matrix(y_test_enc, y_pred))

    # -- Запазване на ВСИЧКО нужно за deployment --
    joblib.dump(clf, os.path.join(args.out_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(args.out_dir, 'label_encoder.pkl'))
    print(f"\nЗапазени: model.pkl, scaler.pkl, label_encoder.pkl в {args.out_dir}/")


if __name__ == '__main__':
    main()