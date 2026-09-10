"""
1D CNN (Convolutional Neural Network) за ЕКГ класификация
==========================================================
Третият модел. За разлика от RF и MLP, тук конволюционната мрежа учи
морфологията на удара САМА от суровия сигнал, вместо да разчита само на
ръчно извлечени признаци.

Хибридна архитектура (две входни глави):
  1. Морфология: суровият сегмент на удара -> Conv1D блокове (учат формата)
  2. RR/P контекст: последните 9 инженерни признака -> директно в dense частта
     (за да запазим тайминг информацията, която много помогна на клас S)

Двата клона се сливат, после dense + softmax за 5-те AAMI класа.

Ползва СЪЩАТА логика за извличане като другите два модела (импортира от
train_arrhythmia), така че сравнението е чисто.

Изисква: tensorflow
Изпълнение:
    python train_cnn.py --data_dir /път/до/mit-bih --out_dir artifacts_cnn
"""

import os
import argparse
import numpy as np
import joblib
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from train_arrhythmia import process_records

# Броят инженерни признаци накрая на всеки вектор (prev_rr, next_rr, rr_ratio,
# prev_rr_norm, next_rr_norm, + 4 P-вълна признака) = 9.
N_CONTEXT = 9


def build_model(morph_len, n_context, n_classes):
    """Хибриден 1D CNN: Conv клон за морфология + dense клон за RR/P контекст."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    # --- Клон 1: морфология през конволюции ---
    morph_in = layers.Input(shape=(morph_len, 1), name='morphology')
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(morph_in)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # --- Клон 2: RR/P контекст директно ---
    ctx_in = layers.Input(shape=(n_context,), name='context')
    c = layers.Dense(16, activation='relu')(ctx_in)

    # --- Сливане ---
    merged = layers.concatenate([x, c])
    z = layers.Dense(64, activation='relu')(merged)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(n_classes, activation='softmax')(z)

    model = Model(inputs=[morph_in, ctx_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def split_features(X):
    """Разделя всеки вектор на (морфология, контекст)."""
    morph = X[:, :-N_CONTEXT]
    ctx = X[:, -N_CONTEXT:]
    return morph, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', default='artifacts_cnn')
    ap.add_argument('--lead', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=256)
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

    # -- Скалиране (fit само на train, после се запазва) --
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # -- Разделяне на морфология и контекст --
    Xtr_morph, Xtr_ctx = split_features(X_train_s)
    Xte_morph, Xte_ctx = split_features(X_test_s)
    morph_len = Xtr_morph.shape[1]

    # Reshape морфологията за Conv1D: (проби, дължина, 1 канал)
    Xtr_morph = Xtr_morph[..., np.newaxis]
    Xte_morph = Xte_morph[..., np.newaxis]

    # -- Балансиране чрез class weights (естествено за невронни мрежи) --
    classes = np.unique(y_train_enc)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_enc)
    # ОГРАНИЧАВАМЕ теглата: класове с шепа примери (Q=15) иначе получават
    # тегло ~1000, което дърпа обучението в грешна посока и дестабилизира
    # останалите класове. Таван от 50 е разумен компромис.
    weights = np.clip(weights, None, 50.0)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights (ограничени):", class_weight)

    # -- Модел --
    import tensorflow as tf
    model = build_model(morph_len, N_CONTEXT, len(le.classes_))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True,
                                         monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5,
                                             monitor='val_loss', min_lr=1e-5),
    ]

    print("Трениране на CNN...")
    model.fit(
        {'morphology': Xtr_morph, 'context': Xtr_ctx},
        y_train_enc,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # -- Оценка --
    probs = model.predict({'morphology': Xte_morph, 'context': Xte_ctx})
    y_pred = np.argmax(probs, axis=1)

    print("\n===== Classification Report (CNN) =====")
    print(classification_report(y_test_enc, y_pred,
                                target_names=le.classes_, zero_division=0))
    print("===== Confusion Matrix =====")
    print("Редове=истина, колони=предсказано; ред/колона =", list(le.classes_))
    print(confusion_matrix(y_test_enc, y_pred))

    # -- Запазване --
    model.save(os.path.join(args.out_dir, 'model.keras'))
    joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(args.out_dir, 'label_encoder.pkl'))
    # Записваме и колко признака са контекст (нужно за deployment)
    joblib.dump({'n_context': N_CONTEXT, 'morph_len': morph_len},
                os.path.join(args.out_dir, 'meta.pkl'))
    print(f"\nЗапазени: model.keras, scaler.pkl, label_encoder.pkl, meta.pkl в {args.out_dir}/")


if __name__ == '__main__':
    main()