import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.joblib"
CONFIG_PATH = APP_DIR / "config.json"
TEST_PATH = APP_DIR / "test.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


st.set_page_config(page_title="Винная подборка", layout="wide")
st.title("Винная подборка")
st.caption("Логистическая регрессия оценивает шанс получить экспертную оценку 7 или выше.")

if not all(path.is_file() for path in (MODEL_PATH, CONFIG_PATH, TEST_PATH)):
    st.error("Нет файлов модели, настроек или тестовой выборки. Выполните ячейку сохранения в ноутбуке.")
    st.stop()

model = load_model()
features = list(model.feature_names_in_)
test = load_test()
X_test = test[features]
y_test = test["target"]
proba = test_proba(model, X_test)
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

threshold = st.sidebar.slider(
    "Порог рекомендации", 0.0, 1.0, float(config["threshold"]), step=0.001, format="%.4f"
)

one_tab, metrics_tab, batch_tab, weights_tab = st.tabs([
    "Одно вино", "Модельные метрики", "CSV с винами", "Коэффициенты"
])

with one_tab:
    labels = {
        "fixed_acidity": "Фиксированная кислотность",
        "volatile_acidity": "Летучая кислотность",
        "citric_acid": "Лимонная кислота",
        "residual_sugar": "Остаточный сахар",
        "chlorides": "Хлориды",
        "free_sulfur_dioxide": "Свободный диоксид серы",
        "total_sulfur_dioxide": "Общий диоксид серы",
        "density": "Плотность",
        "pH": "pH",
        "sulphates": "Сульфаты",
        "alcohol": "Алкоголь",
    }
    columns = st.columns(2)
    wine = {}
    for i, (feature, label) in enumerate(labels.items()):
        values = X_test[feature]
        wine[feature] = columns[i % 2].number_input(
            label,
            min_value=float(values.min()),
            max_value=float(values.max()),
            value=float(values.median()),
            step=0.00001 if feature == "density" else 0.001,
            format="%.5f" if feature == "density" else "%.3f",
        )
    color = columns[1].radio("Цвет вина", ["Белое", "Красное"], horizontal=True)
    wine["is_red"] = int(color == "Красное")

    wine_frame = pd.DataFrame([wine], columns=features)
    wine_proba = float(model.predict_proba(wine_frame)[0, 1])
    st.metric("Вероятность хорошего вина", f"{wine_proba:.1%}")
    st.progress(wine_proba)
    if wine_proba >= threshold:
        st.success("Включить в подборку")
    else:
        st.info("Пропустить при этом пороге")

with metrics_tab:
    pred = proba >= threshold
    matrix = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)

    first, second, third = st.columns(3)
    first.metric("Precision", f"{precision:.1%}")
    second.metric("Recall", f"{recall:.1%}")
    third.metric("F1", f"{f1:.3f}")
    st.write(
        "При более высоком пороге в подборку попадает меньше вин и можно пропустить "
        "часть хороших. Более низкий порог помогает найти больше хороших вин, "
        "но вместе с ними в подборку могут попасть и посредственные."
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    metrics.ConfusionMatrixDisplay(matrix, display_labels=[0, 1]).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set(title="Матрица ошибок", xlabel="Прогноз", ylabel="Факт")

    metrics.RocCurveDisplay.from_predictions(y_test, proba, name="Модель", ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1].scatter(fp / (fp + tn), recall, color="tab:red", label="Текущий порог", zorder=3)
    axes[1].set(title="ROC", xlabel="Доля ложных рекомендаций", ylabel="Recall")
    axes[1].legend(fontsize="small", loc="lower right")

    metrics.PrecisionRecallDisplay.from_predictions(y_test, proba, name="Модель", ax=axes[2])
    axes[2].axhline(y_test.mean(), linestyle="--", color="gray", linewidth=1)
    axes[2].scatter(recall, precision, color="tab:red", label="Текущий порог", zorder=3)
    axes[2].set(title="Precision / Recall", xlabel="Recall", ylabel="Precision")
    axes[2].legend(fontsize="small")
    st.pyplot(fig)
    plt.close(fig)

with batch_tab:
    uploaded = st.file_uploader("CSV с признаками вин", type="csv")
    if uploaded is None:
        st.info("Загрузите CSV со столбцами, на которых обучалась модель.")
        st.caption(", ".join(features))
    else:
        try:
            batch = pd.read_csv(uploaded)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError):
            st.error("Не удалось прочитать CSV. Проверьте формат и кодировку файла.")
        else:
            missing = [column for column in features if column not in batch.columns]
            if missing:
                st.error(f"В файле нет столбцов: {', '.join(missing)}")
            elif batch.empty:
                st.error("В CSV нет строк с данными.")
            else:
                batch_features = batch[features].apply(pd.to_numeric, errors="coerce")
                if not np.isfinite(batch_features.to_numpy()).all():
                    st.error("Все признаки должны быть числами без пропусков.")
                else:
                    batch["probability"] = model.predict_proba(batch_features)[:, 1]
                    batch["prediction"] = (batch["probability"] >= threshold).astype(int)
                    st.metric("Вин в подборке", int(batch["prediction"].sum()))
                    st.dataframe(batch, hide_index=True)
                    st.download_button(
                        "Скачать прогнозы",
                        data=batch.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                        on_click="ignore",
                    )

with weights_tab:
    weights = pd.Series(model[-1].coef_[0], index=features).sort_values()
    colors = ["tab:blue" if value >= 0 else "tab:orange" for value in weights]
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.barh(weights.index, weights, color=colors)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Коэффициент на шкале после стандартизации")
    st.pyplot(fig, width=700)
    plt.close(fig)

    positive = ", ".join(weights[weights > 0].tail(2).index[::-1])
    negative = ", ".join(weights[weights < 0].head(2).index)
    st.write(f"Сильнее всего за хорошее вино голосуют {positive}, против голосуют {negative}.")
