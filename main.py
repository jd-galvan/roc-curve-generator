import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def leer_scores(archivo_clientes, archivo_impostores):
    clientes_data = []
    with open(archivo_clientes.name, 'r') as f:
        for line in f:
            splitted = line.strip().split()
            if len(splitted) == 2:
                score = float(splitted[1])
                clientes_data.append([1, score])  # (y_true=1, y_score=score)

    impostores_data = []
    with open(archivo_impostores.name, 'r') as f:
        for line in f:
            splitted = line.strip().split()
            if len(splitted) == 2:
                score = float(splitted[1])
                impostores_data.append([0, score])  # (y_true=0, y_score=score)

    df = pd.DataFrame(clientes_data + impostores_data,
                      columns=["y_true", "y_score"])
    return df


def compute_metrics(df, desired_fn, desired_fp):
    # Separar positivos y negativos
    df_pos = df[df["y_true"] == 1]["y_score"]
    df_neg = df[df["y_true"] == 0]["y_score"]

    # Curva ROC y AUC
    fpr, tpr, thresholds = roc_curve(df["y_true"], df["y_score"])
    auc_val = auc(fpr, tpr)

    # 1) FP(FN = X) y umbral
    fnr = (1 - tpr)
    idx_fnr = np.argmin(np.abs(fnr - desired_fn))
    fpr_at_desired_fnr = fpr[idx_fnr]
    threshold_fpr_at_desired_fnr = thresholds[idx_fnr]

    # 2) FN(FP=X) y umbral
    idx_fpr = np.argmin(np.abs(fpr - desired_fp))
    fnr_at_desired_fpr = fnr[idx_fpr]
    threshold_fnr_at_desired_fpr = thresholds[idx_fpr]

    # 3) FP = FN y umbral
    idx_eq = np.argmin(np.abs(fpr-fnr))
    equal_fpr = fpr[idx_eq]
    equal_fnr = fnr[idx_eq]
    threshold_eq = thresholds[idx_eq]

    # 4) Cálculo de d' (d-prime) según la fórmula:
    #    d' = (mu_pos - mu_neg) / sqrt( sigma_pos^2 + sigma_neg^2 )
    mu_pos = np.mean(df_pos)
    mu_neg = np.mean(df_neg)
    sigma_pos = np.std(df_pos)
    sigma_neg = np.std(df_neg)

    # Evitar división por cero
    if sigma_pos == 0 and sigma_neg == 0:
        d_prime = 0
    else:
        d_prime = (mu_pos - mu_neg) / np.sqrt((sigma_pos**2) + (sigma_neg**2))

    metrics = {
        "fpr_with_desired_fnr": {
            "fpr": fpr_at_desired_fnr,
            "threshold": threshold_fpr_at_desired_fnr,
        },
        "fnr_with_desired_fpr": {
            "fnr": fnr_at_desired_fpr,
            "threshold": threshold_fnr_at_desired_fpr
        },
        "equal_fpr_fnr": {
            "threshold": threshold_eq,
            "fpr": equal_fpr,
            "fnr": equal_fnr
        },
        "AUC": auc_val,
        "d_prime": d_prime
    }

    return metrics, fpr, tpr


def plot_roc(clientes1_file, impostores1_file, clientes2_file, impostores2_file, desired_fn, desired_fp):
    # Dataset A
    df_A = leer_scores(clientes1_file, impostores1_file)
    metrics_A, fpr_A, tpr_A = compute_metrics(df_A, desired_fn, desired_fp)

    # Dataset B
    df_B = leer_scores(clientes2_file, impostores2_file)
    metrics_B, fpr_B, tpr_B = compute_metrics(df_B, desired_fn, desired_fp)

    # Graficar ambas curvas ROC
    plt.figure()
    plt.plot(fpr_A, tpr_A, label=f"Dataset A")
    plt.plot(fpr_B, tpr_B, label=f"Dataset B")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title("Comparación de Curvas ROC")
    plt.legend(loc="lower right")

    # Resumen de métricas en Markdown
    text = "## Resultados Dataset A:\n"
    text += f"- **FP(FN={desired_fn})** = {metrics_A['fpr_with_desired_fnr']['fpr']:.3f} → Umbral: {metrics_A['fpr_with_desired_fnr']['threshold']:.3f}\n"
    text += f"- **FN(FP={desired_fp})** = {metrics_A['fnr_with_desired_fpr']['fnr']:.3f} → Umbral: {metrics_A['fnr_with_desired_fpr']['threshold']:.3f}\n"
    text += f"- **FP = FN** → Umbral: {metrics_A['equal_fpr_fnr']['threshold']:.3f}, FP: {metrics_A['equal_fpr_fnr']['fpr']:.1f}, FN: {metrics_A['equal_fpr_fnr']['fnr']:.1f}\n"
    text += f"- **Área bajo curva** = {metrics_A['AUC']:.3f}\n"
    text += f"- **d'** = {metrics_A['d_prime']:.3f}\n\n"

    text += "## Resultados Dataset B:\n"
    text += f"- **FP(FN={desired_fn})** = {metrics_B['fpr_with_desired_fnr']['fpr']:.3f} → Umbral: {metrics_B['fpr_with_desired_fnr']['threshold']:.3f}\n"
    text += f"- **FN(FP={desired_fp})** = {metrics_B['fnr_with_desired_fpr']['fnr']:.3f} → Umbral: {metrics_B['fnr_with_desired_fpr']['threshold']:.3f}\n"
    text += f"- **FP = FN** → Umbral: {metrics_B['equal_fpr_fnr']['threshold']:.3f}, FP: {metrics_B['equal_fpr_fnr']['fpr']:.1f}, FN: {metrics_B['equal_fpr_fnr']['fnr']:.1f}\n"
    text += f"- **Área bajo curva** = {metrics_B['AUC']:.3f}\n"
    text += f"- **d'** = {metrics_B['d_prime']:.3f}\n\n"

    return plt, text


# Interfaz Gradio
demo = gr.Interface(
    fn=plot_roc,
    inputs=[
        gr.File(label="Archivo de Clientes A"),
        gr.File(label="Archivo de Impostores A"),
        gr.File(label="Archivo de Clientes B"),
        gr.File(label="Archivo de Impostores B"),
        gr.Number(label="FN deseado", value=0.5),
        gr.Number(label="FP deseado", value=0.5)
    ],
    outputs=["plot", "markdown"],
    title="Curvas ROC",
    description=(
        "<strong>Asignatura</strong>: Biometría<br>"
        "<strong>Estudiante</strong>: José Daniel Galván<br>"
    ),
    allow_flagging="never"
)

demo.launch(debug=True)
