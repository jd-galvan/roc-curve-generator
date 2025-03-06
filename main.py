import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def leer_scores(archivo_clientes, archivo_impostores):
    """
    Lee los archivos de clientes e impostores (sin encabezados) y retorna
    un DataFrame con columnas: y_true y y_score.
    """
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
    """
    Calcula:
      - Curva ROC (fpr, tpr) y AUC
      - Umbral donde FN se acerca a 'desired_fn' (y su FP)
      - Umbral donde FP se acerca a 'desired_fp' (y su FN)
      - Umbral donde FP = FN (lo más cercano posible)
      - d' usando la diferencia de medias y varianzas de clientes/impostores
    """
    # Separar positivos y negativos
    df_pos = df[df["y_true"] == 1]["y_score"]
    df_neg = df[df["y_true"] == 0]["y_score"]

    # Curva ROC y AUC
    fpr, tpr, thresholds = roc_curve(df["y_true"], df["y_score"])
    auc_val = auc(fpr, tpr)

    # Cálculo de FP y FN en valores absolutos
    n_pos = len(df_pos)  # número de clientes
    n_neg = len(df_neg)  # número de impostores
    FP_counts = fpr * n_neg
    FN_counts = (1 - tpr) * n_pos

    # 1) FN = desired_fn (buscar umbral)
    idx_fn = np.argmin(np.abs(FN_counts - desired_fn))
    threshold_fn = thresholds[idx_fn]
    fp_at_desired_fn = FP_counts[idx_fn]
    fn_at_desired_fn = FN_counts[idx_fn]

    # 2) FP = desired_fp (buscar umbral)
    idx_fp = np.argmin(np.abs(FP_counts - desired_fp))
    threshold_fp = thresholds[idx_fp]
    fp_at_desired_fp = FP_counts[idx_fp]
    fn_at_desired_fp = FN_counts[idx_fp]

    # 3) FP = FN (buscar umbral donde la diferencia sea mínima)
    idx_eq = np.argmin(np.abs(FP_counts - FN_counts))
    threshold_eq = thresholds[idx_eq]
    fp_eq = FP_counts[idx_eq]
    fn_eq = FN_counts[idx_eq]

    # 4) Cálculo de d' (d-prime) según la fórmula:
    #    d' = (mu_pos - mu_neg) / sqrt( sigma_pos^2 + sigma_neg^2 )
    mu_pos = df_pos.mean()
    mu_neg = df_neg.mean()
    sigma_pos = df_pos.std(ddof=1)  # ddof=1 para varianza muestral
    sigma_neg = df_neg.std(ddof=1)

    # Evitar división por cero si hay varianza nula
    if sigma_pos == 0 and sigma_neg == 0:
        d_prime = 0
    else:
        d_prime = (mu_pos - mu_neg) / np.sqrt((sigma_pos**2) + (sigma_neg**2))

    metrics = {
        "desired_fn": {
            "threshold": threshold_fn,
            "FP": fp_at_desired_fn,
            "FN": fn_at_desired_fn
        },
        "desired_fp": {
            "threshold": threshold_fp,
            "FP": fp_at_desired_fp,
            "FN": fn_at_desired_fp
        },
        "equal_FP_FN": {
            "threshold": threshold_eq,
            "FP": fp_eq,
            "FN": fn_eq
        },
        "AUC": auc_val,
        "D_prime": d_prime
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
    text += f"- **FN deseado** = {desired_fn} → Umbral: {metrics_A['desired_fn']['threshold']:.3f}, FP: {metrics_A['desired_fn']['FP']:.1f}, FN: {metrics_A['desired_fn']['FN']:.1f}\n"
    text += f"- **FP deseado** = {desired_fp} → Umbral: {metrics_A['desired_fp']['threshold']:.3f}, FP: {metrics_A['desired_fp']['FP']:.1f}, FN: {metrics_A['desired_fp']['FN']:.1f}\n"
    text += f"- **FP = FN** → Umbral: {metrics_A['equal_FP_FN']['threshold']:.3f}, FP: {metrics_A['equal_FP_FN']['FP']:.1f}, FN: {metrics_A['equal_FP_FN']['FN']:.1f}\n"
    text += f"- **Área bajo curva** = {metrics_A['AUC']:.3f}\n"
    text += f"- **d'** = {metrics_A['D_prime']:.3f}\n\n"

    text += "## Resultados Dataset B:\n"
    text += f"- **FN deseado** = {desired_fn} → Umbral: {metrics_B['desired_fn']['threshold']:.3f}, FP: {metrics_B['desired_fn']['FP']:.1f}, FN: {metrics_B['desired_fn']['FN']:.1f}\n"
    text += f"- **FP deseado** = {desired_fp} → Umbral: {metrics_B['desired_fp']['threshold']:.3f}, FP: {metrics_B['desired_fp']['FP']:.1f}, FN: {metrics_B['desired_fp']['FN']:.1f}\n"
    text += f"- **FP = FN** → Umbral: {metrics_B['equal_FP_FN']['threshold']:.3f}, FP: {metrics_B['equal_FP_FN']['FP']:.1f}, FN: {metrics_B['equal_FP_FN']['FN']:.1f}\n"
    text += f"- **Área bajo curva** = {metrics_B['AUC']:.3f}\n"
    text += f"- **d'** = {metrics_B['D_prime']:.3f}\n"

    return plt, text


# Interfaz Gradio
demo = gr.Interface(
    fn=plot_roc,
    inputs=[
        gr.File(label="Archivo de Clientes A"),
        gr.File(label="Archivo de Impostores A"),
        gr.File(label="Archivo de Clientes B"),
        gr.File(label="Archivo de Impostores B"),
        gr.Number(label="FN deseado", value=10),
        gr.Number(label="FP deseado", value=10)
    ],
    outputs=["plot", "markdown"],
    title="Curvas ROC",
    description=(
        "<strong>Asignatura</strong>: Biometría<br>"
        "<strong>Estudiante</strong>: José Daniel Galván<br>"
    ),
    allow_flagging="never"
)

demo.launch()
