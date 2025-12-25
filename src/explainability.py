import shap

def shap_visualization_all_classes(model, X_df):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
    except Exception as e:
        print(f"SHAP TreeExplainer tidak bisa digunakan: {e}")
        return

    if isinstance(shap_values, list):
        for i, sv in enumerate(shap_values):
            print(f"\n=== SHAP Summary Plot: Class {i} ===")
            shap.summary_plot(sv, X_df, plot_type="bar", max_display=10)
            shap.dependence_plot("Glucose", sv, X_df)
            shap.dependence_plot("BMI", sv, X_df)

    elif len(shap_values.shape) == 3:
        n_classes = shap_values.shape[2]
        for i in range(n_classes):
            sv = shap_values[:, :, i]
            print(f"\n=== SHAP Class {i} ===")
            shap.summary_plot(sv, X_df, plot_type="bar", max_display=10)
            shap.dependence_plot("Glucose", sv, X_df)
            shap.dependence_plot("BMI", sv, X_df)

    else:
        shap.summary_plot(shap_values, X_df, plot_type="bar", max_display=10)
        shap.dependence_plot("Glucose", shap_values, X_df)
        shap.dependence_plot("BMI", shap_values, X_df)
