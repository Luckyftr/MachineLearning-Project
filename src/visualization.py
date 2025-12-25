def visualize_decision_tree(model, feature_names):
    plt.figure(figsize=(20,10))
    plot_tree(
        model, feature_names=feature_names,
        class_names=["No Diabetes","Diabetes"],
        filled=True, rounded=True, fontsize=10
    )
    plt.title("Decision Tree Structure")
    plt.show()