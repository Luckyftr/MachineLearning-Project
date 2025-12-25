import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(df):

    print("\n=== Statistik Deskriptif ===")
    display(df.describe())

    print("\n=== Distribusi Target ===")
    plt.figure(figsize=(5,4))
    sns.countplot(data=df, x="Outcome")
    plt.title("Distribusi Target Diabetes")
    plt.show()
    print(df["Outcome"].value_counts(normalize=True))

    print("\n=== Heatmap Korelasi Fitur ===")
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap Korelasi Fitur")
    plt.show()

    print("\n=== Boxplot (Outlier Detection) ===")
    plt.figure(figsize=(12,6))
    df.boxplot()
    plt.title("Boxplot Semua Fitur")
    plt.xticks(rotation=45)
    plt.show()

    # Cek missing value
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]  

    if not missing_counts.empty:
        print("\n=== Missing Value Boxplot ===")
        plt.figure(figsize=(8,4))
        sns.boxplot(x=missing_counts.values)
        plt.title("Distribusi Jumlah Missing Value per Kolom")
        plt.xlabel("Jumlah Missing Value")
        plt.show()
    else:
        print("\nTidak ada missing value di dataset.")
