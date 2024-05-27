import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans

DATA = "./data/heart_failure_clinical_records.csv"

def loadCSV(path: str):
    df = pd.read_csv(path)
    showPlots(df)
    return df

def showPlots(df: pd.DataFrame):
    st.markdown("# Heart failure")
    st.markdown("## Datasets")

    # Og chart with no deaths
    df_no_death = df.copy(deep=True)
    df_no_death = df_no_death.drop(index=df_no_death[df_no_death["DEATH_EVENT"] == 0].index)
    df_no_death = df_no_death.drop("DEATH_EVENT", axis=1)
    df_no_death.sort_values(by="creatinine_phosphokinase")
    st.markdown("### No deaths")
    st.dataframe(df_no_death)
    st.markdown("#### Median")
    no_death_median = df_no_death.median()

    # No death median table creatinine
    st.table(no_death_median)

    # Chart with only deaths
    df_death = df.copy(deep=True)
    df_death = df_death.drop(index=df_death[df_death["DEATH_EVENT"] == 1].index)
    df_death = df_death.drop("DEATH_EVENT", axis=1)
    df_death.sort_values(by="creatinine_phosphokinase")
    st.markdown("### Only deaths")
    st.dataframe(df_death)
    st.markdown("#### Median")
    death_median = df_death.median()

    # Death median table creatinine
    st.table(death_median)

    st.markdown("### Corelation table")
    corr = df.corr()
    st.dataframe(corr)

    # Corelation heatmap
    sb.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size": 8},  cmap='coolwarm')
    st.pyplot(plt)
    plt.close()

    # Pairplot
    relevant_columns = ['ejection_fraction', 'serum_sodium', 'serum_creatinine', 'time', 'creatinine_phosphokinase', 'DEATH_EVENT']
    data_relevant = df[relevant_columns]
    plt.figure(figsize=(10, 6))
    sb.pairplot(data_relevant, hue="DEATH_EVENT")
    st.pyplot(plt)
    plt.close()

    # Histograms of each column
    for column in df.columns:
        if column == 'DEATH_EVENT':
            continue
        plt.figure(figsize=(10, 4))
        plt.title(f'Distribution of {column}')
        if column in ['sex', 'anaemia', 'diabetes', 'high_blood_pressure', 'smoking']:
            sb.countplot(data=df, x=column)
            st.pyplot(plt)
            plt.close()
            continue
        sb.histplot(df[column], kde=True, bins=30)
        st.pyplot(plt)
        plt.close()
    
    # Clustering data
    ## Elbow method
    wcss = []
    # data_scaled = StandardScaler().fit_transform(data_relevant.drop("DEATH_EVENT", axis=1))
    data_scaled = StandardScaler().fit_transform(data_relevant)
    for k in range(1, 21):
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data_scaled)
        
        wcss.append(kmeanModel.inertia_)

    st.markdown("# Clustering")
    st.markdown("## Metoda łokci")
    plt.figure(figsize=(10, 6))

    # Najlepszy wynik daje 5 klastrów
    plt.plot(range(1, 21), wcss, marker='o', linestyle='-')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('Metoda łokci')
    st.pyplot(plt)
    plt.close()

    st.markdown("## Scatters")
    kmeanModel = KMeans(n_clusters=5)
    cdf = df.copy(deep=True)
    cdf['cluster'] = kmeanModel.fit_predict(data_scaled)
    st.write("Cluster mean:")
    st.table(cdf.groupby("cluster", sort=True).mean())
    
    plt.figure(figsize=(10, 6))
    colors = mpl.colormaps['Paired'].resampled(5)
    for cluster_id in range(5):
        cluster_data = df[cdf['cluster'] == cluster_id]
        plt.scatter(cluster_data['ejection_fraction'], cluster_data['serum_sodium'], color=colors(cluster_id/5), label=f'Cluster {cluster_id}', s=50)

    # 'ejection_fraction', 'serum_sodium', 'serum_creatinine', 'time', 'creatinine_phosphokinase', 'DEATH_EVENT'

    plt.xlabel(relevant_columns[0])
    plt.ylabel(relevant_columns[1])
    plt.title('Klasteryzacja pacjentów na podstawie frakcji wyrzutowej i sodu w soczewicy')
    plt.legend(title="Clusters")
    st.pyplot(plt)
    plt.close()


    plt.figure(figsize=(10, 6))
    colors = mpl.colormaps['Paired'].resampled(5)
    for cluster_id in range(5):
        cluster_data = df[cdf['cluster'] == cluster_id]
        plt.scatter(cluster_data['serum_sodium'], cluster_data['serum_creatinine'], color=colors(cluster_id/5), label=f'Cluster {cluster_id}', s=50)

    # 'ejection_fraction', 'serum_sodium', 'serum_creatinine', 'time', 'creatinine_phosphokinase', 'DEATH_EVENT'

    plt.xlabel(relevant_columns[1])
    plt.ylabel(relevant_columns[2])
    plt.title('Klasteryzacja pacjentów na podstawie sodu i kreatyniny w soczewicy')
    plt.legend(title="Clusters")
    st.pyplot(plt)
    plt.close()

    plt.figure(figsize=(10, 6))
    colors = mpl.colormaps['Paired'].resampled(5)
    for cluster_id in range(5):
        cluster_data = df[cdf['cluster'] == cluster_id]
        plt.scatter(cluster_data['ejection_fraction'], cluster_data['creatinine_phosphokinase'], color=colors(cluster_id/5), label=f'Cluster {cluster_id}', s=50)

    # 'ejection_fraction', 'serum_sodium', 'serum_creatinine', 'time', 'creatinine_phosphokinase', 'DEATH_EVENT'

    plt.xlabel(relevant_columns[0])
    plt.ylabel(relevant_columns[4])
    plt.title('Klasteryzacja pacjentów na podstawie frakcji wyrzutu i fosfokinazy kreatyniny')
    plt.legend(title="Clusters")
    st.pyplot(plt)
    plt.close()
    
def create_model(df, model, name):
    x = df.drop("DEATH_EVENT", axis=1).drop("time", axis=1)
    y = df["DEATH_EVENT"]

    # Data scaling from 0 to 1
    min_max_scaler = MinMaxScaler()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_train = min_max_scaler.fit_transform(x_train)
    # x_train = standard_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)
    # x_test = standard_scaler.fit_transform(x_test)

    model.fit(x_train, y_train)

    # Predict
    y_pred = model.predict(x_test)
    st.markdown(f'## {name}')
    showMetrics(y_test, y_pred)

def showMetrics(y_test, y_pred):
    # Metrics
    ## Accuracy
    st.write("Accuracy:", accuracy_score(y_test, y_pred))

    ## Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.header('Confusion Matrix')
    fig, ax = plt.subplots()
    sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    plt.close()

    ## Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    st.header('Classification Report')
    class_report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(class_report_df)

if __name__ == "__main__":
    df = loadCSV(DATA)

    st.markdown("# Prediction Models")

    # Models
    logistic_reg = LogisticRegression()
    create_model(df.copy(deep=True), logistic_reg, "Logistic Regression")

    random_forest = RandomForestClassifier()
    create_model(df.copy(deep=True), random_forest, "Random Forest Classifier")

    gradient = GradientBoostingClassifier()
    create_model(df.copy(deep=True), gradient, "Gradient Boosting")

    svc = LinearSVC(dual='auto')
    create_model(df.copy(deep=True), svc, "Linear Support Vector Machines (SVC)")

    k_neighbors = KNeighborsClassifier()
    create_model(df.copy(deep=True), k_neighbors, "K-Nearest Neighbors (KNN)")
