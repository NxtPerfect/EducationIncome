import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sb
from time import perf_counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
    no_death_creatinine = no_death_median["creatinine_phosphokinase"]
    no_death_time = no_death_median["time"]
    st.bar_chart(df_no_death[["creatinine_phosphokinase"]])

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

    death_creatinine = death_median["creatinine_phosphokinase"]
    death_time = death_median["time"]
    st.bar_chart(df_death[["creatinine_phosphokinase"]])

    st.markdown("### Corelation?")
    corr = df.corr()
    st.dataframe(corr)
    st.markdown("*Ejection fraction* and *serum sodium* are correlated the most?")
    st.markdown(f'Ejection fraction {no_death_median["ejection_fraction"]}-{death_median["ejection_fraction"]}={((no_death_median["ejection_fraction"] - death_median["ejection_fraction"])/death_median["ejection_fraction"])*100.0:.2f}%')
    st.markdown(f'Serum creatinine {no_death_median["serum_creatinine"]}-{death_median["serum_creatinine"]}={((no_death_median["serum_creatinine"] - death_median["serum_creatinine"])/death_median["serum_creatinine"])*100.0:.2f}%')

    st.markdown("## Results")
    st.markdown(f'Percentage change of creatinine phosphokinase for alive vs dead patients is {no_death_creatinine} - {death_creatinine}=**{((no_death_creatinine - death_creatinine) / no_death_creatinine) * 100.0:.2f}%**. This means if your creatinine phosphokinase is around 230mcg/L, you might be at risk of heart failure. The time between follow up appointment for patients who died is {death_time} - {no_death_time}=**{((death_time - no_death_time) / death_time) * 100.0:.2f}%** higher, during which they were found dead.')

    # Corelation heatmap
    sb.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size": 8},  cmap='coolwarm')
    st.pyplot(plt)

    # Pairplot
    relevant_columns = ['ejection_fraction', 'serum_sodium', 'serum_creatinine', 'time', 'creatinine_phosphokinase', 'DEATH_EVENT']
    data_relevant = df[relevant_columns]
    sb.pairplot(data_relevant, hue="DEATH_EVENT")
    st.pyplot(plt)

    # Histograms of each column
    for column in df.columns:
        if column != 'DEATH_EVENT':
            plt.figure(figsize=(10, 4))
            sb.histplot(df[column], kde=True, bins=30)
            plt.title(f'Distribution of {column}')
            st.pyplot(plt)
    
def create_model(df, model, name):
    x = df.drop("DEATH_EVENT", axis=1)
    y = df["DEATH_EVENT"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model.fit(x_train, y_train)

    # Predict
    y_pred = model.predict(x_test)
    st.markdown(f'# {name}')
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

    ## Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    st.header('Classification Report')
    class_report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(class_report_df)

if __name__ == "__main__":
    df = loadCSV(DATA)

    # Models
    logistic_reg = LogisticRegression()
    create_model(df, logistic_reg, "Logistic Regression")

    random_forest = RandomForestClassifier()
    create_model(df, random_forest, "Random Forest Classifier")

    svc = SVC()
    create_model(df, svc, "Support Vector Machine (SVC)")

    gradient = GradientBoostingClassifier()
    create_model(df, gradient, "Gradient Boosting")

    k_neighbors = KNeighborsClassifier()
    create_model(df, k_neighbors, "K-Nearest Neighbors (KNN)")
