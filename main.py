import numpy as np
import pandas as pd
import streamlit as st
from time import perf_counter

DATA = "./data/heart_failure_clinical_records.csv"

def loadCSV(path: str):
    df = pd.read_csv(path)
    showPlot(df)

def showPlot(df: pd.DataFrame):
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
    # st.table(df_no_death.median())
    st.table(no_death_median)
    # st.markdown("#### Mean")
    # st.table(df_no_death.mean())
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
    # st.table(df_death.median())
    st.table(death_median)
    # st.markdown("#### Mean")
    # st.table(df_death.mean())
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

if __name__ == "__main__":
    # process = Process(target=loadCSV, args=("./data/incom vs education.csv",))
    # process.start()
    # process.join()
    loadCSV(DATA)
