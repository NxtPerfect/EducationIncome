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
    df_og = df.copy(deep=True)
    df_og = df_og.drop(index=df_og[df_og["DEATH_EVENT"] == 0].index)
    df_og = df_og.drop("DEATH_EVENT", axis=1)
    df_og.sort_values(by="creatinine_phosphokinase")
    st.markdown("### No deaths")
    st.dataframe(df_og)
    st.markdown("#### Median")
    st.table(df_og.median())
    # st.markdown("#### Mean")
    # st.table(df_og.mean())
    no_death_creatinine = df_og.median()["creatinine_phosphokinase"]
    no_death_time = df_og.median()["time"]
    st.bar_chart(df_og[["creatinine_phosphokinase"]])

    # Chart with only deaths
    df = df.drop(index=df[df["DEATH_EVENT"] == 1].index)
    df = df.drop("DEATH_EVENT", axis=1)
    df.sort_values(by="creatinine_phosphokinase")
    st.markdown("### Only deaths")
    st.dataframe(df)
    st.markdown("#### Median")
    st.table(df.median())
    # st.markdown("#### Mean")
    # st.table(df.mean())
    death_creatinine = df.median()["creatinine_phosphokinase"]
    death_time = df.median()["time"]
    st.bar_chart(df[["creatinine_phosphokinase"]])

    st.markdown("## Results")
    st.markdown(f'Percentage change of creatinine phosphokinase for alive vs dead patients is {no_death_creatinine} - {death_creatinine}=**{((no_death_creatinine - death_creatinine) / no_death_creatinine) * 100.0:.2f}%**. This means if your creatinine phosphokinase is around 230mcg/L, you might be at risk of heart failure. The time between follow up appointment for patients who died is {death_time} - {no_death_time}=**{((death_time - no_death_time) / death_time) * 100.0:.2f}%** higher, during which they were found dead.')

if __name__ == "__main__":
    # process = Process(target=loadCSV, args=("./data/incom vs education.csv",))
    # process.start()
    # process.join()
    loadCSV(DATA)
