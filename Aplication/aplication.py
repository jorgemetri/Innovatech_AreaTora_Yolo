import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import string
import altair as alt
from streamlit_extras.metric_cards import style_metric_cards


@st.cache_data
def load_model(model):
    pass
def Tabela():
    pass
tab1, tab2 = st.tabs(["📊 Aplicação", "📥 Baixar dados"])

with tab1:
    st.title("Aplicaçãos :chart_with_upwards_trend:")
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)

    if picture:
        st.image(picture)



with tab2:
    st.write("📥 Baixar Dados")
