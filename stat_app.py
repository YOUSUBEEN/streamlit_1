# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_option_menu import option_menu
import utils
import intro_app
import data_app
import eda_app
import pandas as pd

@st.cache_data
def load_data():
    train = pd.read_csv(utils.train_path)
    test = pd.read_csv(utils.test_path)
    transactions = pd.read_csv(utils.transactions_path)
    stores = pd.read_csv(utils.stores_path)
    oil = pd.read_csv(utils.oil_path)
    holidays = pd.read_csv(utils.holidays_path)

    return train, test, transactions, stores, oil, holidays

def stat_app():

    # 데이터 불러오기
    train, test, transactions, stores, oil, holidays = load_data()

    analysislist = ['Holidays and Events', 'Time Related Features', 'Did Earhquake affect the store Sales?', 'ACF & PACF for each famliy', 'Simple Moving Average', 'Exponential Moving Average' ]
    datalist = st.selectbox("Select Ananlysis", analysislist, index=0)
    st.markdown("---")
    st.subheader(f"📝{datalist} Data Description")
