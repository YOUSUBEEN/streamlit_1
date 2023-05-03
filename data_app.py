# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import utils


@st.cache_data
def load_data():
    train = pd.read_csv(utils.train_path)
    test = pd.read_csv(utils.test_path)
    transactions = pd.read_csv(utils.transactions_path)
    stores = pd.read_csv(utils.stores_path)
    oil = pd.read_csv(utils.oil_path)
    holidays = pd.read_csv(utils.holidays_path)

    return train, test, transactions, stores, oil, holidays

def data_app():

    # 요약 정보를 출력하기 위한 함수 생성
    def summary(dataframe):

        # 화면 분할을 위한 컬럼 설정 2:1 비율
        col1, col2 = st.columns([2, 1])

        with col1:
            st.title("📣 Data")
            st.dataframe(dataframe, height=810, width=900)

        with col2:
            st.title("📣 Data Type")
            st.dataframe(dataframe.dtypes, height=350, width=500)

            st.title("📣 Describe")
            st.dataframe(dataframe.describe(), height=350, width=500)

    # 데이터 불러오기
    train, test, transactions, stores, oil, holidays = load_data()

    # 데이터 딕셔너리 생성
    datalist_dict = {
        "Train" : train,
        "Test" : test,
        "Transactions" : transactions,
        "Stores" : stores,
        "Oil" : oil,
        "Holidays/Events" : holidays
    }

    # selectbox 생성
    datalist = st.selectbox("SELECT DATA", list(datalist_dict.keys()), index = 0)
    st.markdown("---")
    st.subheader(f"📝{datalist} Data Description")
    # 조건문으로 요약 정보 넣기
    if datalist == "Train":
        st.markdown("""✔ 이 훈련 데이터는 상점 번호, 제품군, 프로모션 및 목표 매출로 구성된 시계열 기능으로 구성된 데이터입니다.""")
        st.markdown("✔ store_nbr은 제품이 판매되는 상점을 나타냅니다.")
        st.markdown("✔ family는 판매되는 제품 유형을 나타냅니다.")
        st.markdown("✔ sales는 특정 날짜에 특정 가게에서 제품군의 총 매출을 나타냅니다. 제품은 소수점 단위로 판매될 수 있으므로 분수 값이 가능합니다.")
        st.markdown("✔ onpromotion은 특정 날짜에 상점에서 프로모션 중인 제품군의 항목 수를 나타냅니다.")
        summary(train)
    elif datalist == "Test":
        st.markdown("✔ 학습 데이터와 동일한 기능을 가지는 테스트 데이터입니다. 이 파일의 날짜에 대한 목표 매출을 예측할 것입니다.")
        st.markdown("✔ 테스트 데이터의 날짜는 학습 데이터의 마지막 날짜 이후 15일 동안입니다.")
        summary(test)
    elif datalist == "Transactions":
        st.markdown("✔ 올바른 형식의 샘플 제출 파일입니다.")
        summary(transactions)
    elif datalist == "Stores":
        st.markdown("✔ 도시, 주, 유형, 클러스터를 포함한 상점 메타데이터입니다.")
        st.markdown("✔ 클러스터는 유사한 상점의 그룹화입니다.")
        summary(stores)
    elif datalist == "Oil":
        st.markdown("✔ 일일 유가, 학습 및 테스트 데이터 기간 모두의 값을 포함합니다.")
        st.markdown("✔ 에콰도르는 석유 의존국이며, 석유 가격 충격에 매우 민감합니다.")
        summary(oil)
    elif datalist == "Holidays/Events":
        st.markdown("✔ 메타데이터와 함께 휴일 및 이벤트 정보가 포함된 파일입니다.")
        st.markdown("""✔ 참고: transferred 열에 주목해야 합니다. 
        공식적으로 이전되는 공휴일은 해당 달력의 날에 해당하지만 정부에 의해 다른 날짜로 이동되었습니다. 
        이전된 날은 휴일보다는 일반적인 날과 유사합니다. 날짜를 찾으려면 해당 행의 유형이 Transfer인 행을 찾으면 됩니다. 
        예를 들어 Independencia de Guayaquil의 휴일은 2012-10-09에서 2012-10-12로 이전되었으며, 이는 2012-10-12에 기념되었음을 의미합니다. 
        Bridge 유형의 날은 휴일이 추가되는 추가 일입니다. (예 : 긴 주말을 연장하기 위해서). 이런 경우 일반적으로 Bridge에 대한 보상으로 예정되지 않은 근무일(Work Day))로 구성되는 경우가 많습니다. """)
        st.markdown("✔ 추가적인 휴일은 일반적인 달력 휴일에 추가되는 날입니다. 예를 들어, 전형적으로 크리스마스 이브를 휴일로 만드는 것과 같이.")
        summary(holidays)