import streamlit as st

import pandas as pd
import numpy as np
np.random.seed(7)
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True, context="talk")

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


def dataInsight(path):
    raw_input_data = pd.read_csv(path)
    df = pd.DataFrame(raw_input_data)
    st.write(df.columns)
    b = df['driver_race'] == 'Black'
    h = df['driver_race'] == 'Hispanic'
    w = df['driver_race'] == 'White'
    ticket = df['stop_outcome'] == 'Ticket'
    sp = df['violation'] == 'Speeding'
    m = df['driver_gender'] == 'M'
    f = df['driver_gender'] == 'F'
    tk = df[ticket]
    print('Ticket', tk.shape)
    s = df[sp]
    print('Speed', s.shape)
    ma = df[m]
    print("male",ma.shape)
    fm = df[f]
    print("Female", fm.shape)
    a = df[b]
    print("Black", a.shape)
    bl = df[h]
    print("Hispanic", bl.shape)
    wh = df[w]
    print("White", wh.shape)
    print('Black ticket', df[b&ticket].shape)
    print('White ticket', df[w&ticket].shape)
    print('Black, speed, ticket', df[b&sp&ticket].shape)
    print('Black, f, speed, ticket', df[b&f&sp&ticket].shape)
    print('Hispanic, speed, ticket', df[h&sp&ticket].shape)
    print('Hispanic, f, speed, ticket', df[h&f&sp&ticket].shape)
    print('white, speed, ticket', df[w&sp&ticket].shape)

dataInsight('../../data/raw/CT_Cleaned_raw.csv')
