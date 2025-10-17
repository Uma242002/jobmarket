# Base Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")

# UI Library

import streamlit as st

################################# Dashboard ##########################################

# Ref for Streamlit commands: https://docs.streamlit.io/develop/api-reference

# Black background using custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;  /* optional: makes text readable on black background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 style='text-align: center; color: green;'>Job Marketing Data Analytics</h3>", unsafe_allow_html=True)
st.divider()
st.markdown("#### :blue[Data Taken For Analysis..]")
df = pd.read_csv("naukri_processed1.csv")
st.dataframe(df.head())
st.divider()
st.subheader(":red[Uni-Variate Analytics (Single Column Data Study):]")
cola, colb , colc = st.columns(3)
with colb:
    colname = st.selectbox("Select Column:", df.columns)
# Create 2 columns for layout
col1, col2 = st.columns(2)

# --- CATEGORICAL or OBJECT type ---
if df[colname].dtype == object or pd.api.types.is_categorical_dtype(df[colname]):
    top10 = df[colname].value_counts().nlargest(10)

    with col1:
        st.subheader("📌 Descriptive Stats")
        st.write("Top 10 Category Counts:")
        st.write(top10)
        st.write("Total Unique Categories:", df[colname].nunique())
        st.write("Most Frequent:", df[colname].mode()[0])

    with col2:
       
       st.subheader("📈 Pie chart")
       fig = px.pie(names=top10.index,values=top10.values,title=f"Top 10 Categories in {colname}",
     )
       fig.update_traces(textposition='inside', textinfo='percent+label')
       fig.update_layout(showlegend=True)

       st.plotly_chart(fig)

# --- FLOAT type ---
elif pd.api.types.is_float_dtype(df[colname]):
    rounded = df[colname].round()

    with col1:
        st.subheader("📌 Descriptive Stats")
        st.write(df[colname].describe())
        st.write("Rounded Unique Values:", rounded.nunique())

    with col2:
        st.subheader("📈 Histogram")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(rounded, kde=True, bins=10, ax=ax)
        st.pyplot(fig)

# --- INTEGER type ---
elif pd.api.types.is_integer_dtype(df[colname]):
    with col1:
        st.subheader("📌 Descriptive Stats")
        st.write(df[colname].describe())
        st.write("Unique Values:", df[colname].nunique())
        st.write("Mode:", df[colname].mode()[0])

    with col2:
        st.subheader("📈 Histogram")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(df[colname], kde=True, bins=10, ax=ax)
        st.pyplot(fig)

# --- Others ---
else:
    st.warning("Unsupported column type for analysis.")
st.divider()
st.subheader(":red[Bi-Variate Analytics (two Column Data Study):]")
st.divider()
# Select two columns
cola, colb,colc= st.columns(3)
with cola:
    colx = st.selectbox("Select X Column", df.columns)
with colc:
    coly = st.selectbox("Select Y Column", df.columns)

# Remove missing values
data = df[[colx, coly]].dropna()

# Data types
x_type = df[colx].dtype
y_type = df[coly].dtype

# Create columns for display
col1, col2 = st.columns(2)
if colx != coly:
    x_type = data[colx].dtype
    y_type = data[coly].dtype

    # ---------------------- 📌 Descriptive Statistics ----------------------
    with col1:
        st.subheader("📌 Descriptive Stats")
        
        if pd.api.types.is_numeric_dtype(x_type) and pd.api.types.is_numeric_dtype(y_type):
            corr = data[colx].corr(data[coly])
            st.write(f"**Correlation between {colx} and {coly}:** `{corr:.3f}`")
        
        elif pd.api.types.is_object_dtype(x_type) and pd.api.types.is_numeric_dtype(y_type):
            group_stats = data.groupby(colx)[coly].sum().reset_index()
            st.write(f"**{coly} Grouped by {colx}**")
            st.dataframe(group_stats)
        
        elif pd.api.types.is_numeric_dtype(x_type) and pd.api.types.is_object_dtype(y_type):
            group_stats = data.groupby(coly)[colx].sum().reset_index()
            st.write(f"**{colx} Grouped by {coly}**")
            st.dataframe(group_stats)
        
        else:
            crosstab = pd.crosstab(data[colx], data[coly])
            st.write(f"**Crosstab between {colx} and {coly}**")
            st.dataframe(crosstab)

    # ---------------------- 📈 Visualization ----------------------
    with col2:
        st.subheader("📈 Visualization")
        
        if pd.api.types.is_numeric_dtype(x_type) and pd.api.types.is_numeric_dtype(y_type):
            fig, ax = plt.subplots()
            sns.scatterplot(data=data, x=colx, y=coly, ax=ax)
            st.pyplot(fig)
        
        elif pd.api.types.is_object_dtype(x_type) and pd.api.types.is_numeric_dtype(y_type):
            top10 = data[colx].value_counts().nlargest(10).index
            filtered = data[data[colx].isin(top10)]
            fig, ax = plt.subplots()
            sns.boxplot(data=filtered, x=colx, y=coly, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        elif pd.api.types.is_numeric_dtype(x_type) and pd.api.types.is_object_dtype(y_type):
            top10 = data[coly].value_counts().nlargest(10).index
            filtered = data[data[coly].isin(top10)]
            fig, ax = plt.subplots()
            sns.boxplot(data=filtered, x=coly, y=colx, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        else:
            fig = px.sunburst(data, path=[colx, coly])
            st.plotly_chart(fig)
else:
    st.warning("Please select two different columns.")
st.divider()
st.subheader(":red[Multi-Variate Analytics]")
st.divider()
selected_cols = st.multiselect("Select Multiple Columns for Multivariate Analysis:", df.columns)
if len(selected_cols) < 2:
    st.warning("Please select at least **two** columns to perform multivariate analysis.")
else:
    data = df[selected_cols].dropna()
    num_cols = data.select_dtypes(include='number').columns.tolist()
    cat_cols = data.select_dtypes(include='object').columns.tolist()

    col1, col2 = st.columns(2)

    # ----------------- 📌 Descriptive Statistics -----------------
    with col1:
        st.subheader("📌 Descriptive Stats")
        
        if len(num_cols) >= 2:
            st.write("**Correlation Matrix:**")
            corr_matrix = data[num_cols].corr()
            st.dataframe(corr_matrix.round(2))
        
        if cat_cols and num_cols:
            for cat in cat_cols:
                st.markdown(f"**Group-wise statistics by `{cat}`**")
                grouped = data.groupby(cat)[num_cols].describe().reset_index()
                st.dataframe(grouped)
        if not num_cols and cat_cols:
            st.write("**Frequency Counts for All Categorical Columns:**")
            for col in cat_cols:
                st.markdown(f"**{col}**")
                freq_df = data[col].value_counts().reset_index()
                freq_df.columns = [col, "Count"]
                st.dataframe(freq_df)

    # ----------------- 📈 Visualizations -----------------
    with col2:
        st.subheader("📈 Visualizations")

        # Heatmap for correlation
        if len(num_cols) >= 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(data[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

        # Pairplot
        if len(num_cols) >= 2:
            st.write("📊 Pairplot of Numeric Variables:")
            fig = sns.pairplot(data[num_cols])
            st.pyplot(fig)

        # Boxplots for categorical vs numerical
        if cat_cols and num_cols:
            for cat in cat_cols:
                top10 = data[cat].value_counts().nlargest(10).index
                filtered = data[data[cat].isin(top10)]
                for num in num_cols:
                    st.write(f"**Boxplot: `{num}` by `{cat}`**")
                    fig, ax = plt.subplots()
                    sns.boxplot(data=filtered, x=cat, y=num, ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        if not num_cols and cat_cols:
            st.write("📊 Frequency Bar Plots for Categorical Columns:")
            for col in cat_cols:
                freq = data[col].value_counts().nlargest(10)  # Top 10 categories
                fig, ax = plt.subplots()
                sns.barplot(x=freq.index, y=freq.values, ax=ax, palette="viridis")
                ax.set_title(f"Top Categories in `{col}`")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig)

