
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(
    page_title="Ahmedabad House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background-color: #111A2E;
    border-right: 1px solid #26334D;
}

h1 {
    color: #F8FAFC;
    font-size: 34px !important;
    font-weight: 800;
}

h2, h3 {
    color: #DCE6F7;
}

[data-testid="stMetric"] {
    background-color: #17243A;
    border: 1px solid #2A3B58;
    padding: 20px;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

[data-testid="stMetricValue"] {
    color: #60A5FA;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

[data-testid="stFileUploader"] {
    background-color: #17243A;
    border: 1px dashed #3B82F6;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stFileUploader"] button {
    background-color: #D53F8C !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #B83280 !important;
    color: white !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #2A3B58;
    border-radius: 12px;
}

hr {
    border-color: #26334D;
}
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "model_data" not in st.session_state:
    st.session_state.model_data = None

with st.sidebar:
    st.title("🏠 HouseAI")
    st.caption("Ahmedabad Real Estate Analytics")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "Dataset", "Model Comparison"]
    )

    st.divider()
    st.caption("Machine Learning Project")
    st.caption("Python • Streamlit • Scikit-learn")

st.title("Ahmedabad House Price Prediction")
st.write("Explore property data and compare machine learning models.")

uploaded_file = st.file_uploader(
    "Upload your Ahmedabad housing dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    if "Area" not in df.columns or "Price" not in df.columns:
        st.error("Your CSV must contain Area and Price columns.")
        st.stop()

    df["Area"] = pd.to_numeric(
        df["Area"], errors="coerce"
    )

    df["Price"] = pd.to_numeric(
        df["Price"], errors="coerce"
    )

    df = df.dropna(subset=["Area", "Price"])
    df = df.dropna()

    if df.empty:
        st.error("No valid data available after cleaning.")
        st.stop()

    st.session_state.dataset = df

    if page == "Dashboard":

        st.subheader("Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Properties",
                f"{len(df):,}"
            )

        with col2:
            st.metric(
                "Average House Price",
                f"₹ {df['Price'].mean():,.2f}"
            )

        with col3:
            st.metric(
                "Average Area",
                f"{df['Area'].mean():,.2f}"
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Area vs Price")

            fig1, ax1 = plt.subplots()

            ax1.scatter(
                df["Area"],
                df["Price"],
                color="steelblue",
                alpha=0.6
            )

            ax1.set_xlabel("Area")
            ax1.set_ylabel("Price")
            ax1.set_title("Area vs Price")

            fig1.patch.set_facecolor("#17243A")
            ax1.set_facecolor("#17243A")
            ax1.tick_params(colors="white")
            ax1.xaxis.label.set_color("white")
            ax1.yaxis.label.set_color("white")
            ax1.title.set_color("white")

            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            st.subheader("Price Distribution")

            fig2, ax2 = plt.subplots()

            ax2.hist(
                df["Price"],
                bins=30,
                color="mediumpurple"
            )

            ax2.set_xlabel("Price")
            ax2.set_ylabel("Count")
            ax2.set_title("Price Distribution")

            fig2.patch.set_facecolor("#17243A")
            ax2.set_facecolor("#17243A")
            ax2.tick_params(colors="white")
            ax2.xaxis.label.set_color("white")
            ax2.yaxis.label.set_color("white")
            ax2.title.set_color("white")

            st.pyplot(fig2)
            plt.close(fig2)

        st.divider()

        st.subheader("Train Machine Learning Models")

        if st.button("🚀 Train All Models"):

            if len(df) < 2:
                st.error("At least two rows are required.")

            else:
                train_df = pd.get_dummies(
                    df,
                    drop_first=True
                )

                X = train_df.drop("Price", axis=1)
                y = train_df["Price"]

                if X.empty:
                    st.error("No input features available.")

                else:
                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42
                        )
                    )

                    models = {
                        "Linear Regression": LinearRegression(),
                        "Lasso": Lasso(),
                        "Ridge": Ridge(),
                        "Decision Tree": DecisionTreeRegressor(
                            random_state=42
                        ),
                        "Random Forest": RandomForestRegressor(
                            n_estimators=100,
                            random_state=42
                        )
                    }

                    results = []

                    with st.spinner("Training models..."):
                        for name, model in models.items():

                            model.fit(X_train, y_train)

                            y_pred = model.predict(X_test)

                            results.append({
                                "Model": name,
                                "R2 Score": round(
                                    r2_score(y_test, y_pred), 4
                                ),
                                "MSE": round(
                                    mean_squared_error(
                                        y_test, y_pred
                                    ), 2
                                )
                            })

                    st.session_state.results = pd.DataFrame(
                        results
                    )

                    st.session_state.model_data = {
                        "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test
                    }

                    st.success("All models trained successfully!")

        if st.session_state.results is not None:

            results_df = st.session_state.results

            st.subheader("Model Results")

            st.dataframe(
                results_df,
                use_container_width=True
            )

            best = results_df.loc[
                results_df["R2 Score"].idxmax()
            ]

            st.success(
                f"🏆 Best Model: {best['Model']} | "
                f"R2 Score: {best['R2 Score']}"
            )
        
    elif page == "Dataset":
        st.subheader("Dataset Preview")
        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.subheader("Dataset Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        st.subheader("Missing Values")

        st.dataframe(
            df.isnull().sum().to_frame("Missing Values"),
            use_container_width=True
        )

        st.subheader("Download Dataset")

        st.download_button(
            "Download Cleaned CSV",
            data=df.to_csv(index=False),
            file_name="ahmedabad_cleaned.csv",
            mime="text/csv"
        )

    elif page == "Model Comparison":

        st.subheader("ML Model Comparison")

        if st.session_state.results is None:

            st.info(
                "Go to Dashboard and click Train All Models first."
            )

        else:

            results_df = st.session_state.results

            st.dataframe(
                results_df,
                use_container_width=True
            )

            st.subheader("R2 Score Comparison")

            fig3, ax3 = plt.subplots()

            ax3.bar(
                results_df["Model"],
                results_df["R2 Score"],
                color="steelblue"
            )

            ax3.set_ylabel("R2 Score")
            ax3.tick_params(axis="x", rotation=25)

            st.pyplot(fig3)
            plt.close(fig3)

            st.subheader("MSE Comparison")

            fig4, ax4 = plt.subplots()

            ax4.bar(
                results_df["Model"],
                results_df["MSE"],
                color="mediumpurple"
            )

            ax4.set_ylabel("Mean Squared Error")
            ax4.tick_params(axis="x", rotation=25)

            st.pyplot(fig4)
            plt.close(fig4)

            st.download_button(
                "⬇️ Download Model Results",
                data=results_df.to_csv(index=False),
                file_name="model_results.csv",
                mime="text/csv"
            )

else:

    st.info("👆 Upload your Ahmedabad.csv file to get started.")

    st.markdown("""
    ### Welcome to HouseAI

    Upload your real estate dataset to:

    - Explore house prices and property areas.
    - View data visualizations.
    - Train five machine learning models.
    - Compare model performance.
    """)
