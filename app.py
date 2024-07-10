#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import sys
import contextlib
import pandas as pd
import streamlit as st
from navicat_spock.spock import run_spock_from_args

# Add spock directory to system path if not already present
spock_dir: str = os.path.dirname(os.path.abspath(__file__)) + "/spock"
if spock_dir not in sys.path:
    sys.path.append(spock_dir)


# Check if the dataframe contains a target column
def check_columns(df: pd.DataFrame) -> None:
    if not any(["target" in col.lower() for col in df.columns]):
        raise ValueError(
            "Missing the target column. Please add a column that contains `target` in the name."
        )


# Cache the function to run spock with the provided dataframe and arguments
@st.cache_data(
    show_spinner=False, hash_funcs={pd.DataFrame: lambda df: df.to_numpy().tobytes()}
)
def run_fn(df, *args, **kwargs) -> str:
    check_columns(df)
    fig, ax = run_spock_from_args(df, *args, **kwargs)
    return fig


# Mock function for testing purposes
def mock_fn(df, *args, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt

    check_columns(df)
    print("WORKING")
    fig, ax = plt.subplots()
    ax.plot(np.random.rand(10))
    return fig


# Load data from the uploaded file
def load_data(file):
    accepted_ext = ["csv", "xlsx"]
    if file.name.split(".")[-1] not in accepted_ext:
        raise ValueError("Invalid file type. Please upload a CSV or Excel file.")
    return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)


# Context manager to capture stdout with a timestamp
@contextlib.contextmanager
def capture_stdout_with_timestamp():
    class TimestampedIO(io.StringIO):
        def write(self, msg):
            if msg.strip():  # Only add a timestamp if the message is not just a newline
                timestamped_msg = f"[{pd.Timestamp.now()}] {msg}"
            else:
                timestamped_msg = msg
            super().write(timestamped_msg)

    new_stdout = TimestampedIO()
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout


# Main function to run the Streamlit app
def main():
    st.title("Navicat Spock")
    st.subheader("A tool for generating volcano plots from your data")

    with st.sidebar:
        st.header("Settings")

        wp = st.number_input(
            "Weighting Power",
            min_value=0,
            value=2,
            help="Weighting power used to adjust the target values",
        )
        verb = st.number_input(
            "Verbosity",
            min_value=0,
            max_value=7,
            value=1,
            help="Verbosity level (0-7) for the logs",
        )

        imputer_strat_dict = {
            None: "none",
            "Iterative": "iterative",
            "Simple": "simple",
            "KNN": "knn",
        }
        imputer_strat_value = st.selectbox(
            "Imputer Strategy",
            filter(lambda x: x, list(imputer_strat_dict.keys())),
            index=None,
            help="Imputer Strategy used to fill missing values",
        )

        imputer_strat = imputer_strat_dict[imputer_strat_value]

        plotmode = st.number_input(
            "Plot Mode",
            min_value=0,
            max_value=3,
            value=1,
            help="Different plot modes",
        )
        seed = st.number_input(
            "Seed", min_value=0, value=None, help="Seed number to fix the random state"
        )
        prefit = st.toggle("Prefit", value=False)
        setcbms = st.toggle("CBMS", value=True)

    with st.expander("Instructions"):
        st.markdown(
            """
            1. Upload your data in an Excel or CSV file.
            2. View and curate your data in the table below.
            3. Click "Run Plot" to generate your plot.
            4. View the generated plot and all the associated logs in the respective tabs.
            """
        )

    uploaded_file = st.file_uploader(
        "Choose a file", type=["csv", "xlsx"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.markdown("### Data")
            st.dataframe(df, use_container_width=True)

            if st.button("Run Plot"):
                with st.spinner("Generating plot..."):
                    with capture_stdout_with_timestamp() as stdout_io:
                        result = run_fn(
                            df,
                            wp=wp,
                            verb=verb,
                            imputer_strat=imputer_strat,
                            plotmode=plotmode,
                            seed=seed,
                            prefit=prefit,
                            setcbms=setcbms,
                            fig=None,
                            ax=None,
                        )

                    st.markdown("### Result")
                    plot, logs = st.tabs(["Plot", "Logs"])
                    with plot:
                        st.pyplot(result)
                    with logs:
                        st.code(stdout_io.getvalue(), language="bash")
        except Exception as e:
            st.toast(f":red[{e}]", icon="🚨")
    else:
        st.write("Please first upload a file to generate the volcano plot.")


if __name__ == "__main__":
    main()
