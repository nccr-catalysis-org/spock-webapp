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
    show_spinner=False,
    # hash_funcs={pd.DataFrame: lambda df: df.to_numpy().tobytes()},
)
def cached_run_fn(df, wp, verb, imputer_strat, plotmode, seed, prefit, setcbms):
    with capture_stdout_with_timestamp() as stdout_io:
        fig, _ = run_spock_from_args(
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
    return fig, stdout_io.getvalue()


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


@st.experimental_dialog("Import Data")
def import_data():
    st.write("Choose a dataset or upload your own file")

    option = st.radio("Select an option:", ["Use example dataset", "Upload file"])

    if option == "Use example dataset":
        examples = {
            "Sabatier": "examples/sabatier.csv",
            # Add more examples here
        }
        selected_example = st.selectbox(
            "Choose an example dataset", list(examples.keys())
        )
        if st.button("Load Example"):
            df = pd.read_csv(examples[selected_example])
            st.session_state.df = df
            st.rerun()
    else:
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file", type=["csv", "xlsx"]
        )
        if uploaded_file is not None:
            try:
                df = load_data(uploaded_file)
                st.session_state.df = df
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")


def main():
    st.title("Navicat Spock")
    st.subheader("Generate volcano plots from your data")

    # Instructions
    with st.expander("Instructions", expanded=False):
        st.markdown(
            """
            1. Click "Import Data" to upload a file or select an example dataset.
            2. Review your data in the table.
            3. Adjust the plot settings in the sidebar if needed.
            4. Click "Generate plot" to create your plot.
            5. View the generated plot and logs in the respective tabs.
            """
        )

    if "df" not in st.session_state:
        if st.button("Import Data"):
            import_data()
        st.stop()

    # Display the data
    st.header("Review the data")
    st.dataframe(st.session_state.df, use_container_width=True)

    # Option to import new data
    if st.button("Import New Data"):
        import_data()

    # Settings
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

    # Run the plot
    st.header("Generate plot")
    if st.button("Generate plot"):
        with st.spinner("Generating plot..."):
            fig, logs = cached_run_fn(
                st.session_state.df,
                wp=wp,
                verb=verb,
                imputer_strat=imputer_strat,
                plotmode=plotmode,
                seed=seed,
                prefit=prefit,
                setcbms=setcbms,
            )

        st.header("Results")
        plot, logs_tab = st.tabs(["Plot", "Logs"])
        with plot:
            st.pyplot(fig)
        with logs_tab:
            st.code(logs, language="bash")


if __name__ == "__main__":
    main()
