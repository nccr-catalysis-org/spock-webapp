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


ACCEPTED_FILE_EXT = ["csv", "xlsx", "xls", "xlsm", "xlsb", "odf", "ods", "odt"]


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
            save_fig=False,
            save_csv=False,
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
def load_data(filepath, filename):
    if filename.split(".")[-1] not in ACCEPTED_FILE_EXT:
        raise ValueError(
            f"Invalid file type. Please upload a file with one of the following extensions: {ACCEPTED_FILE_EXT}"
        )
    return (
        pd.read_csv(filepath) if filename.endswith(".csv") else pd.read_excel(filepath)
    )


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
    filepath = None
    filename = None

    st.write("Choose a dataset or upload your own file")

    option = st.radio("Select an option:", ["Use example dataset", "Upload file"])

    if option == "Use example dataset":
        examples = {
            "Clean dataset": {
                "filepath": "examples/clean_data_example.xlsx",
                "description": "The clean dataset is a reference dataset that includes 1 target variable and 2 descriptors. This is a typical example, where the goal of the model is to find a single descriptor or a combined descriptor (mathematical function of descriptor 1 and 2) that gives volcano like correlation with the target variable.",
            },
            "Noisy dataset": {
                "filepath": "examples/noisy_data_example.xlsx",
                "description": "The noisy dataset is a reference dataset that includes 1 target variable and 1 descriptor. This is a specific example where the kinetic data was compiled from duplicate or triplicate experiments, and the performance metric (target variable) is represented by the average value and standard deviation. In such instances, a single experimental is populated over three rows, such the first, second, and third row contains information on the upper bound, mean, and lower bound data, respectively, of the performance metric. The descriptor is values corresponding to these observations remain the same. The model fits through the data and generates a volcano-like trend.",
            },
        }

        selected_example = st.selectbox(
            "Choose an example dataset",
            list(examples.keys()),
            # format_func=lambda x: examples[x]["description"],
        )

        st.info(examples[selected_example]["description"])

        if st.button("Load Example", use_container_width=True):
            filepath = examples[selected_example]["filepath"]
            filename = filepath.split("/")[-1]
    else:
        file = st.file_uploader("Upload a file", type=ACCEPTED_FILE_EXT)
        if file is not None:
            filepath = file
            filename = file.name

    if filepath is not None and filename is not None:
        try:
            df = load_data(filepath, filename)
            st.session_state.df = df
            st.session_state.filename = filename
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")


@st.experimental_dialog("Guidelines", width="large")
def guidelines():
    st.write(
        """
To effectively populate an Excel sheet (.xlsx or .csv format) to upload on the SPOCK web-app, we recommend the following practices to ensure the data is curated and organized properly. Begin by placing the name of the catalyst in the first column, followed by performance metric (target variable) in the second column. The header of the second column must be labeled as Target Tr and ensure this column does not have any missing or erroneous entries. Next, place each descriptor of interest (input features) in the adjacent columns, one variable per column (see the example provided: “clean_data_example”).  Label each column clearly and ensure all cells are filled correctly without missing values or placeholders like "NAN" or "NA". All variables, including the performance metric and descriptors, must be numerical values and follow consistent formatting styles and decimal points. Double-check for outliers or anomalies that could skew model training and remove or correct these entries if necessary. 

In cases where the kinetic data was compiled from duplicate or triplicate experiments, the performance metric will be represented by the average value and standard deviation. In such instances, a single experimental observation needs to be populated over three rows, such the first, second, and third row contains information on the upper bound, mean, and lower bound data, respectively, of the performance metric. The descriptor values corresponding to these observations remain the same (see the example provided: “noisy_data_example”). Before proceeding with model training, validate the calculations for the mean and standard deviations by cross-checking with the original raw data and using statistical formulas to ensure accuracy. Document all essential transformations or preprocessing steps in a separate document linked to the main sheet. This documentation helps ensure transparency and reproducibility in subsequent steps of the project. Maintain version control to track changes and updates to the dataset, ensuring long term reproducibility of results.
"""
    )


def main():
    st.set_page_config(
        page_title="Navicat Spock",
        page_icon="🌋",
        initial_sidebar_state="expanded",
    )
    _, center, _ = st.columns(spec=[0.2, 0.6, 0.2])
    center.image("res/spock_logo.png")
    st.subheader("Generate volcano plots from catalytic data")

    # Instructions
    with st.expander("Instructions", expanded=False):
        known_tab, unknown_tab = st.tabs(["Descriptor Known", "Descriptor Unknown"])

        with known_tab:
            st.markdown(
                """
            ### When the Descriptor is Known
            
            1. **Prepare Your Data**
               - Organize data in a tabular format
               - Column 1: Catalyst name
               - Column 2: Performance metric
               - Column 3: Descriptor
               - Label columns according to guidelines
            
            2. **Import Data**
               - Click "Import Data" to upload your Excel or CSV file
            
            3. **Review and Adjust**
               - Check your data in the displayed table
               - Modify plot settings in the sidebar if needed
            
            4. **Generate and Analyze**
               - Click "Generate plot"
               - Examine the plot and logs in their respective tabs
            
            5. **Refine Results**
               - Adjust the weighting power parameter
               - Repeat steps 4-5 until you achieve satisfactory results
            """
            )

        with unknown_tab:
            st.markdown(
                """
            ### When the Descriptor is Unknown
            
            1. **Prepare Your Data**
               - Organize data in a tabular format
               - Column 1: Catalyst name
               - Column 2: Performance metric
               - Columns 3+: Potential descriptors
               - Label columns according to guidelines
            
            2. **Import Data**
               - Click "Import Data" to upload your Excel or CSV file
            
            3. **Review and Adjust**
               - Check your data in the displayed table
               - Modify plot settings in the sidebar if needed
            
            4. **Generate and Analyze**
               - Click "Generate plot"
               - Examine the plot and logs in their respective tabs
            
            5. **Refine Results**
               - Adjust the weighting power parameter
               - Repeat steps 4-5 until you achieve satisfactory results
            """
            )

        if st.button(
            "Click here for more information/guidelines", use_container_width=True
        ):
            guidelines()

    if "df" not in st.session_state:
        if st.button("Import Data", type="primary", use_container_width=True):
            import_data()
        st.stop()

    # Display the data
    st.header(f"Dataset : {st.session_state.filename}")
    st.dataframe(st.session_state.df, use_container_width=True)

    # Settings
    with st.sidebar:
        st.header("Settings")

        wp = st.number_input(
            "Weighting Power",
            value=1,
            help="The weighting power is the tuning parameter to fit the line segments on the data. Default value is set to 1. We recommend to vary this value between 0-3 for desired results.",
        )
        verb = st.number_input(
            "Verbosity",
            min_value=0,
            max_value=7,
            value=3,
            help="This parameter is used to generate reports based on the outcome of the mode liftting. Default value is set to 3. We recommend to vary this value between 2-5 for desired level of report (log) generation.",
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
            min_value=1,
            max_value=3,
            value=2,
            help="Different plot modes",
        )
        seed = st.number_input(
            "Seed", min_value=0, value=None, help="Seed number to fix the random state"
        )
        prefit = st.toggle("Prefit", value=False)
        setcbms = st.toggle("CBMS", value=True)

    # Option to import new data
    if st.button("Import New Data", type="secondary", use_container_width=True):
        import_data()

    if st.button("Generate plot", type="primary", use_container_width=True):
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
