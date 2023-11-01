import itertools
import uuid
from typing import Dict, Any
import numpy as np
import streamlit as st
import pandas as pd
import chardet
from io import StringIO
from ftfy import fix_text

st.set_page_config(layout="wide")
time_format_options = {
    "Seconds": "%S",
    "Minutes": "%M",
    "Hours": "%H",
    "Hours:Minutes": "%H:%M",
    "Hours:Minutes:Seconds": "%H:%M:%S",
    "Days:Hours": "%d:%H"
}


def parse_datetime(date, time):
    date_time_str = f"{date} {time}"

    try:
        date_time_obj = pd.to_datetime(date_time_str, format='%d.%m.%Y %H:%M:%S.%f', errors='raise')
    except ValueError:
        date_time_obj = pd.to_datetime(date_time_str, errors='coerce')

    return date_time_obj


@st.cache_data
def process_file(uploaded_file):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    encodings_to_try = [encoding, 'utf-8', 'latin1', 'ISO-8859-1']

    for enc in encodings_to_try:
        try:
            string_data = raw_data.decode(enc)
            fixed_text = fix_text(string_data)
            break
        except UnicodeDecodeError:
            continue
    else:
        fixed_text = raw_data.decode('utf-8', errors='replace')

    data = pd.read_csv(StringIO(fixed_text), low_memory=False)
    data.columns = [''.join([char for char in col if ord(char) < 128]) for col in data.columns]
    if 'Date' not in data.columns or 'Time' not in data.columns:
        data.columns = data.iloc[0]
        data = data.drop(0).reset_index(drop=True)

    if 'Date' in data.columns and 'Time' in data.columns:
        data['Datetime'] = data.apply(lambda row: parse_datetime(row['Date'], row['Time']), axis=1)
        data = data.drop(columns=['Date', 'Time'])
        if data['Datetime'].isnull().any():
            data = data.dropna(subset=['Datetime'])

    for col in data.columns:
        if col != 'Datetime':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.loc[:, ~data.columns.str.startswith('Unnamed:')]
    data = data.assign(Source=uploaded_file.name)

    return data, uploaded_file.name


def upload_files() -> tuple[dict[Any, Any], dict[Any, Any]]:
    uploaded_files = st.file_uploader("Choose CSV files, only HWiNFO64 is compatible for now.", type='csv',
                                      accept_multiple_files=True)
    dataframes = {}
    names = {}
    if uploaded_files:
        for uploaded_file in uploaded_files:
            data, name = process_file(uploaded_file)
            dataframes[name] = data
            custom_name = st.text_input(f"Provide a name for {name}:", value=name)
            names[name] = custom_name
            dataframes[name] = data
    else:
        pass
    return dataframes, names


def select_display_options(dataframes: Dict[str, pd.DataFrame]):
    if not dataframes:
        st.warning("No data available. Please upload data files.")
        st.stop()

    common_columns = set(next(iter(dataframes.values())).columns)
    for df in dataframes.values():
        common_columns.intersection_update(df.columns)

    excluded_fields = ['Time', 'Date']
    options = [col for col in common_columns
               if col.lower() not in excluded_fields
               and next(iter(dataframes.values()))[col].dtype in [float, int]]

    if not options:
        st.error("No valid data fields available for selection.")
        st.stop()

    selected_options = st.multiselect(
        'What data would you like to explore ?',
        options,
    )

    return selected_options


def render_comparator(unique_sources, means):
    with st.container():
        if len(unique_sources) < 2:
            return

        comparator_sources = st.multiselect(
            "Select sources for comparison:",
            unique_sources,
            default=unique_sources)

        if len(comparator_sources) != 2:
            st.warning("Please select exactly two sources for comparison.")
            return

        for pair in itertools.combinations(unique_sources, 2):
            mean_diff = abs(means[pair[0]] - means[pair[1]])
            avg_mean = (means[pair[0]] + means[pair[1]]) / 2
            relative_diff = (mean_diff / avg_mean) * 100
            st.write(f"Relative difference between {pair[0]} and {pair[1]}: {relative_diff:.2f}%")


def get_chart_preference():
    choice = st.radio("Choose chart display mode:", ["Separated", "Combined"])
    return choice


def render_separated_chart(option, dataframes, names):
    col_selector, _ = st.columns([1, 3])
    selected_time_format = col_selector.selectbox('Select Time Format', list(time_format_options.keys()), index=3,
                                                  key=f"time_format_selector_x_{option}"
                                                  )

    for name, data in dataframes.items():
        if option in data.columns:
            current_data = data[['Datetime', option]].copy()
            current_data = current_data.rename(columns={option: 'Value'})
            current_data['Datetime'] = pd.to_datetime(current_data['Datetime'], errors='coerce')
            current_data.dropna(subset=['Datetime'], inplace=True)
            formatted_datetime = current_data['Datetime'].dt.strftime(time_format_options[selected_time_format])
            current_data['Datetime'] = formatted_datetime
            col_chart, col_stats = st.columns(2, gap="small")
            col_chart.line_chart(current_data.set_index('Datetime'), y='Value', use_container_width=True)
            stats_list = list(calculate_statistics(current_data, 'Value'))
            _, min_val, max_val, avg_val, divergence, std_dev, convergence = stats_list[0]
            col_stats.subheader(f"Statistics for {option} @ ({name})", divider=True)
            col_stats.write(f"Minimum: {round(min_val, 2)}")
            col_stats.write(f"Maximum: {round(max_val, 2)}")
            col_stats.write(f"Mean: {round(avg_val, 2)}")
            col_stats.write(f"Convergence: {convergence:.2f}")
            col_stats.write(f"Divergence %: {divergence:.2f}%")
            col_stats.write(f"Standard Deviation: {std_dev:.2f}")


def render_combined_chart(option, combined_data):
    col_selector, _ = st.columns([1, 3])
    selected_time_format = col_selector.selectbox('Select Time Format', list(time_format_options.keys()), index=3,
                                                  key=f"time_format_selector_2_{option}"
)
    combined_data['Datetime'] = pd.to_datetime(combined_data['Datetime'])
    combined_data['FormattedDatetime'] = combined_data['Datetime'].dt.strftime(
        time_format_options[selected_time_format])
    pivoted_data = combined_data.pivot_table(index='FormattedDatetime', columns='Source', values='Value',
                                             aggfunc='mean')
    col1, col2 = st.columns(2, gap="small")
    unique_sources = combined_data['Source'].unique()
    if len(unique_sources) == 1:
        st.warning("Only one data source uploaded. A combined view requires multiple data sources.")
        return col2
    col1.line_chart(pivoted_data, use_container_width=True)
    return col2


def combine_data_for_option(option, dataframes, names):
    combined_data = None
    for name, data in dataframes.items():
        custom_name = names.get(name, name)
        if option in data.columns:
            current_data = data[['Datetime', option, 'Source']].copy()
            current_data['Source'] = custom_name
            current_data = current_data.rename(columns={option: 'Value'})
            if combined_data is None:
                combined_data = current_data
            else:
                combined_data = pd.concat([combined_data, current_data])
    return combined_data


def calculate_statistics(data, option):
    if 'Source' in data.columns:
        unique_sources = data['Source'].unique()
        for source in unique_sources:
            data_for_source = data[data['Source'] == source]
            min_val = data_for_source['Value'].min()
            max_val = data_for_source['Value'].max()
            avg_val = data_for_source['Value'].mean()
            divergence = ((max_val - min_val) / avg_val) * 100
            std_dev = data_for_source['Value'].std()
            convergence = np.mean(np.diff(data_for_source['Value']))
            yield source, min_val, max_val, avg_val, divergence, std_dev, convergence
    else:
        min_val = data['Value'].min()
        max_val = data['Value'].max()
        avg_val = data['Value'].mean()
        divergence = ((max_val - min_val) / avg_val) * 100
        std_dev = data['Value'].std()
        convergence = np.mean(np.diff(data['Value']))
        yield option, min_val, max_val, avg_val, divergence, std_dev, convergence


def render_charts(dataframes, selected_options, names):
    if not selected_options:
        st.write("No options selected for display.")
        return

    chart_preference = get_chart_preference()

    for option in selected_options:
        combined_data = combine_data_for_option(option, dataframes, names)

        col2 = None
        if chart_preference == "Combined":
            col2 = render_combined_chart(option, combined_data)
        else:
            render_separated_chart(option, dataframes, names)

        if col2:
            for source, min_val, max_val, avg_val, divergence, std_dev, convergence in calculate_statistics(
                    combined_data, option):
                col2.subheader(f"Statistics for {option} @ ({source})", divider=True)
                col2.write(f"Minimum: {round(min_val, 2)}")
                col2.write(f"Maximum: {max_val}")
                col2.write(f"Mean: {round(avg_val, 2)}")
                col2.write(f"Divergence %: {divergence:.2f}%")
                col2.write(f"Convergence: {convergence:.2f}")
                col2.write(f"Standard Deviation: {std_dev:.2f}")

        if len(combined_data['Source'].unique()) > 1:
            render_comparator(combined_data['Source'].unique(),
                              {src: avg for src, _, _, avg, _, _, _ in calculate_statistics(combined_data, option)})


def main():
    st.title("HWiNFO64 log analyzer.")
    dataframes, names = upload_files()
    selected_options = select_display_options(dataframes)
    render_charts(dataframes, selected_options, names)


if __name__ == "__main__":
    main()
