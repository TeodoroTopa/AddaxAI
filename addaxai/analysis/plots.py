"""Plotting and visualization utilities for AddaxAI.

Chart helpers, figure conversion, logo overlay, and the produce_plots()
orchestrator extracted from app.py.
"""

import calendar
import datetime
import io
import os
from pathlib import Path
from typing import Any, Callable, Tuple

from PIL import Image

# Optional heavy deps — imported here so they can be mocked in unit tests.
# Each group is wrapped in try/except so the module can be imported without
# the plotting libraries installed (unit test environment).
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None  # type: ignore[assignment]
    sns = None  # type: ignore[assignment]

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]

try:
    import folium
    from folium.plugins import HeatMap, Draw, MarkerCluster
except ImportError:
    folium = None  # type: ignore[assignment]
    HeatMap = None  # type: ignore[assignment]
    Draw = None  # type: ignore[assignment]
    MarkerCluster = None  # type: ignore[assignment]

from addaxai.core.events import event_bus
from addaxai.core.event_types import POSTPROCESS_PROGRESS


def fig2img(fig: Any) -> Image.Image:
    """Convert a matplotlib figure to a PIL Image via in-memory buffer.

    Args:
        fig: A matplotlib figure or pyplot module.

    Returns:
        PIL.Image.Image
    """
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


def overlay_logo(image_path: str, logo: Image.Image) -> None:
    """Paste a logo image onto the top-right corner of a chart image.

    Args:
        image_path: Path to the chart image (modified in place).
        logo: PIL.Image.Image of the logo to overlay.
    """
    main_image = Image.open(image_path)
    main_width, main_height = main_image.size
    logo_width, logo_height = logo.size
    position = (main_width - logo_width - 10, 10)
    main_image.paste(logo, position, logo)
    main_image.save(image_path)


def calculate_time_span(df: Any) -> Tuple[int, int, int, int]:
    """Analyze the date range in a detection DataFrame.

    Args:
        df: pandas DataFrame with a 'DateTimeOriginal' datetime column.

    Returns:
        Tuple of (years, months, weeks, days) as integers.
        Returns (0, 0, 0, 0) if no dates are present.
    """
    any_dates_present = df['DateTimeOriginal'].notnull().any()
    if not any_dates_present:
        return 0, 0, 0, 0
    first_date = df['DateTimeOriginal'].min()
    last_date = df['DateTimeOriginal'].max()
    time_difference = last_date - first_date
    days = time_difference.days
    years = int(days / 365)
    months = int(days / 30)
    weeks = int(days / 7)
    return years, months, weeks, days


# ---------------------------------------------------------------------------
# Private helpers called by produce_plots() — extracted to module level so
# they can be mocked in unit tests without needing the plotting libraries.
# ---------------------------------------------------------------------------

def _create_time_plots(data: Any, save_path_base: str, temporal_units: Any,
                       update_pbar: Callable, counts_df: Any) -> None:
    """Create temporal bar charts and heatmaps for each time unit."""

    max_n_ticks = 50

    time_format_mapping = {
        "year": {'freq': 'Y', 'time_format': '%Y', 'dir': "grouped-by-year"},
        "month": {'freq': 'M', 'time_format': '%Y-%m', 'dir': "grouped-by-month"},
        "week": {'freq': 'W', 'time_format': '%Y-wk %U', 'dir': "grouped-by-week"},
        "day": {'freq': 'D', 'time_format': '%Y-%m-%d', 'dir': "grouped-by-day"}
    }

    grouped_data = data.groupby(['label', pd.Grouper(key='DateTimeOriginal', freq='1D')]).size().unstack(fill_value=0)

    def plot_obs_over_time_total_static(time_unit):
        plt.figure(figsize=(10, 6))
        combined_data = grouped_data.sum(axis=0).resample(time_format_mapping[time_unit]['freq']).sum()
        plt.bar(combined_data.index.strftime(time_format_mapping[time_unit]['time_format']), combined_data, width=0.9)
        plt.suptitle("")
        plt.title(f'Total observations (grouped per {time_unit}, n = {counts_df["count"].sum()})')
        plt.ylabel('Count')
        plt.xlabel(time_unit)
        plt.xticks(rotation=90)
        x_vals = np.arange(len(combined_data))
        tick_step = max(len(combined_data) // max_n_ticks, 1)
        selected_ticks = x_vals[::tick_step]
        while_iteration = 0
        while len(selected_ticks) >= max_n_ticks:
            tick_step += 1
            while_iteration += 1
            selected_ticks = x_vals[::tick_step]
            if while_iteration > 100:
                break
        selected_labels = combined_data.index.strftime(time_format_mapping[time_unit]['time_format'])[::tick_step]
        plt.xticks(selected_ticks, selected_labels)
        plt.tight_layout()
        save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "combined-single-layer.png")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        update_pbar()

    def plot_obs_over_time_total_interactive(time_unit):
        combined_data = grouped_data.sum(axis=0).resample(time_format_mapping[time_unit]['freq']).sum()
        hover_text = [f'Period: {date}<br>Count: {count}<extra></extra>'
                    for date, count in zip(combined_data.index.strftime(time_format_mapping[time_unit]['time_format']),
                                            combined_data)]
        fig = go.Figure(data=[go.Bar(x=combined_data.index.strftime(time_format_mapping[time_unit]['time_format']),
                                    y=combined_data,
                                    hovertext=hover_text,
                                    hoverinfo='text')])
        fig.update_traces(hovertemplate='%{hovertext}')
        fig.update_layout(title=f'Total observations (grouped per {time_unit})',
                        xaxis_title='Period',
                        yaxis_title='Count',
                        xaxis_tickangle=90)
        save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "combined-single-layer.html")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        update_pbar()

    def plot_obs_over_time_combined_static(time_unit):
        plt.figure(figsize=(10, 6))
        for label in grouped_data.index:
            grouped_data_indexed = grouped_data.loc[label].resample(time_format_mapping[time_unit]['freq']).sum()
            plt.plot(grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format']), grouped_data_indexed, label=label)
        plt.suptitle("")
        plt.title(f'Observations over time (grouped per {time_unit}, n = {counts_df["count"].sum()})')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.xlabel(time_unit)
        plt.legend(loc='upper right')
        x_vals = np.arange(len(grouped_data_indexed))
        tick_step = max(len(grouped_data_indexed) // max_n_ticks, 1)
        selected_ticks = x_vals[::tick_step]
        while_iteration = 0
        while len(selected_ticks) >= max_n_ticks:
            tick_step += 1
            while_iteration += 1
            selected_ticks = x_vals[::tick_step]
            if while_iteration > 100:
                break
        selected_labels = grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format'])[::tick_step]
        plt.xticks(selected_ticks, selected_labels)
        plt.tight_layout()
        save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "combined-multi-layer.png")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        update_pbar()

    def plot_obs_over_time_combined_interactive(time_unit):
        fig = go.Figure()
        for label in grouped_data.index:
            grouped_data_indexed = grouped_data.loc[label].resample(time_format_mapping[time_unit]['freq']).sum()
            fig.add_trace(go.Scatter(x=grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format']),
                                    y=grouped_data_indexed,
                                    mode='lines',
                                    name=label))
        fig.update_layout(title=f'Observations over time (grouped per {time_unit})',
                        xaxis_title='Period',
                        yaxis_title='Count',
                        xaxis_tickangle=90,
                        legend=dict(x=0, y=1.0))
        fig.update_layout(hovermode="x unified")
        save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "combined-multi-layer.html")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        update_pbar()

    def plot_obs_over_time_separate_static(label, time_unit):
        plt.figure(figsize=(10, 6))
        grouped_data_indexed = grouped_data.loc[label].resample(time_format_mapping[time_unit]['freq']).sum()
        plt.bar(grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format']), grouped_data_indexed, label=label, width=0.9)
        plt.suptitle("")
        plt.title(f'Observations over time for {label} (grouped per {time_unit}, n = {counts_df[counts_df["label"] == label]["count"].values[0]})')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.xlabel(time_unit)
        x_vals = np.arange(len(grouped_data_indexed))
        tick_step = max(len(grouped_data_indexed) // max_n_ticks, 1)
        selected_ticks = x_vals[::tick_step]
        while_iteration = 0
        while len(selected_ticks) >= max_n_ticks:
            tick_step += 1
            while_iteration += 1
            selected_ticks = x_vals[::tick_step]
            if while_iteration > 100:
                break
        selected_labels = grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format'])[::tick_step]
        plt.xticks(selected_ticks, selected_labels)
        plt.tight_layout()
        save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "class-specific", f"{label}.png")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        update_pbar()

    def plot_obs_over_time_separate_interactive(label, time_unit):
        grouped_data_indexed = grouped_data.loc[label].resample(time_format_mapping[time_unit]['freq']).sum()
        hover_text = [f'Period: {date}<br>Count: {count}<extra></extra>'
                    for date, count in zip(grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format']),
                                            grouped_data_indexed)]
        fig = go.Figure(go.Bar(x=grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format']),
                                y=grouped_data_indexed,
                                hovertext=hover_text,
                                hoverinfo='text'))
        fig.update_traces(hovertemplate='%{hovertext}')
        fig.update_layout(title=f'Observations over time for {label} (grouped per {time_unit})',
                        xaxis_title='Period',
                        yaxis_title='Count',
                        xaxis_tickangle=90)
        save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "class-specific", f"{label}.html")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        update_pbar()

    def plot_obs_over_time_heatmap_static_absolute(time_unit):
        data['Period'] = data['DateTimeOriginal'].dt.strftime(time_format_mapping[time_unit]['time_format'])
        time_range = pd.Series(pd.date_range(data['DateTimeOriginal'].min(), data['DateTimeOriginal'].max(), freq=time_format_mapping[time_unit]['freq']))
        df_time = pd.DataFrame({time_unit: time_range.dt.strftime(time_format_mapping[time_unit]['time_format'])})
        heatmap_data = data.groupby(['Period', 'label']).size().unstack(fill_value=0)
        merged_data = pd.merge(df_time, heatmap_data, left_on=time_unit, right_index=True, how='left').fillna(0)
        merged_data.set_index(time_unit, inplace=True)
        merged_data = merged_data.sort_index()
        plt.figure(figsize=(14, 8))
        ax = sns.heatmap(merged_data, cmap="Blues")
        sorted_labels = sorted(merged_data.columns)
        ax.set_xticks([i + 0.5 for i in range(len(sorted_labels))])
        ax.set_xticklabels(sorted_labels)
        plt.title(f'Temporal heatmap (absolute values, grouped per {time_unit}, n = {counts_df["count"].sum()})')
        plt.tight_layout()
        legend_text = 'Number of observations'
        ax.collections[0].colorbar.set_label(legend_text)
        save_path = os.path.join(save_path_base, "graphs", "temporal-heatmaps", time_format_mapping[time_unit]['dir'], "absolute", "temporal-heatmap.png")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        update_pbar()

    def plot_obs_over_time_heatmap_static_relative(time_unit):
        data['Period'] = data['DateTimeOriginal'].dt.strftime(time_format_mapping[time_unit]['time_format'])
        time_range = pd.Series(pd.date_range(data['DateTimeOriginal'].min(), data['DateTimeOriginal'].max(), freq=time_format_mapping[time_unit]['freq']))
        df_time = pd.DataFrame({time_unit: time_range.dt.strftime(time_format_mapping[time_unit]['time_format'])})
        heatmap_data = data.groupby(['Period', 'label']).size().unstack(fill_value=0)
        normalized_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
        merged_data = pd.merge(df_time, normalized_data, left_on=time_unit, right_index=True, how='left').fillna(0)
        merged_data.set_index(time_unit, inplace=True)
        merged_data = merged_data.sort_index()
        plt.figure(figsize=(14, 8))
        ax = sns.heatmap(merged_data, cmap="Blues")
        sorted_labels = sorted(normalized_data.columns)
        ax.set_xticks([i + 0.5 for i in range(len(sorted_labels))])
        ax.set_xticklabels(sorted_labels)
        plt.title(f'Temporal heatmap (relative values, grouped per {time_unit}, n = {counts_df["count"].sum()})')
        plt.tight_layout()
        legend_text = 'Number of observations normalized per label'
        ax.collections[0].colorbar.set_label(legend_text)
        save_path = os.path.join(save_path_base, "graphs", "temporal-heatmaps", time_format_mapping[time_unit]['dir'], "relative", "temporal-heatmap.png")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        update_pbar()

    def plot_obs_over_time_heatmap_interactive_absolute(time_unit):
        data['Period'] = data['DateTimeOriginal'].dt.strftime(time_format_mapping[time_unit]['time_format'])
        time_range = pd.Series(pd.date_range(data['DateTimeOriginal'].min(), data['DateTimeOriginal'].max(), freq=time_format_mapping[time_unit]['freq']))
        df_time = pd.DataFrame({time_unit: time_range.dt.strftime(time_format_mapping[time_unit]['time_format'])})
        heatmap_data = data.groupby(['Period', 'label']).size().unstack(fill_value=0)
        merged_data = pd.merge(df_time, heatmap_data, left_on=time_unit, right_index=True, how='left').fillna(0)
        merged_data.set_index(time_unit, inplace=True)
        heatmap_trace = go.Heatmap(z=merged_data.values,
                                x=merged_data.columns,
                                y=merged_data.index,
                                customdata=merged_data.stack().reset_index().values.tolist(),
                                colorscale='Blues',
                                hovertemplate='Class: %{x}<br>Period: %{y}<br>Count: %{z}<extra></extra>',
                                colorbar=dict(title='Number of<br>observations'))
        fig = go.Figure(data=heatmap_trace)
        fig.update_layout(title=f'Temporal heatmap (absolute values, grouped per {time_unit}, n = {counts_df["count"].sum()})',
                        xaxis_title='Label',
                        yaxis_title='Period',
                        yaxis={'autorange': 'reversed'})
        save_path = os.path.join(save_path_base, "graphs", "temporal-heatmaps", time_format_mapping[time_unit]['dir'], "absolute", "temporal-heatmap.html")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        update_pbar()

    def plot_obs_over_time_heatmap_interactive_relative(time_unit):
        data['Period'] = data['DateTimeOriginal'].dt.strftime(time_format_mapping[time_unit]['time_format'])
        time_range = pd.date_range(data['DateTimeOriginal'].min(), data['DateTimeOriginal'].max(), freq=time_format_mapping[time_unit]['freq'])
        df_time = pd.DataFrame({time_unit: time_range.strftime(time_format_mapping[time_unit]['time_format'])})
        heatmap_data = data.groupby(['Period', 'label']).size().unstack(fill_value=0)
        merged_data = pd.merge(df_time, heatmap_data, left_on=time_unit, right_index=True, how='left').fillna(0)
        merged_data.set_index(time_unit, inplace=True)
        normalized_data = merged_data.div(merged_data.sum(axis=0), axis=1)
        heatmap_trace = go.Heatmap(
            z=normalized_data.values,
            x=normalized_data.columns,
            y=normalized_data.index,
            customdata=normalized_data.stack().reset_index().values.tolist(),
            colorscale='Blues',
            hovertemplate='Class: %{x}<br>Period: %{y}<br>Normalized count: %{z}<extra></extra>',
            colorbar=dict(title='Number of<br>observations<br>normalized<br>per label'))
        fig = go.Figure(data=heatmap_trace)
        fig.update_layout(
            title=f'Temporal heatmap (relative values, grouped per {time_unit}, n = {counts_df["count"].sum()}))',
            xaxis_title='Label',
            yaxis_title='Period',
            yaxis={'autorange': 'reversed'})
        save_path = os.path.join(save_path_base, "graphs", "temporal-heatmaps", time_format_mapping[time_unit]['dir'], "relative", "temporal-heatmap.html")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        update_pbar()

    for time_unit in temporal_units:
        plot_obs_over_time_total_static(time_unit);plt.close('all')
        plot_obs_over_time_total_interactive(time_unit);plt.close('all')
        plot_obs_over_time_combined_static(time_unit);plt.close('all')
        plot_obs_over_time_combined_interactive(time_unit);plt.close('all')
        plot_obs_over_time_heatmap_static_absolute(time_unit);plt.close('all')
        plot_obs_over_time_heatmap_static_relative(time_unit);plt.close('all')
        plot_obs_over_time_heatmap_interactive_absolute(time_unit);plt.close('all')
        plot_obs_over_time_heatmap_interactive_relative(time_unit);plt.close('all')
        for label in grouped_data.index:
            plot_obs_over_time_separate_static(label, time_unit);plt.close('all')
            plot_obs_over_time_separate_interactive(label, time_unit);plt.close('all')


def _create_activity_patterns(df: Any, save_path_base: str,
                               update_pbar: Callable) -> None:
    """Create hourly and monthly activity pattern plots."""
    df['DateTimeOriginal'] = pd.to_datetime(df['DateTimeOriginal'])
    grouped_data = df.groupby(['label', pd.Grouper(key='DateTimeOriginal', freq='1D')]).size().unstack(fill_value=0)
    df['Hour'] = df['DateTimeOriginal'].dt.hour
    hourly_df = df.groupby(['label', 'Hour']).size().reset_index(name='count')
    df['Month'] = df['DateTimeOriginal'].dt.month
    monthly_df = df.groupby(['label', 'Month']).size().reset_index(name='count')

    def plot_static_activity_pattern(df, unit, label=''):
        if label != '':
            df = df[df['label'] == label]
        total_observations = df['count'].sum()
        plt.figure(figsize=(10, 6))
        if unit == "Hour":
            x_ticks = range(24)
            x_tick_labels = [f'{x:02}-{(x + 1) % 24:02}' for x in x_ticks]
        else:
            x_ticks = range(1, 13)
            x_tick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.bar(df[unit], df['count'], width=0.9, align='center')
        plt.xlabel(unit)
        plt.ylabel('Number of observations')
        plt.title(f'Activity pattern of {label if label != "" else "all animals combined"} by {"hour of the day" if unit == "Hour" else "month of the year"} (n = {total_observations})')
        plt.xticks(x_ticks, x_tick_labels, rotation=90)
        plt.tight_layout()
        if label != '':
            save_path = os.path.join(save_path_base, "graphs", "activity-patterns", "hour-of-day" if unit == "Hour" else "month-of-year", "class-specific", f"{label}.png")
        else:
            save_path = os.path.join(save_path_base, "graphs", "activity-patterns", "hour-of-day" if unit == "Hour" else "month-of-year", "combined.png")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        update_pbar()

    def plot_dynamic_activity_pattern(df, unit, label=''):
        if label != '':
            df = df[df['label'] == label]
        n_ticks = 24 if unit == "Hour" else 12
        if unit == "Hour":
            x_ticks = list(range(24))
            x_tick_labels = [f'{x:02}-{(x + 1) % 24:02}' for x in x_ticks]
        else:
            x_ticks = list(range(12))
            x_tick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            df.loc[:, 'Month'] = df['Month'].map({i: calendar.month_abbr[i] for i in range(1, 13)})
        df = df.groupby(unit, as_index=False)['count'].sum()
        if unit == "Month":
            all_months = pd.DataFrame({
                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            })
            merged_df = all_months.merge(df, on='Month', how='left')
            merged_df['count'] = merged_df['count'].fillna(0)
            merged_df['count'] = merged_df['count'].astype(int)
            df = merged_df
        else:
            df = df.set_index(unit).reindex(range(n_ticks), fill_value=0).reset_index()
        total_observations = df['count'].sum()
        fig = px.bar(df, x=unit, y='count', title=f'Activity pattern of {label if label != "" else "all animals combined"} by {"hour of the day" if unit == "Hour" else "month of the year"} (n = {total_observations})').update_traces(width=0.7)
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=x_ticks,
                ticktext=x_tick_labels
            ),
            xaxis_title=unit,
            yaxis_title='Count',
            bargap=0.1
        )
        if label != '':
            save_path = os.path.join(save_path_base, "graphs", "activity-patterns", "hour-of-day" if unit == "Hour" else "month-of-year", "class-specific", f"{label}.html")
        else:
            save_path = os.path.join(save_path_base, "graphs", "activity-patterns", "hour-of-day" if unit == "Hour" else "month-of-year", "combined.html")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        update_pbar()

    for label in grouped_data.index:
        plot_static_activity_pattern(hourly_df, "Hour", label);plt.close('all')
        plot_static_activity_pattern(monthly_df, "Month", label);plt.close('all')
        plot_dynamic_activity_pattern(hourly_df, "Hour", label);plt.close('all')
        plot_dynamic_activity_pattern(monthly_df, "Month", label);plt.close('all')

    plot_static_activity_pattern(hourly_df, "Hour", "");plt.close('all')
    plot_static_activity_pattern(monthly_df, "Month", "");plt.close('all')
    plot_dynamic_activity_pattern(hourly_df, "Hour", "");plt.close('all')
    plot_dynamic_activity_pattern(monthly_df, "Month", "");plt.close('all')


def _create_geo_plots(data: Any, save_path_base: str, update_pbar: Callable) -> None:
    """Create folium heatmaps and marker cluster maps."""

    def create_combined_multi_layer_clustermap(data, save_path_base):
        if len(data) == 0:
            return
        map_path = os.path.join(save_path_base, "graphs", "maps")
        unique_labels = data['label'].unique()
        checkboxes = {label: folium.plugins.MarkerCluster(name=label) for label in unique_labels}
        for label in unique_labels:
            label_data = data[data['label'] == label]
            max_lat, min_lat = label_data['Latitude'].max(), label_data['Latitude'].min()
            max_lon, min_lon = label_data['Longitude'].max(), label_data['Longitude'].min()
            center_lat, center_lon = label_data['Latitude'].mean(), label_data['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            m.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])
            for _, row in label_data.iterrows():
                folium.Marker(location=[row['Latitude'], row['Longitude']]).add_to(checkboxes[label])
            folium.TileLayer('openstreetmap').add_to(m)
            folium.LayerControl().add_to(m)
            Draw(export=True).add_to(m)
        max_lat, min_lat = data['Latitude'].max(), data['Latitude'].min()
        max_lon, min_lon = data['Longitude'].max(), data['Longitude'].min()
        center_lat, center_lon = data['Latitude'].mean(), data['Longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        m.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])
        for label, marker_cluster in checkboxes.items():
            marker_cluster.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        Draw(export=True).add_to(m)
        combined_multi_layer_file = os.path.join(map_path, "combined-multi-layer.html")
        Path(os.path.dirname(combined_multi_layer_file)).mkdir(parents=True, exist_ok=True)
        m.save(combined_multi_layer_file)
        update_pbar()

    def create_obs_over_geo_both_heat_and_mark(data, save_path_base, category=''):
        if category != '':
            data = data[data['label'] == category]
        data = data.dropna(subset=['Latitude', 'Longitude'])
        if len(data) == 0:
            return
        map_path = os.path.join(save_path_base, "graphs", "maps")
        max_lat, min_lat = data['Latitude'].max(), data['Latitude'].min()
        max_lon, min_lon = data['Longitude'].max(), data['Longitude'].min()
        center_lat, center_lon = data['Latitude'].mean(), data['Longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        m.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])
        folium.TileLayer('OpenStreetMap', overlay=False).add_to(m)
        Draw(export=True).add_to(m)
        heatmap_layer = folium.FeatureGroup(name='Heatmap', show=True, overlay=True).add_to(m)
        cluster_layer = MarkerCluster(name='Markers', show=False, overlay=True).add_to(m)
        HeatMap(data[['Latitude', 'Longitude']]).add_to(heatmap_layer)
        for _, row in data.iterrows():
            folium.Marker(location=[row['Latitude'], row['Longitude']]).add_to(cluster_layer)
        folium.LayerControl(collapsed=False).add_to(m)
        if category != '':
            map_file = os.path.join(map_path, "class-specific", f"{category}.html")
        else:
            map_file = os.path.join(map_path, 'combined-single-layer.html')
        Path(os.path.dirname(map_file)).mkdir(parents=True, exist_ok=True)
        m.save(map_file)
        update_pbar()

    create_obs_over_geo_both_heat_and_mark(data, save_path_base);plt.close('all')
    create_combined_multi_layer_clustermap(data, save_path_base);plt.close('all')
    for label in data['label'].unique():
        create_obs_over_geo_both_heat_and_mark(data, save_path_base, label);plt.close('all')


def _create_pie_plots_detections(df: Any, results_dir: str,
                                  update_pbar: Callable) -> None:
    """Create pie charts for the distribution of detections."""

    def create_pie_chart_detections_static():
        label_counts = df['label'].value_counts()
        total_count = len(df['label'])
        percentages = label_counts / total_count * 100
        hidden_categories = list(percentages[percentages < 0].index)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        wedges, _, autotexts = ax1.pie(label_counts, autopct='', startangle=140)
        ax1.axis('equal')
        for i, autotext in enumerate(autotexts):
            if label_counts.index[i] in hidden_categories:
                autotext.set_visible(False)
        legend_labels = ['%s (n = %s, %.1f%%)' % (label, count, (float(count) / len(df['label'])) * 100) for label, count in zip(label_counts.index, label_counts)]
        ax2.legend(wedges, legend_labels, loc="center", fontsize='medium')
        ax2.axis('off')
        for autotext in autotexts:
            autotext.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'))
        fig.suptitle(f"Distribution of detections (n = {total_count})", fontsize=16, y=0.95)
        plt.subplots_adjust(wspace=0.1)
        plt.tight_layout()
        save_path = os.path.join(results_dir, "graphs", "pie-charts", "distribution-detections.png")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        update_pbar()

    def create_pie_chart_detections_interactive():
        grouped_df = df.groupby('label').size().reset_index(name='count')
        total_count = grouped_df['count'].sum()
        grouped_df['percentage'] = (grouped_df['count'] / total_count) * 100
        grouped_df['percentage'] = grouped_df['percentage'].round(2).astype(str) + '%'
        fig = px.pie(grouped_df, names='label', values='count', title=f"Distribution of detections (n = {total_count})", hover_data={'percentage'})
        fig.update_traces(textinfo='label')
        save_path = os.path.join(results_dir, "graphs", "pie-charts", "distribution-detections.html")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        update_pbar()

    create_pie_chart_detections_static();plt.close('all')
    create_pie_chart_detections_interactive();plt.close('all')


def _create_pie_plots_files(df: Any, results_dir: str,
                             update_pbar: Callable) -> None:
    """Create pie charts for the distribution of files (detection vs empty)."""

    def create_pie_chart_files_static():
        df['label'] = df['n_detections'].apply(lambda x: 'detection' if x >= 1 else 'empty')
        label_counts = df['label'].value_counts()
        total_count = len(df['label'])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        def autopct_func(pct):
            if pct > 0:
                return f'{pct:.1f}%'
            else:
                return ''
        labels = [label for label in label_counts.index]
        wedges, texts, autotexts = ax1.pie(label_counts, labels=labels, autopct=autopct_func, startangle=140)
        ax1.axis('equal')
        legend_labels = ['%s (n = %s, %.1f%%)' % (label, count, (float(count) / len(df['label'])) * 100) for label, count in zip(label_counts.index, label_counts)]
        ax2.legend(wedges, legend_labels, loc="center", fontsize='medium')
        ax2.axis('off')
        for autotext in autotexts:
            autotext.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'))
        fig.suptitle(f"Distribution of files (n = {total_count})", fontsize=16, y=0.95)
        plt.subplots_adjust(wspace=0.5)
        save_path = os.path.join(results_dir, "graphs", "pie-charts", "distribution-files.png")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        update_pbar()

    def create_pie_chart_files_interactive():
        df['label'] = df['n_detections'].apply(lambda x: 'detection' if x >= 1 else 'empty')
        grouped_df = df.groupby('label').size().reset_index(name='count')
        total_count = grouped_df['count'].sum()
        grouped_df['percentage'] = (grouped_df['count'] / total_count) * 100
        grouped_df['percentage'] = grouped_df['percentage'].round(2).astype(str) + '%'
        fig = px.pie(grouped_df, names='label', values='count', title=f"Distribution of files (n = {total_count})", hover_data={'percentage'})
        fig.update_traces(textinfo='label')
        save_path = os.path.join(results_dir, "graphs", "pie-charts", "distribution-files.html")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        update_pbar()

    create_pie_chart_files_static();plt.close('all')
    create_pie_chart_files_interactive();plt.close('all')


def produce_plots(
    results_dir,           # type: str
    cancel_check,          # type: Callable[[], bool]
    cancel_func,           # type: Callable[[], None]
    logo_image,            # type: Any
    logo_width,            # type: int
    logo_height,           # type: int
):
    # type: (...) -> None
    """Generate all postprocessing plots for a results directory.

    Reads results_detections.csv and results_files.csv, creates temporal,
    geographic, pie chart, and activity pattern plots under results_dir/graphs/,
    then overlays the AddaxAI logo on every PNG.

    Args:
        results_dir: Path to the results directory containing CSV files.
        cancel_check: Callable that returns True if the user has cancelled.
        cancel_func: Callable that performs the cancel action (passed to events).
        logo_image: PIL Image of the logo to overlay on each chart.
        logo_width: Original logo width in pixels (used to compute overlay size).
        logo_height: Original logo height in pixels (used to compute overlay size).
    """
    results_dir = os.path.normpath(results_dir)
    plots_dir = os.path.join(results_dir, "graphs")
    det_df = pd.read_csv(os.path.join(results_dir, "results_detections.csv"))
    fil_df = pd.read_csv(os.path.join(results_dir, "results_files.csv"))

    det_df['DateTimeOriginal'] = pd.to_datetime(det_df['DateTimeOriginal'], format='%d/%m/%y %H:%M:%S')
    n_years, n_months, n_weeks, n_days = calculate_time_span(det_df)

    temporal_units = []
    max_units = 100
    if n_years > 1:
        temporal_units.append("year")
    if 1 < n_months <= max_units:
        temporal_units.append("month")
    if 1 < n_weeks <= max_units:
        temporal_units.append("week")
    if 1 < n_days <= max_units:
        temporal_units.append("day")

    det_df_geo = det_df[(det_df['Latitude'].notnull()) & (det_df['Longitude'].notnull())]
    if len(det_df_geo) > 0:
        data_permits_map_creation = True
        n_categories_geo = len(det_df_geo['label'].unique())
    else:
        data_permits_map_creation = False
        n_categories_geo = 0

    any_dates_present = det_df['DateTimeOriginal'].notnull().any()
    n_categories_with_timestamps = len(det_df[det_df['DateTimeOriginal'].notnull()]['label'].unique())
    n_obs_per_label_with_timestamps = det_df[det_df['DateTimeOriginal'].notnull()].groupby('label').size().reset_index(name='count')
    activity_patterns_n_plots = (((n_categories_with_timestamps * 2) + 2) * 2) if any_dates_present else 0
    bar_charts_n_plots = (((n_categories_with_timestamps * 2) + 4) * len(temporal_units)) if any_dates_present else 0
    maps_n_plots = (n_categories_geo + 2) if data_permits_map_creation else 0
    pie_charts_n_plots = 4
    temporal_heatmaps_n_plots = (4 * len(temporal_units)) if any_dates_present else 0
    n_plots = (activity_patterns_n_plots + bar_charts_n_plots + maps_n_plots + pie_charts_n_plots + temporal_heatmaps_n_plots)

    def update_pbar_plt():
        pbar.update(1)
        tqdm_stats = pbar.format_dict
        event_bus.emit(POSTPROCESS_PROGRESS,
                       process="plt", status="running",
                       cur_it=tqdm_stats['n'],
                       tot_it=tqdm_stats['total'],
                       time_ela=str(datetime.timedelta(seconds=round(tqdm_stats['elapsed']))),
                       time_rem=str(datetime.timedelta(seconds=round((tqdm_stats['total'] - tqdm_stats['n']) / tqdm_stats['n'] * tqdm_stats['elapsed'] if tqdm_stats['n'] else 0))),
                       cancel_func=cancel_func)

    def _close_all():
        if plt:
            plt.close('all')

    with tqdm(total=n_plots, disable=False) as pbar:
        event_bus.emit(POSTPROCESS_PROGRESS, process="plt", status="load")
        if any_dates_present:
            _create_time_plots(det_df, results_dir, temporal_units, update_pbar_plt, n_obs_per_label_with_timestamps);_close_all()
        if cancel_check():
            return
        if data_permits_map_creation:
            _create_geo_plots(det_df_geo, results_dir, update_pbar_plt);_close_all()
        if cancel_check():
            return
        _create_pie_plots_detections(det_df, results_dir, update_pbar_plt);_close_all()
        if cancel_check():
            return
        _create_pie_plots_files(fil_df, results_dir, update_pbar_plt);_close_all()
        if cancel_check():
            return
        if any_dates_present:
            _create_activity_patterns(det_df, results_dir, update_pbar_plt);_close_all()
        if cancel_check():
            return

    logo_for_graphs = logo_image.resize((int(logo_width / 1.2), int(logo_height / 1.2)))
    for root, dirs, files in os.walk(plots_dir):
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                overlay_logo(image_path, logo_for_graphs)

    event_bus.emit(POSTPROCESS_PROGRESS, process="plt", status="done")
