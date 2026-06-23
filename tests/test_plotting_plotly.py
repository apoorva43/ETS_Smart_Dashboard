"""
Unit tests for src/plotting_plotly.py

These tests validate chart outputs (figure type, trace count, axis config)
without rendering to a browser. No Streamlit context required.
"""
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.plotting_plotly import (
    _parse_color,
    _base_layout,
    _check_sufficient_data,
    plot_country_shaded_density,
    plot_group_shaded_density,
    plot_percentile_change_from_baseline,
    plot_resource_scatter,
    plot_intersectional_heatmap,
)


# helpers
class TestParseColor:
    def test_hex_color(self):
        # Parses six-digit hex to correct RGB tuple
        assert _parse_color("#0072B2") == (0, 114, 178)

    def test_rgb_string(self):
        # Parses rgb() string to integer tuple
        assert _parse_color("rgb(0, 114, 178)") == (0, 114, 178)

    def test_rgba_string(self):
        # Parses rgba() string ignoring the alpha channel
        r, g, b = _parse_color("rgba(0, 114, 178, 0.5)")
        assert (r, g, b) == (0, 114, 178)

    def test_fallback_for_unknown_format(self):
        # Unknown format returns gray fallback
        assert _parse_color("unknown") == (128, 128, 128)


class TestBaseLayout:
    def test_returns_dict(self):
        # _base_layout returns a plain dict
        assert isinstance(_base_layout(), dict)

    def test_custom_title(self):
        # Title is passed through correctly
        layout = _base_layout(title="My Chart")
        assert layout["title"] == "My Chart"

    def test_custom_height(self):
        # Height override is respected
        layout = _base_layout(height=600)
        assert layout["height"] == 600


class TestCheckSufficientData:
    def test_returns_valid_data_and_none_fig(self, base_df):
        # Returns (filtered_df, None) when data is sufficient
        sub = base_df[base_df["CNT"] == "CAN"]
        valid, fig = _check_sufficient_data(sub, ["PV1MATH"], "CAN", min_n=10)
        assert valid is not None
        assert fig is None

    def test_returns_none_data_and_fig(self, tiny_df):
        # Returns (None, Figure) when data is insufficient
        valid, fig = _check_sufficient_data(tiny_df, ["PV1MATH"], "TST", min_n=100)
        assert valid is None
        assert isinstance(fig, go.Figure)


# plot_country_shaded_density
class TestPlotCountryShapedDensity:
    def test_returns_figure(self, base_df):
        # Returns a Plotly Figure for valid data
        fig = plot_country_shaded_density(base_df, "MATH", ["CAN"], year=2022)
        assert isinstance(fig, go.Figure)

    def test_returns_figure_for_multiple_countries(self, base_df):
        # Works when two countries are passed
        fig = plot_country_shaded_density(base_df, "MATH", ["CAN", "USA"], year=2022)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_insufficient_data_returns_annotated_figure(self, tiny_df):
        # Returns an annotated warning figure rather than crashing
        fig = plot_country_shaded_density(tiny_df, "MATH", ["TST"], year=2022)
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0

    def test_oecd_row_added_when_oecd_data_present(self, base_df):
        # OECD row trace appears in figure when OECD countries are in df
        fig = plot_country_shaded_density(base_df, "MATH", ["CAN"], year=2022)
        trace_names = [t.name for t in fig.data if t.name]
        assert any("OECD" in name for name in trace_names)

    def test_each_country_has_legend_entry(self, base_df):
        # Each selected country produces at least one visible legend entry
        fig = plot_country_shaded_density(base_df, "MATH", ["CAN", "USA"], year=2022)
        legend_names = {t.name for t in fig.data if t.showlegend}
        assert "Canada" in legend_names or "CAN" in legend_names
        assert "United States" in legend_names or "USA" in legend_names

    def test_x_axis_range_set(self, base_df):
        # X axis range is pinned to 100–900
        fig = plot_country_shaded_density(base_df, "MATH", ["CAN"], year=2022)
        assert fig.layout.xaxis.range == [100, 900]

    def test_year_filter_applied(self, base_df):
        # Chart for 2022 and 2015 produce different figures (different data)
        fig_22 = plot_country_shaded_density(base_df, "MATH", ["CAN"], year=2022)
        fig_15 = plot_country_shaded_density(base_df, "MATH", ["CAN"], year=2015)
        # Different years → different median positions → different trace x values
        xs_22 = [t.x for t in fig_22.data if t.x is not None]
        xs_15 = [t.x for t in fig_15.data if t.x is not None]
        assert xs_22 != xs_15


# plot_group_shaded_density
class TestPlotGroupShapedDensity:
    def test_returns_figure_for_gender(self, base_df):
        # Produces a valid figure for gender grouping
        from src.config import GENDER_MAP
        fig = plot_group_shaded_density(
            base_df, "MATH", "CAN", "ST004D01T", GENDER_MAP, "Gender", year=2022
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_returns_figure_for_immigration(self, base_df):
        # Produces a valid figure for immigration status grouping
        from src.config import IMMIG_MAP
        fig = plot_group_shaded_density(
            base_df, "MATH", "CAN", "IMMIG", IMMIG_MAP, "Immigration status", year=2022
        )
        assert isinstance(fig, go.Figure)

    def test_ses_binning_when_labels_none(self, base_df):
        # SES (continuous) variable is binned into quartiles without error
        fig = plot_group_shaded_density(
            base_df, "MATH", "CAN", "ESCS", None, "Socioeconomic status", year=2022
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_percentage_appears_in_y_tick_labels(self, base_df):
        # Y-axis tick labels include percentage share of each group
        from src.config import GENDER_MAP
        fig = plot_group_shaded_density(
            base_df, "MATH", "CAN", "ST004D01T", GENDER_MAP, "Gender", year=2022
        )
        ticktext = list(fig.layout.yaxis.ticktext or [])
        assert any("%" in t for t in ticktext)

    def test_insufficient_data_returns_warning_figure(self, tiny_df):
        # Returns annotated warning figure when data is too sparse
        from src.config import GENDER_MAP
        fig = plot_group_shaded_density(
            tiny_df, "MATH", "TST", "ST004D01T", GENDER_MAP, "Gender"
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0


# plot_percentile_change_from_baseline
class TestPlotPercentileChange:
    def test_returns_figure(self, base_df):
        # Returns a Plotly Figure with multi-year data
        fig = plot_percentile_change_from_baseline(base_df, "MATH", "CAN", reference_year=2015)
        assert isinstance(fig, go.Figure)

    def test_has_five_percentile_traces(self, base_df):
        # One trace per percentile (10, 25, 50, 75, 90)
        fig = plot_percentile_change_from_baseline(base_df, "MATH", "CAN", reference_year=2015)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 5

    def test_baseline_year_delta_is_zero(self, base_df):
        # Delta at the reference year should be 0 for all percentiles
        fig = plot_percentile_change_from_baseline(base_df, "MATH", "CAN", reference_year=2015)
        for trace in fig.data:
            if isinstance(trace, go.Scatter) and trace.x is not None:
                for x_val, y_val in zip(trace.x, trace.y):
                    if x_val == 2015:
                        assert abs(y_val) < 1e-6

    def test_missing_baseline_returns_warning_figure(self, base_df):
        # Returns warning figure when reference year has no data
        sub = base_df[base_df["YEAR"] != 2015]
        fig = plot_percentile_change_from_baseline(sub, "MATH", "CAN", reference_year=2015)
        assert isinstance(fig, go.Figure)

    def test_title_contains_country_and_subject(self, base_df):
        # Chart title includes country name and subject
        fig = plot_percentile_change_from_baseline(base_df, "MATH", "CAN", reference_year=2015)
        title = fig.layout.title.text
        assert "Mathematics" in title or "MATH" in title


# plot_resource_scatter
class TestPlotResourceScatter:
    def test_returns_figure(self, base_df):
        # Returns a Plotly Figure for a valid resource column
        fig = plot_resource_scatter(base_df, "MATH", "ESCS", "SES Index", year=2022)
        assert isinstance(fig, go.Figure)

    def test_highlighted_countries_get_separate_trace(self, base_df):
        # Highlighted countries appear as a distinct trace
        fig = plot_resource_scatter(
            base_df, "MATH", "ESCS", "SES Index",
            year=2022, highlight_countries=["CAN"]
        )
        trace_names = [t.name for t in fig.data]
        assert "Selected" in trace_names

    def test_quadrant_annotations_present(self, base_df):
        # Four quadrant text annotations are added to the layout
        fig = plot_resource_scatter(base_df, "MATH", "ESCS", "SES Index", year=2022)
        assert len(fig.layout.annotations) >= 4

    def test_empty_df_returns_warning_figure(self):
        # Empty DataFrame returns a warning figure, not an exception
        fig = plot_resource_scatter(pd.DataFrame(), "MATH", "ESCS", "SES Index")
        assert isinstance(fig, go.Figure)

    def test_partner_and_oecd_traces_present(self, base_df):
        # OECD members and partner countries each get their own trace
        fig = plot_resource_scatter(base_df, "MATH", "ESCS", "SES Index", year=2022)
        trace_names = [t.name for t in fig.data]
        assert "OECD members" in trace_names
        assert "Partner countries" in trace_names


# plot_intersectional_heatmap
class TestPlotIntersectionalHeatmap:
    def test_returns_figure(self, base_df):
        # Returns a Plotly Figure for SES x Belonging
        fig = plot_intersectional_heatmap(
            base_df, "MATH", "CAN", row_var="ESCS", col_var="BELONG", year=2022
        )
        assert isinstance(fig, go.Figure)

    def test_heatmap_trace_present(self, base_df):
        # Figure contains a Heatmap trace
        fig = plot_intersectional_heatmap(
            base_df, "MATH", "CAN", row_var="ESCS", col_var="BELONG", year=2022
        )
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_z_values_in_score_range(self, base_df):
        # All non-NaN heatmap cells contain valid PISA scores
        fig = plot_intersectional_heatmap(
            base_df, "MATH", "CAN", row_var="ESCS", col_var="BELONG", year=2022
        )
        z = np.array(fig.data[0].z, dtype=float)
        valid = z[~np.isnan(z)]
        assert (valid >= 100).all() and (valid <= 900).all()

    def test_gender_as_categorical_variable(self, base_df):
        # Works correctly when col_var is a categorical like gender
        fig = plot_intersectional_heatmap(
            base_df, "MATH", "CAN",
            row_var="ESCS", col_var="ST004D01T",
            col_label="Gender", year=2022
        )
        assert isinstance(fig, go.Figure)

    def test_insufficient_data_returns_warning(self, tiny_df):
        # Returns annotated warning figure when too few cells can be filled
        fig = plot_intersectional_heatmap(
            tiny_df, "MATH", "TST", row_var="ESCS", col_var="BELONG"
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0
