# percentile score profile
def plot_country_distributions(df, subject: str,
                               countries: list,
                               year: int = None,
                               show_oecd: bool = True,
                               primary_country: str = None,
                               active_countries = None) -> go.Figure:
    fig = go.Figure()

    for i, cnt in enumerate(countries):
        subset = df[df["CNT"] == cnt]
        if year is not None and "YEAR" in df.columns:
            subset = subset[subset["YEAR"] == year]
            
        percs = weighted_percentiles_pv(subset, subject, PERCENTILES_COARSE)
        if np.isnan(percs).all():
            continue
            
        color = _country_color(cnt, active_countries if active_countries is not None else countries)
        
        line_width = 3 if cnt == primary_country else 2
        marker_size = 10 if cnt == primary_country else 8
        
        fig.add_trace(go.Scatter(
            x=PERCENTILES_COARSE, y=percs,
            mode="lines+markers", 
            name=_cnt_label(cnt),
            line=dict(color=color, width=line_width),
            marker=dict(symbol=SYMBOLS_COARSE, size=marker_size, line=dict(color="white", width=1)),
            hovertemplate=f"<b>{_cnt_label(cnt)}</b><br>{SUBJECTS[subject]}: %{{y:.0f}}<extra></extra>"
        ))

    if show_oecd:
        oecd = get_oecd_percentiles(df, subject, PERCENTILES_COARSE, year)
        if not np.isnan(oecd).all():
            fig.add_trace(go.Scatter(
                x=PERCENTILES_COARSE, y=oecd,
                mode="lines+markers", name="OECD avg",
                line=dict(color="#777777", width=2, dash="dash"),
                marker=dict(symbol=SYMBOLS_COARSE, size=9),
                hovertemplate=f"<b>OECD avg</b><br>{SUBJECTS[subject]}: %{{y:.0f}}<extra></extra>"
            ))

    symbol_note = "Percentile symbols: ▼ 10th   ■ 25th   ◆ 50th   ● 75th   ▲ 90th"
    fig.update_layout(**_base_layout(title=f"Percentile Score Profile | {SUBJECTS[subject]}<br><sup>{symbol_note}</sup>"))
    fig.update_xaxes(title="Percentile", tickvals=PERCENTILES_COARSE)
    fig.update_yaxes(title=f"{SUBJECTS[subject]} score")
    return fig

# group - ses
def plot_escs_gap(df, subject, cnt, year=None):
    subset = df[df["CNT"] == cnt].copy()
    if year is not None:
        subset = subset[subset["YEAR"] == year]
        
    valid_data, error_fig = _check_sufficient_data(
        subset, ["ESCS"], cnt, msg=f"Insufficient ESCS data for {cnt}"
    )
    if error_fig is not None:
        return error_fig

    curves = compute_escs_quartile_percentiles(valid_data, subject, PERCENTILES_COARSE, cnt=cnt, year=year)
    fig = go.Figure()

    for color, (label, percs) in zip(PALETTE, curves.items()):
        if np.isnan(percs).all():
            continue
        fig.add_trace(go.Scatter(
            x=PERCENTILES_COARSE, y=percs,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2.5),
            marker=dict(symbol=SYMBOLS_COARSE, size=9, line=dict(color="white", width=1)),
            hovertemplate=f"<b>{label}</b><br>Percentile: %{{x}}<br>{SUBJECTS[subject]}: %{{y:.0f}}<extra></extra>"
        ))

    symbol_note = "Percentile symbols: ▼ 10th   ■ 25th   ◆ 50th   ● 75th   ▲ 90th"
    fig.update_layout(**_base_layout(title=f"Score by Socioeconomic Status | {SUBJECTS[subject]} | {_cnt_label(cnt)}<br><sup>{symbol_note}</sup>"))
    fig.update_xaxes(title="Percentile", tickvals=PERCENTILES_COARSE)
    fig.update_yaxes(title=f"{SUBJECTS[subject]} score")
    return fig