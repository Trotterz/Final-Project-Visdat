import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import bokeh
from bokeh.plotting import figure
from bokeh.io import output_notebook
from bokeh.io import curdoc
from bokeh.models import HoverTool, ColumnDataSource, Span, Label
from streamlit_bokeh import streamlit_bokeh
import plotly.express as px

st.set_page_config(layout="wide", page_title="Nottingham Forest Dashboard")

# Formation
def get_vertical_coords(formation_str):
    lines = list(map(int, formation_str.split('-')))
    y_levels = [13 + i * (40 / (len(lines) - 1)) for i in range(len(lines))]

    coords = []
    for y, players_in_line in zip(y_levels, lines):
        spacing = 100 / (players_in_line + 1)
        for i in range(1, players_in_line + 1):
            x = -10 + i * spacing
            coords.append((x, y))
    return coords

def plot_vertical_formation(formation_str, color='dodgerblue'):
    pitch = VerticalPitch(pitch_color='#292a2b', pitch_type='statsbomb', line_color='white')
    fig, ax = pitch.draw(figsize=(3, 6))

    coords = get_vertical_coords(formation_str)
    for i, (x, y) in enumerate(coords, 1):
        ax.plot(x, y, 'o', color=color, markersize=8)
    st.pyplot(fig)

# Match Result Container
def display_match_result(result):
    """
    Displays a colored container in Streamlit based on the match result.

    Args:
        result (str): The result of the match ('Win', 'Loss', or 'Draw').
    """
    if result == 'W':
        container_html = """
            <div style="background-color: #90EE90; padding: 10px; border-radius: 5px;">
                <h3 style="color: black; text-align: center;">Won</h3>
            </div>
        """
        st.markdown(container_html, unsafe_allow_html=True)
    elif result == 'L':
        container_html = """
            <div style="background-color: #F08080; padding: 10px; border-radius: 5px;">
                <h3 style="color: black; text-align: center;">Lost</h3>
            </div>
        """
        st.markdown(container_html, unsafe_allow_html=True)
    elif result == 'D':
        container_html = """
            <div style="background-color: #D3D3D3; padding: 10px; border-radius: 5px;">
                <h3 style="color: black; text-align: center;">Draw</h3>
            </div>
        """
        st.markdown(container_html, unsafe_allow_html=True)

# Possession Container
def display_possession_bar(possession):
    """
    Displays a container that fills horizontally based on a percentage,
    like a progress bar for ball possession.

    Args:
        possession (int or float): The possession percentage (0 to 100).
    """
    # Ensure possession is within the 0-100 range
    possession = max(0, min(100, possession))

    # Define the colors
    fill_color = "#31B531"  # A nice light green
    bg_color = "#383838"   # A light grey for the empty part

    # The CSS for the linear gradient background
    # This creates a sharp line between the filled and non-filled parts
    background_style = (
        f"background: linear-gradient(to right, {fill_color} {possession}%, {bg_color} {possession}%);"
    )

    # The HTML for the container
    container_html = f'''
        <div style="{background_style} padding: 4px 10px; border-radius: 4px;">
            <h5 style="color: white; text-align: center; margin: 0;">
                {possession}%
            </h5>
        </div>
    '''
    st.markdown(container_html, unsafe_allow_html=True)

# Load DataFrames
df_gn_display = pd.read_pickle("/mount/src/final-project-visdat/df_gn.pkl")
df_sh_display = pd.read_pickle("/mount/src/final-project-visdat/df_sh.pkl")
df_ps_display = pd.read_pickle("/mount/src/final-project-visdat/df_ps.pkl")
df_pass_display = pd.read_pickle("/mount/src/final-project-visdat/df_pass.pkl")
df_dfd_display = pd.read_pickle("/mount/src/final-project-visdat/df_dfd.pkl")
df_gca_display = pd.read_pickle("/mount/src/final-project-visdat/df_gca.pkl")
df_shooting_display = pd.read_pickle("/mount/src/final-project-visdat/df_shooting.pkl")
merge_gcas_display = pd.read_pickle("/mount/src/final-project-visdat/merge_gcas.pkl")

rounds = df_gn_display.index.unique().tolist()


# Tabs for layout
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 GCA",
    "🎯 Shooting",
    "📈 Possession",
    "🎯 Passing",
    "🛡️ Defensive",
    "📌 GCA Details",
    "⚽ Players Shooting"
])

with tab1:
    st.title("GCA")

    st.subheader("Chance of Goal for Team vs Opponent")
    with st.container():
        x = df_gn_display.index.to_list()
        y = df_gn_display['xG'].to_list()
        y_opp = df_gn_display['xGA'].to_list()

        # Create a ColumnDataSource with all the necessary data
        source = ColumnDataSource(data=dict(
            x=x,
            y=y,
            y_opp=y_opp,
        ))

        # create a new plot with a title and axis labels
        p = figure(x_axis_label="Round", y_axis_label="Value", x_range=x, height=140, sizing_mode="stretch_width")

        # Add a HoverTool
        hover = HoverTool(
            tooltips=[
                ("Round", "@x"),
                ("Team Chances", "@y"),
                ("Opponent Chances", "@y_opp"),
            ],
            mode='mouse' # Add mode='mouse' for better interaction
        )
        p.add_tools(hover)

        # add multiple renderers
        p.line(x='x', y='y', source=source, legend_label="Team Chances", color="#0072B2", line_width=3)
        p.line(x='x', y='y_opp', source=source, legend_label="Opponent Chances", color="#E69F00", line_width=3)
        p.xaxis.major_label_orientation = 0.785

        # show the results
        streamlit_bokeh(p, theme='dark_minimal')

    st.markdown("---")

    st.subheader("Game-by-Game GCA")
    selected_round = st.selectbox("Select Match Round", options=rounds)
    df_gn_filtered = df_gn_display.loc[[selected_round]]

    # Formation
    col1, col2, col3 = st.columns([0.4, 0.3, 0.3], vertical_alignment='center')

    with col1:
        # Opponent
        with st.container():
            st.markdown("##### **Opponent**")
        with st.container(border=True):
            opponent = df_gn_filtered['Opponent'][0]
            opponent_html = f'''
                <div style="background-color: #0000FF00; padding: 10px; border-radius: 5px;">
                    <h5 style="color: white; text-align: center;">{opponent}</h5>
                </div>
            '''
            st.markdown(opponent_html, unsafe_allow_html=True)

        # Match Result
        with st.container():
            st.markdown("##### **Match Result**")
            match_result = df_gn_filtered['Result'][0]
            display_match_result(match_result)

        # Team & Opponent Goals
        with st.container():
            nested_col1, nested_col2 = st.columns(2, vertical_alignment="center")
            with nested_col1:
                with st.container():
                    st.markdown("##### **Team Goals (GF)**")
                with st.container(border=True):
                    gf_result = df_gn_filtered['GF'][0]
                    gf_html = f'''
                        <div style="background-color: #0000FF00; padding: 5px; border-radius: 5px;">
                            <h5 style="color: white; text-align: center;">{gf_result}</h5>
                        </div>
                    '''
                    st.markdown(gf_html, unsafe_allow_html=True)
            with nested_col2:
                with st.container():
                    st.markdown("##### **Opponent Goals (GA)**")
                with st.container(border=True):
                    ga_result = df_gn_filtered['GA'][0]
                    ga_html = f'''
                        <div style="background-color: #0000FF00; padding: 5px; border-radius: 5px;">
                            <h5 style="color: white; text-align: center;">{ga_result}</h5>
                        </div>
                    '''
                    st.markdown(ga_html, unsafe_allow_html=True)

        # Team & Opponent Goals Chance
        with st.container():
            nested_col3, nested_col4 = st.columns(2, vertical_alignment="center")
            with nested_col3:
                with st.container():
                    st.markdown("##### **Team Goals Chance (xG)**")
                with st.container(border=True):
                    xg_result = df_gn_filtered['xG'][0]
                    xg_html = f'''
                        <div style="background-color: #0000FF00; padding: 5px; border-radius: 5px;">
                            <h5 style="color: white; text-align: center;">{xg_result}</h5>
                        </div>
                    '''
                    st.markdown(xg_html, unsafe_allow_html=True)
            with nested_col4:
                with st.container():
                    st.markdown("##### **Opponent Goals Chance (xGA)**")
                with st.container(border=True):
                    xga_result = df_gn_filtered['xGA'][0]
                    xga_html = f'''
                        <div style="background-color: #0000FF00; padding: 5px; border-radius: 5px;">
                            <h5 style="color: white; text-align: center;">{xga_result}</h5>
                        </div>
                    '''
                    st.markdown(xga_html, unsafe_allow_html=True)

        # Team Possession
        with st.container():
            st.markdown("##### **Team Possession**")
            possession = df_gn_filtered['Poss'][0]
            display_possession_bar(possession)

    # Team Formation
    with col2:
        st.markdown("##### **Team Formation**")
        team_formation = df_gn_filtered['Formation'][0]
        plot_vertical_formation(team_formation, color='dodgerblue')

    # Opponent Formation
    with col3:
        st.markdown("##### **Opponent Formation**")
        opp_formation = df_gn_filtered['Opp Formation'][0]
        plot_vertical_formation(opp_formation, color='crimson')

with tab2:
    st.subheader("📊 Shooting Visualization")

    # Create a list of categories for coloring
    # This helps Plotly understand which bar to color differently.
    color_categories = ["Highlight" if club == "Nott'ham Forest" else "Other" for club in df_sh_display.index]

    with st.container():
        fig_xg = px.bar(
            df_sh_display,
            x=df_sh_display.index,
            y='xG',
            title='Goals Chance per Club',
            color=color_categories,
            color_discrete_map={
                "Highlight": "red",
                "Other": "grey"
            },
            labels={'x': 'Squad', 'xG': 'Expected Goals (xG)'}
        )

        fig_xg.update_layout(showlegend=False)
        fig_xg.update_xaxes(tickangle = -45)
        st.plotly_chart(fig_xg, use_container_width=True)


    # Calculate the 'Goals - xG' column
    df_sh_display['G_minus_xG'] = df_sh_display['Goals'] - df_sh_display['xG']

    with st.container():
        fig_gxg = px.bar(
            df_sh_display,
            x=df_sh_display.index,
            y='G_minus_xG',
            title='Goals Effectivity per Club',
            color=color_categories,
            color_discrete_map={
                "Highlight": "red",
                "Other": "grey"
            },
            labels={'x': 'Squad', 'G_minus_xG': 'Goals minus xG'}
        )

        # Remove the legend
        fig_gxg.update_layout(showlegend=False)
        fig_gxg.update_xaxes(tickangle = -45)
        # Display the figure
        st.plotly_chart(fig_gxg, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Shooting Data")
    st.dataframe(df_sh_display, use_container_width=True)

with tab3:
    st.subheader("📊 Possession Visualization")

    clubs = df_sh_display.index.tolist()

    # Scatter Plot Possession vs Tackles
    with st.container():

        # 1. Define color categories for the markers
        color_categories = ["Highlight" if club == "Nott'ham Forest" else "Other" for club in clubs]

        # 2. Create the base scatter plot without text
        fig_scatter = px.scatter(
            df_ps_display,
            x='Poss',
            y='Tackles',
            color=color_categories, # Use categories to color the markers
            color_discrete_map={
                "Highlight": "red",
                "Other": "grey"
            },
            title='Possession vs Tackles per Club',
            labels={'Poss': 'Possession (%)', 'Tackles': 'Tackles'}
        )
        fig_scatter.update_layout(showlegend=False) # Hide the legend for "Highlight" / "Other"

        # 3. Add annotations in a loop (just like the matplotlib version)
        for i, club in enumerate(clubs):
            fig_scatter.add_annotation(
                x=df_ps_display['Poss'].iloc[i],
                y=df_ps_display['Tackles'].iloc[i],
                text=club,
                showarrow=False,
                xshift=10, # Shift text slightly to the right of the point
                yshift=5,  # Shift text slightly above the point
                font=dict(
                    size=10,
                    color="red" if club == "Nott'ham Forest" else "white",
                ),
                font_weight="bold" if club == "Nott'ham Forest" else "normal" # Set font weight
            )

        st.plotly_chart(fig_scatter, use_container_width=True)
    
    col1, col2 = st.columns([0.5, 0.5], vertical_alignment='center')

    with col1:
        # Touches per Goal    
        with st.container():
            fig_tpg = px.bar(
                df_sh_display,
                x=df_sh_display.index,
                y='Touches per Goal',
                title='Touches per Goal',
                color=color_categories,
                color_discrete_map={
                    "Highlight": "red",
                    "Other": "grey"
                },
                labels={'x': 'Squad', 'Touches per Goal': 'Touches per Goal'}
            )

            fig_tpg.update_layout(showlegend=False)
            fig_tpg.update_xaxes(tickangle = -45)
            st.plotly_chart(fig_tpg, use_container_width=True)

    with col2:
        # Possession
        with st.container():
            fig_poss = px.bar(
                df_ps_display,
                x=df_sh_display.index,
                y='Poss',
                title='Possessions per Club',
                color=color_categories,
                color_discrete_map={
                    "Highlight": "red",
                    "Other": "grey"
                },
                labels={'x': 'Squad', 'Poss': 'Possession (%)'}
            )

            fig_poss.update_layout(showlegend=False)
            fig_poss.update_xaxes(tickangle = -45)
            st.plotly_chart(fig_poss, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Possession Data")
    st.dataframe(df_ps_display, use_container_width=True)

with tab4:
    st.subheader("📊 Passing Visualization")
    
    with st.container():
        fig_ppg = px.bar(
            df_sh_display,
            x=df_sh_display.index,
            y='Passes per Goal',
            title='Passes per Goal',
            color=color_categories,
            color_discrete_map={
                "Highlight": "red",
                "Other": "grey"
            },
            labels={'x': 'Squad', 'Passes per Goal': 'Passes per Goal'}
        )

        fig_ppg.update_layout(showlegend=False)
        fig_ppg.update_xaxes(tickangle = -45)
        st.plotly_chart(fig_ppg, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Passing Data")
    st.dataframe(df_pass_display, use_container_width=True)

with tab5:
    st.subheader("📅 Defensive Data (Squad)")

    squad_options = df_dfd_display.index.unique().tolist()
    selected_squad = st.selectbox("Select Squad", squad_options)

    # Tampilkan tabel untuk squad terpilih
    st.dataframe(df_dfd_display.loc[[selected_squad]], use_container_width=True)

    st.markdown("---")
    st.subheader("🔴 Defensive Visualizations (Pilih Sesuai Kebutuhan)")

    show_goals_against = st.checkbox("Tampilkan Goals Against")
    show_cleansheets = st.checkbox("Tampilkan Cleansheets")
    show_xga = st.checkbox("Tampilkan xGA")
    show_tackles_stack = st.checkbox("Tampilkan Tackles per Area (Stacked)")
    show_radar_def = st.checkbox("Tampilkan Radar Chart Pertahanan (Nottingham Forest)")

    # Goals Against
    if show_goals_against:
        st.markdown("### Goals Against (Seluruh Tim)")
        fig_def1, ax_def1 = plt.subplots(figsize=(10, 6))
        color_ga = ['red' if club == "Nott'ham Forest" else 'grey' for club in df_dfd_display.index]
        df_dfd_display['Goals Against'].plot(kind='bar', ax=ax_def1, color=color_ga, title='Goals Against Nottingham Forest')
        st.pyplot(fig_def1)

    # Cleansheets
    if show_cleansheets:
        st.markdown("### Cleansheets (Seluruh Tim)")
        fig_def2, ax_def2 = plt.subplots(figsize=(10, 6))
        color_cs = ['red' if club == "Nott'ham Forest" else 'grey' for club in df_dfd_display.index]
        df_dfd_display['Cleansheets'].plot(kind='bar', ax=ax_def2, color=color_cs, title='Cleansheets Nottingham Forest')
        st.pyplot(fig_def2)

    # xGA
    if show_xga:
        st.markdown("### xGA (Expected Goals Against)")
        fig_def3, ax_def3 = plt.subplots(figsize=(10, 6))
        color_xga = ['red' if club == "Nott'ham Forest" else 'grey' for club in df_dfd_display.index]
        df_dfd_display['xGA'].plot(kind='bar', ax=ax_def3, color=color_xga, title='xGA Nottingham Forest')
        st.pyplot(fig_def3)

    # Normalisasi metrik defensif (hanya dihitung jika butuh stacked chart)
    if show_tackles_stack:
        df_dfd_display['Tackles%'] = df_dfd_display['Tackles'] / df_dfd_display['Tackles'].mean()
        df_dfd_display['Tackles Won%'] = df_dfd_display['Tackles Won'] / df_dfd_display['Tackles Won'].mean()
        df_dfd_display['Goals Against%'] = df_dfd_display['Goals Against'] / df_dfd_display['Goals Against'].mean()
        df_dfd_display['Cleansheets%'] = df_dfd_display['Cleansheets'] / df_dfd_display['Cleansheets'].mean()
        df_dfd_display['xGA%'] = df_dfd_display['xGA'] / df_dfd_display['xGA'].mean()

        scaler = MinMaxScaler()
        df_dfd_display[['Tackles%', 'Tackles Won%', 'Goals Against%', 'Cleansheets%', 'xGA%']] = scaler.fit_transform(
            df_dfd_display[['Tackles%', 'Tackles Won%', 'Goals Against%', 'Cleansheets%', 'xGA%']]
        )

        # Stacked Bar Chart
        st.markdown("### Tackles per Area (Stacked Bar)")
        tackles_data = df_dfd_display[['Tackles Def 3rd', 'Tackles Mid 3rd', 'Tackles Att 3rd', 'Goals Against']].copy()
        tackles_data = tackles_data.sort_values(by='Goals Against', ascending=False)

        fig_def4, ax_def4 = plt.subplots(figsize=(10, 6))
        ax_def4.bar(tackles_data.index, tackles_data['Tackles Def 3rd'], label='Def 3rd')
        ax_def4.bar(tackles_data.index, tackles_data['Tackles Mid 3rd'],
                    bottom=tackles_data['Tackles Def 3rd'], label='Mid 3rd')
        ax_def4.bar(tackles_data.index, tackles_data['Tackles Att 3rd'],
                    bottom=tackles_data['Tackles Def 3rd'] + tackles_data['Tackles Mid 3rd'],
                    label='Att 3rd')

        ax_def4.set_title('Tackles per Area (Stacked)', fontsize=14, fontweight='bold')
        ax_def4.set_ylabel('Jumlah Tackles')
        ax_def4.set_xlabel('Klub')
        ax_def4.tick_params(axis='x', rotation=45)
        ax_def4.legend()
        st.pyplot(fig_def4)
        
    if show_radar_def:
        # Radar chart untuk performa defensif Nottingham Forest
        df_dfd_display['Tackles%'] = df_dfd_display['Tackles'] / df_dfd_display['Tackles'].mean()
        df_dfd_display['Tackles Won%'] = df_dfd_display['Tackles Won'] / df_dfd_display['Tackles Won'].mean()
        df_dfd_display['Goals Against%'] = df_dfd_display['Goals Against'] / df_dfd_display['Goals Against'].mean()
        df_dfd_display['Cleansheets%'] = df_dfd_display['Cleansheets'] / df_dfd_display['Cleansheets'].mean()
        df_dfd_display['xGA%'] = df_dfd_display['xGA'] / df_dfd_display['xGA'].mean()

        to_scale = ['Tackles%', 'Tackles Won%', 'Goals Against%', 'Cleansheets%', 'xGA%']
        scaler = MinMaxScaler()
        df_dfd_display[to_scale] = scaler.fit_transform(df_dfd_display[to_scale])

        club_name = "Nott'ham Forest"
        club_values = df_dfd_display.loc[club_name, to_scale].tolist()
        club_values_full = club_values + [club_values[0]]

        percentiles = []
        for col in to_scale:
            val = df_dfd_display.loc[club_name, col]
            if col in ['Goals Against%', 'xGA%']:
                pct = (df_dfd_display[col] > val).mean()
            else:
                pct = (df_dfd_display[col] < val).mean()
            percentiles.append(int(pct * 100))

        percentiles_full = percentiles + [percentiles[0]]

        labels = ['Tackles', 'Tackles Won', 'Goals Against', 'Cleansheets', 'xGA']
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles_full = angles + [angles[0]]

        fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(labels, fontsize=11, fontweight='medium')
        ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_radar.set_ylim(0, 1.2)

        ax_radar.plot(angles_full, club_values_full, color='royalblue', linewidth=2)
        ax_radar.fill(angles_full, club_values_full, color='royalblue', alpha=0.25)

        for angle, value, pct in zip(angles_full, club_values_full, percentiles_full):
            y_pos = value + 0.06
            y_pos = min(y_pos, 1.15)
            ax_radar.text(angle, y_pos, f"{pct}%", ha='center', va='center', fontsize=10, fontweight='bold', color='black')

        st.pyplot(fig_radar)


with tab6:
    st.subheader("📅 Detailed GCA Data")
    st.dataframe(df_gca_display, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Additional GCA Visualization")

    show_gca_chart = st.checkbox("Tampilkan Line Chart GCA Nottingham Forest")

    if show_gca_chart:
        st.markdown("### GCA Nottingham Forest (Line Chart)")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax = merge_gcas_display['GCA'].plot(
            kind='line',
            ax=ax,
            title='GCA Nottingham Forest',
            color=['#0072B2', '#E69F00']
        )

        change_pos = df_gn_display.index.get_loc('23/24_19')
        ax.axvline(x=change_pos, color='#D55E00', linestyle='--', linewidth=2, label='Perubahan Pelatih Kepala')
        ax.axvspan(0, change_pos, color='#cfe2f3', alpha=0.3, label='Sebelum')
        ax.axvspan(change_pos, len(df_gn_display), color='#fce5cd', alpha=0.3, label='Setelah')

        ax.set_xlabel('Round')
        ax.set_ylabel('Value')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Ringkasan total GF-GA
    st.markdown("---")
    st.subheader("📈 Ringkasan Total GF-GA per Musim")

    total_gfga_2223 = df_gn_display.loc['22/23_1':'22/23_38', 'GF-GA'].sum()
    total_gfga_2324 = df_gn_display.loc['23/24_1':'23/24_38', 'GF-GA'].sum()
    total_gfga_2425 = df_gn_display.loc['24/25_1':'24/25_38', 'GF-GA'].sum()

    st.markdown(f"""
    - **2022/23:** {total_gfga_2223}
    - **2023/24:** {total_gfga_2324}
    - **2024/25:** {total_gfga_2425}
    """)
    
    # Menyimpan total GF-GA ke dataframe baru
    df_gfga = pd.DataFrame(
        {'Total GF-GA': [total_gfga_2223, total_gfga_2324, total_gfga_2425]},
        index=['22/23', '23/24', '24/25']
    )

    # Plot tren GF-GA
    st.markdown("### Grafik Total GF-GA Nottingham Forest tiap Musim")
    fig_gfga, ax_gfga = plt.subplots(figsize=(10, 6))
    df_gfga.plot(kind='bar', ax=ax_gfga, legend=False, title='Total GF-GA Nottingham Forest tiap Musim')
    ax_gfga.axhline(y=0, color='r', linestyle='--')
    ax_gfga.grid(axis='y')
    plt.tight_layout()
    st.pyplot(fig_gfga)

with tab7:
    st.subheader("📋 Player Shooting Data")
    st.dataframe(df_shooting_display, use_container_width=True)

# Sidebar
sidecol1, sidecol2 = st.sidebar.columns([0.2, 0.8], vertical_alignment='center')
with sidecol1:
    st.image('https://upload.wikimedia.org/wikipedia/en/thumb/e/e5/Nottingham_Forest_F.C._logo.svg/800px-Nottingham_Forest_F.C._logo.svg.png', use_container_width=True)

with sidecol2:
    st.markdown('# Nottingham Forest Dashboard')

st.sidebar.markdown(
    '''
    ---
    Berikut adalah hasil pertandingan Premier League selama musim 2022 sampai 2025.
    Data diperoleh dari [fbref.com](https://fbref.com/) dengan fokus kepada klub bola Nottingham Forest.
    '''
)

with st.sidebar.expander("# **Info**"):
    st.markdown('''
        1. **90s**: Banyaknya pertandingan yang dimainkan tiap tim
        2. **Goals**: Banyaknya gol yang diciptakan tiap tim
        3. **Shots**: Banyaknya tendangan yang diciptakan tiap tim
        4. **Shot on Target**: Banyaknya tendangan mengarah ke gawang yang diciptakan tiap tim
        5. **Shot on Target (%)**: Perbandingan Shot on Target dengan banyaknya Shots dalam persentase
        6. **Shot/90**: Banyaknya tendangan yang diciptakan tiap tim per pertandingan
        7. **Shot on Target/90**: Banyaknya tendangan mengarah ke gawang yang diciptakan tiap tim per pertandingan
        8. **Goals/Shot**: Banyaknya tendangan yang dibutuhkan untuk menciptakan satu gol
        9. **Goals/Shot on Target**: Banyaknya tendangan mengarah ke gawang yang dibutuhkan untuk menciptakan satu gol
        10. **xG**: Banyaknya probabilitas tendangan menjadi gol yang diciptakan oleh tiap tim
        11. **npxG**: Banyaknya probabilitas tendangan menjadi gol yang diciptakan oleh tiap tim tanpa penalti
        12. **npxG/Shot**: Banyaknya probabilitas tendangan menjadi gol yang diciptakan oleh tiap tim per tendangan dan tanpa penalti
        13. **Goals-xG**: Mengecek apakah suatu tim efektif dalam menyelesaikan peluang yang mereka dapatkan dari banyaknya gol yang mereka ciptakan dikurangi nilai probabilitas suatu tendangan menjadi gol
        14. **npxG-npG**: Mengecek apakah suatu tim efektif dalam menyelesaikan peluang yang mereka dapatkan dari banyaknya gol yang mereka ciptakan dikurangi nilai probabilitas suatu tendangan menjadi gol (tanpa menghitung penalti)
        15. **Touches per Goal**: Berapa banyak sentuhan terhadap bola yang dibutuhkan oleh suatu tim untuk menciptakan gol
        16. **Passes per Goal**: Berapa banyak operan yang dibutuhkan oleh suatu tim untuk menciptakan gol
    ''')
