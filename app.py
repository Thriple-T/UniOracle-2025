import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="UniOracle 2025", layout="wide", page_icon="🎓")

ABBREVIATIONS = {
    'Academic_Reputation_Score': 'Acad Rep',
    'Employer_Reputation_Score': 'Emp Rep',
    'Faculty_Student_Score': 'Fac/Stu',
    'Citations_per_Faculty_Score': 'Cites',
    'International_Faculty_Score': 'Intl Fac',
    'International_Students_Score': 'Intl Stu',
    'International_Research_Network_Score': 'Intl Net',
    'Employment_Outcomes_Score': 'Jobs Outcome',
    'Sustainability_Score': 'Sustain'
}
FEATURES = list(ABBREVIATIONS.keys())

@st.cache_data
def load_data():
    df = pd.read_csv("QS World University Rankings 2025 (Top global universities).csv", encoding='latin1')
    df.columns = [c.strip().replace(' ', '_').replace('=', '') for c in df.columns]

    def find_col(keyword):
        match = [c for c in df.columns if keyword.lower() in c.lower()]
        return match[0] if match else None

    loc_col = find_col('Location') or 'Location'
    reg_col = find_col('Region') or 'Region'
    size_col = find_col('Size') or 'Size'

    df = df.rename(columns={loc_col: 'Location', reg_col: 'Region', size_col: 'Size'})

    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    return df

try:
    df_raw = load_data()
    st.sidebar.header("Filter Options")

    available_regions = sorted(df_raw['Region'].dropna().astype(str).unique())
    selected_regions = st.sidebar.multiselect("Region", available_regions)

    if selected_regions:
        filtered_by_region = df_raw[df_raw['Region'].isin(selected_regions)]
        available_locations = sorted(filtered_by_region['Location'].dropna().unique())
    else:
        available_locations = sorted(df_raw['Location'].dropna().unique())
        
    selected_locations = st.sidebar.multiselect("Country / Location", available_locations)

    available_sizes = sorted(df_raw['Size'].dropna().astype(str).unique())
    selected_size = st.sidebar.multiselect("Institution Size", available_sizes)

    st.sidebar.divider()
    st.sidebar.header("Your Priorities")
    user_inputs = {}
    for col in FEATURES:
        user_inputs[col] = st.sidebar.slider(ABBREVIATIONS[col], 0, 100, 50)

    st.title("🎓 UniOracle 2025")
    st.markdown("### Find your perfect university match.")
    st.divider()

    df_filtered = df_raw.copy()

    if selected_regions:
        df_filtered = df_filtered[df_filtered['Region'].isin(selected_regions)]
    
    if selected_locations:
        df_filtered = df_filtered[df_filtered['Location'].isin(selected_locations)]
        
    if selected_size:
        df_filtered = df_filtered[df_filtered['Size'].isin(selected_size)]

    if df_filtered.empty:
        st.error("❌ No universities match these filters. Try removing the 'Size' or 'Country' filter.")
    
    else:
        st.info(f"🔍 Analyzing **{len(df_filtered)}** universities that match your criteria...")

        scaler = StandardScaler()
        X_filtered = scaler.fit_transform(df_filtered[FEATURES])
        n_neighbors = min(5, len(df_filtered))
        
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X_filtered)
        
        user_vector = scaler.transform([list(user_inputs.values())])
        distances, indices = knn.kneighbors(user_vector)
        recommendations = df_filtered.iloc[indices[0]]

        tab1, tab2 = st.tabs(["✨ Top Matches", "📊 Distribution Analysis"])
        
        with tab1:
            for i, (idx, row) in enumerate(recommendations.iterrows()):
                with st.container():
                    c1, c2 = st.columns([2, 3])
                    with c1:
                        rank_display = int(row['RANK_2025']) if str(row['RANK_2025']).isdigit() else row['RANK_2025']
                        st.subheader(f"#{i+1}: {row['Institution_Name']}")
                        st.write(f"📍 **{row['Location']}** ({row['Region']})")
                        st.write(f"🏫 Size: **{row['Size']}** | 🏆 Rank: **{rank_display}**")
                        
                        top_metric = row[FEATURES].idxmax()
                        st.success(f"🔥 Strongest Stat: **{ABBREVIATIONS[top_metric]}** ({row[top_metric]})")

                    with c2:
                        short_labels = [ABBREVIATIONS[f] for f in FEATURES]
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(r=list(user_inputs.values()), theta=short_labels, fill='toself', name='You', line_color='#FF4B4B'))
                        fig.add_trace(go.Scatterpolar(r=row[FEATURES].tolist(), theta=short_labels, fill='toself', name='Uni', line_color='#1F77B4'))
                        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8))), height=220, margin=dict(l=30, r=30, t=10, b=10), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    st.divider()

        with tab2:
            st.subheader("Where are these universities located?")
            
            loc_counts = df_filtered['Location'].value_counts().reset_index()
            loc_counts.columns = ['Location', 'Count']
            
            fig_bar = px.bar(
                loc_counts, 
                x='Location', 
                y='Count', 
                color='Count',
                title="Matches by Country",
                labels={'Count': 'Number of Universities'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_pie = px.pie(
                df_filtered, 
                names='Region', 
                title="Matches by Region",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

except Exception as e:
    st.error(f"Something went wrong! Error: {e}")
    st.write("Debug info - Columns found:", df_raw.columns.tolist() if 'df_raw' in locals() else "Data not loaded")