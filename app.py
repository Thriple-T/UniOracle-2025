import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="UniOracle 2025", layout="wide", page_icon="🎓")
@st.cache_data
def load_data():
    df = pd.read_csv("QS World University Rankings 2025 (Top global universities).csv", encoding='latin1')
    df.columns = [c.replace(' ', '_').replace('=', '') for c in df.columns]
    return df

@st.cache_resource
def load_ml_assets():
    scaler = joblib.load('scaler.joblib')
    rf_model = joblib.load('rank_predictor.joblib')
    features = joblib.load('features.joblib')
    return scaler, rf_model, features

try:
    df = load_data()
    scaler, rf_model, features = load_ml_assets()

    st.sidebar.header("🎯 Define Your Ideal University")
    st.sidebar.write("Adjust scores to see matches and tier predictions.")
    
    user_inputs = {}
    for col in features:
        label = col.replace('_', ' ').replace('Score', '')
        user_inputs[col] = st.sidebar.slider(label, 0, 100, 50)

    st.title("🎓 UniOracle 2025: AI University Insights")
    tab1, tab2, tab3 = st.tabs(["✨ Match Finder", "🧠 Tier AI", "📊 Global Analytics"])

    # KNN Engine
    with tab1:
        st.subheader("Your Top 3 University Matches")
    
        df_ml = df[features].fillna(0)
        X_scaled = scaler.transform(df_ml)
        knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
        knn.fit(X_scaled)
    
        user_vector = scaler.transform([list(user_inputs.values())])
        distances, indices = knn.kneighbors(user_vector)
        recs = df.iloc[indices[0]]

        cols = st.columns(3)
        for i, (idx, row) in enumerate(recs.iterrows()):
            with cols[i]:
                st.info(f"**Match #{i+1}**")
                st.metric("Global Rank", row['RANK_2025'])
                st.markdown(f"### {row['Institution_Name']}")
                st.caption(f"📍 {row['Location']}")
                
                # Radar Chart
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=list(user_inputs.values()), theta=features, fill='toself', name='You'))
                fig.add_trace(go.Scatterpolar(r=row[features].tolist(), theta=features, fill='toself', name='Uni'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), showlegend=False, height=250, margin=dict(l=20,r=20,t=20,b=20))
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("AI Tier Prediction")
        
        prediction = rf_model.predict(user_vector)
        probability = rf_model.predict_proba(user_vector)[0][1]

        c1, c2 = st.columns([1, 1])
        with c1:
            if prediction[0] == 1:
                st.success("### Prediction: TOP 100 TIER")
                st.balloons()
            else:
                st.warning("### Prediction: HIGH GLOBAL TIER")
            
            st.write(f"Confidence Level: **{probability:.1%}**")
            st.progress(probability)

        with c2:
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                             title="Why this prediction?", color_discrete_sequence=['#4B0082'])
            fig_imp.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

    with tab3:
        st.subheader("Explore the 2025 Dataset")
        st.dataframe(df[['RANK_2025', 'Institution_Name', 'Location'] + features].head(100), use_container_width=True)

except FileNotFoundError:
    st.error("Please ensure your CSV and .joblib files are in the same folder!")