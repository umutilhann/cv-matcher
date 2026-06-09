import streamlit as st
import pandas as pd
from matcher import find_matches, load_and_prepare_data

st.set_page_config(
    page_title="CV Matcher — AI Job Recommender",
    page_icon="🎯",
    layout="centered"
)

# Başlık
st.title("🎯 CV & Job Matcher")
st.markdown("CV metnini yapıştır, yapay zeka sana en uygun iş ilanlarını bulsun.")
st.divider()

# Veriyi önceden yükle (cache sayesinde bir kez çalışır)
with st.spinner("İş ilanları yükleniyor..."):
    df, _, _, _ = load_and_prepare_data()

st.caption(f"Veritabanında **{len(df)} iş ilanı** mevcut.")

# CV girişi
cv_input = st.text_area(
    "CV metnini buraya yapıştır:",
    height=250,
    placeholder="Skills, experience, education, tools you use..."
)

col1, col2 = st.columns([2, 1])
with col1:
    top_n = st.slider("Kaç sonuç gösterilsin?", min_value=3, max_value=10, value=5)
with col2:
    st.write("")
    st.write("")
    run = st.button("🔍 Eşleştir", type="primary", use_container_width=True)

st.divider()

# Eşleştirme
if run:
    if not cv_input.strip():
        st.warning("Lütfen CV metnini gir.")
    else:
        with st.spinner("Analiz ediliyor..."):
            results = find_matches(cv_input, top_n=top_n)

        st.success(f"En iyi {top_n} eşleşme bulundu!")

        # --- Bar Chart ---
        st.subheader("📊 Uyum Skorları")
        chart_data = pd.DataFrame({
            "Pozisyon": [r["title"][:35] + ("…" if len(r["title"]) > 35 else "") for r in results],
            "Uyum (%)": [r["score"] for r in results],
        })
        st.bar_chart(chart_data.set_index("Pozisyon"))

        st.divider()

        # --- Detay Kartları ---
        st.subheader("📋 Detaylı Sonuçlar")
        for i, r in enumerate(results):
            with st.expander(f"#{i+1}  {r['title']}  —  %{r['score']} uyum"):

                # Skor progress bar
                st.progress(r["score"] / 100)

                # Eşleşen beceriler
                if r["matched_skills"]:
                    skills_str = "  ".join([f"`{s}`" for s in r["matched_skills"]])
                    st.markdown(f"**Eşleşen beceriler:** {skills_str}")
                else:
                    st.markdown("**Eşleşen beceriler:** —")

                # İlan özeti
                st.markdown("**İlan özeti:**")
                st.caption(r["description"])