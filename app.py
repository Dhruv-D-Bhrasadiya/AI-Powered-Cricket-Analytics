import streamlit as st
import os
from pathlib import Path
import cover_drive_analysis_realtime 

# Streamlit page settings
st.set_page_config(page_title="AthleteRise - Cricket Analytics", layout="wide")

st.title("🏏 AthleteRise - AI-Powered Cricket Analytics")
st.markdown("Upload your cricket video and configuration file to get an AI-based performance analysis.")

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# File uploads
video_file = st.file_uploader("📂 Upload Cricket Video (MP4)", type=["mp4"])
config_file = st.file_uploader("📂 Upload Config JSON", type=["json"])

# Run analysis button
if st.button("🚀 Run Analysis"):
    if video_file and config_file:
        with st.spinner("Processing... Please wait, this may take a few minutes."):

            # Save uploaded video as temp.mp4
            with open("temp.mp4", "wb") as f:
                f.write(video_file.read())

            # Save uploaded config as config.json
            with open("config.json", "wb") as f:
                f.write(config_file.read())

            try:
                cover_drive_analysis_realtime.main()
                st.success("✅ Analysis complete!")
            except Exception as e:
                st.error(f"❌ Error while running analysis: {e}")
                st.stop()

        # Display annotated video with download option
        output_video_path = "output/annotated_video.mp4"
        if os.path.exists(output_video_path):
            st.subheader("🎥 Annotated Video")
            with open(output_video_path, "rb") as video_file_bin:
                video_bytes = video_file_bin.read()
                st.video(video_bytes)
                st.download_button(
                    label="📥 Download Annotated Video",
                    data=video_bytes,
                    file_name="annotated_video.mp4",
                    mime="video/mp4"
                )

        # Display temporal smoothness plot
        temporal_plot_path = "output/temporal_smoothness.png"
        if os.path.exists(temporal_plot_path):
            st.subheader("📈 Temporal Smoothness")
            st.image(temporal_plot_path)

        # Display evaluation report text
        report_path = "output/evaluation_report.txt"
        if os.path.exists(report_path):
            st.subheader("📄 Evaluation Report")
            with open(report_path, "r") as f:
                st.text(f.read())

        # Download buttons for reports
        json_path = "output/evaluation.json"
        if os.path.exists(json_path):
            with open(json_path, "rb") as f:
                st.download_button("📥 Download Evaluation JSON", f, file_name="evaluation.json")

        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                st.download_button("📥 Download Text Report", f, file_name="evaluation_report.txt")

    else:
        st.error("❌ Please upload both the cricket video and config JSON before running.")
