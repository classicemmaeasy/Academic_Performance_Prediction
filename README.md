# 🎓 Student Academic Performance Predictor

An interactive web application built with **Streamlit**, **Scikit-Learn**, and **Plotly** to evaluate and predict student academic performance using the trained Random Forest model (`academic_performance_model_RF.joblib`).

**GitHub Repository**: [https://github.com/classicemmaeasy/Academic_Performance_Prediction](https://github.com/classicemmaeasy/Academic_Performance_Prediction)

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/classicemmaeasy/Academic_Performance_Prediction.git
cd Academic_Performance_Prediction
pip install -r requirements.txt
```

### 2. Launch the Web Application
Run the Streamlit app:
```bash
streamlit run main.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 🌟 Application Features

1. **Individual Student Assessment**:
   - Demographic details (Gender, Nationality, Place of Birth)
   - Academic details (Educational Stage, Grade level, Section, Subject / Topic, Semester)
   - Attendance & Parental Engagement (Absence frequency, Survey responses, School satisfaction, Primary relation)
   - LMS & Classroom Engagement sliders (Raised hands, Visited resources, Announcements viewed, Discussion posts)
   - Instant predictions with **Confidence Probability Distribution** and **Behavioral Radar Charts**.
   - Custom pedagogical interventions and recommendations for High, Medium, and Low performance categories.

2. **Quick Preset Profile Loader**:
   - Instantly test predefined student profiles:
     - 🌟 **High Achiever**
     - ⚖️ **Average / Moderate Student**
     - ⚠️ **At-Risk Student**

3. **Batch Evaluation (CSV Upload)**:
   - Upload class cohorts in `.csv` format.
   - Batch inference with exportable classified CSV reports.
   - Built-in sample CSV template download.
   - Visual cohort breakdown charts (Pie / Donut and Histogram).

4. **Model Explainability & Insights**:
   - Interactive feature importance chart revealing the primary drivers of student academic performance.
