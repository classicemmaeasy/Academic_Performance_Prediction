import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Safe import for Plotly with automatic fallback to Streamlit native charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F9FAFB;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .result-badge-high {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 6px 15px rgba(16, 185, 129, 0.35);
    }
    .result-badge-med {
        background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 6px 15px rgba(245, 158, 11, 0.35);
    }
    .result-badge-low {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 6px 15px rgba(239, 68, 68, 0.35);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Machine Learning Model
# ---------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "academic_performance_model_RF.joblib")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessor.joblib")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: `{MODEL_PATH}`. Please ensure the `.joblib` model is located in the application directory.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_preprocessor():
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            return joblib.load(PREPROCESSOR_PATH)
        except Exception:
            return None
    return None

model = load_model()
preprocessor = load_preprocessor()

# ---------------------------------------------------------
# Metadata & Definitions
# ---------------------------------------------------------
CLASS_MAPPING = {
    0: {
        "code": "H",
        "label": "High-Level Academic Performance",
        "description": "Student is on track for excellence (score range 85%–100%). Demonstrates strong classroom engagement and consistent study habits.",
        "badge_class": "result-badge-high",
        "color": "#10B981"
    },
    1: {
        "code": "L",
        "label": "Low-Level Performance (At-Risk)",
        "description": "Student is at risk of academic underperformance (score range below 70%). Immediate teacher/parental intervention and support are advised.",
        "badge_class": "result-badge-low",
        "color": "#EF4444"
    },
    2: {
        "code": "M",
        "label": "Middle-Level Performance",
        "description": "Student shows satisfactory progress (score range 70%–84%). With targeted guidance, the student can transition into the high-achieving tier.",
        "badge_class": "result-badge-med",
        "color": "#F59E0B"
    }
}

NATIONALITIES = [
    'KW', 'Jordan', 'Palestine', 'Iraq', 'lebanon', 'Tunis', 'SaudiArabia',
    'Egypt', 'Syria', 'USA', 'Iran', 'Lybia', 'Morocco', 'venzuela'
]

PLACES_OF_BIRTH = [
    'KuwaIT', 'Jordan', 'Palestine', 'Iraq', 'lebanon', 'Tunis', 'SaudiArabia',
    'Egypt', 'Syria', 'USA', 'Iran', 'Lybia', 'Morocco', 'venzuela'
]

STAGES = ['lowerlevel', 'MiddleSchool', 'HighSchool']
GRADES = ['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12']
SECTIONS = ['A', 'B', 'C']
TOPICS = ['IT', 'Math', 'Arabic', 'Science', 'English', 'Quran', 'Spanish', 'French', 'History', 'Biology', 'Chemistry', 'Geology']
SEMESTERS = [('F', 'Fall / First Semester (F)'), ('S', 'Spring / Second Semester (S)')]

# ---------------------------------------------------------
# Feature Transformation Utility
# ---------------------------------------------------------
def transform_to_model_features(model_instance, record_dict):
    """
    Transforms a dictionary of raw input features into the exact 72-feature
    one-hot encoded DataFrame expected by the Random Forest model.
    """
    if preprocessor is not None:
        try:
            raw_df = pd.DataFrame([record_dict])
            features_out = list(preprocessor.get_feature_names_out())
            transformed_arr = preprocessor.transform(raw_df)
            return pd.DataFrame(transformed_arr, columns=features_out)
        except Exception:
            pass

    feature_names = list(model_instance.feature_names_in_)
    row = {col: 0.0 for col in feature_names}
    
    # Categorical indicator flags
    cat_keys = [
        f"cat__gender_{record_dict['gender']}",
        f"cat__NationalITy_{record_dict['NationalITy']}",
        f"cat__PlaceofBirth_{record_dict['PlaceofBirth']}",
        f"cat__StageID_{record_dict['StageID']}",
        f"cat__GradeID_{record_dict['GradeID']}",
        f"cat__SectionID_{record_dict['SectionID']}",
        f"cat__Topic_{record_dict['Topic']}",
        f"cat__Semester_{record_dict['Semester']}",
        f"cat__Relation_{record_dict['Relation']}",
        f"cat__ParentAnsweringSurvey_{record_dict['ParentAnsweringSurvey']}",
        f"cat__ParentschoolSatisfaction_{record_dict['ParentschoolSatisfaction']}",
        f"cat__StudentAbsenceDays_{record_dict['StudentAbsenceDays']}",
    ]
    
    for k in cat_keys:
        if k in row:
            row[k] = 1.0
            
    # Continuous / numerical features
    row['remainder__raisedhands'] = float(record_dict.get('raisedhands', 0))
    row['remainder__VisITedResources'] = float(record_dict.get('VisITedResources', 0))
    row['remainder__AnnouncementsView'] = float(record_dict.get('AnnouncementsView', 0))
    row['remainder__Discussion'] = float(record_dict.get('Discussion', 0))
    
    return pd.DataFrame([row], columns=feature_names)

# ---------------------------------------------------------
# Sidebar: Helpful Assistant & Quick Fill
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Quick Assistant")
    st.markdown(
        "Need a quick test? Pick an example student below to automatically fill in the form:"
    )
    
    preset_choice = st.selectbox(
        "Choose an Example Student:",
        [
            "Custom Student (Enter your own)",
            "Example: High-Achieving Student",
            "Example: Average Student",
            "Example: Student Needing Support"
        ],
        index=0
    )
    
    # Preset student values
    defaults = {
        "gender": "M",
        "nationality": "KW",
        "pob": "KuwaIT",
        "stage": "MiddleSchool",
        "grade": "G-07",
        "section": "A",
        "topic": "IT",
        "semester": "F",
        "relation": "Mum",
        "survey": "Yes",
        "satisfaction": "Good",
        "absences": "Under-7",
        "raisedhands": 75,
        "visited": 80,
        "announcements": 85,
        "discussion": 70
    }
    
    if preset_choice == "Example: High-Achieving Student":
        defaults.update({
            "gender": "F", "nationality": "KW", "pob": "KuwaIT", "stage": "MiddleSchool",
            "grade": "G-07", "section": "A", "topic": "IT", "semester": "F",
            "relation": "Mum", "survey": "Yes", "satisfaction": "Good",
            "absences": "Under-7", "raisedhands": 90, "visited": 95, "announcements": 90, "discussion": 85
        })
    elif preset_choice == "Example: Average Student":
        defaults.update({
            "gender": "M", "nationality": "Jordan", "pob": "Jordan", "stage": "MiddleSchool",
            "grade": "G-08", "section": "B", "topic": "Science", "semester": "S",
            "relation": "Father", "survey": "Yes", "satisfaction": "Good",
            "absences": "Under-7", "raisedhands": 45, "visited": 50, "announcements": 40, "discussion": 40
        })
    elif preset_choice == "Example: Student Needing Support":
        defaults.update({
            "gender": "M", "nationality": "KW", "pob": "KuwaIT", "stage": "HighSchool",
            "grade": "G-11", "section": "A", "topic": "Math", "semester": "F",
            "relation": "Father", "survey": "No", "satisfaction": "Bad",
            "absences": "Above-7", "raisedhands": 8, "visited": 10, "announcements": 6, "discussion": 12
        })

    st.markdown("---")
    st.markdown("### How to Use This Tool")
    st.markdown("""
    1. **Fill out the student details** in the 3 simple sections.
    2. Click **'Predict Student Outcome'**.
    3. Review the **estimated score tier and recommended advice**.
    """)

    st.markdown("---")
    st.link_button("GitHub Repository", "https://github.com/classicemmaeasy/Academic_Performance_Prediction", use_container_width=True)

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎓 Student Success & Performance Advisor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">An easy-to-use tool for teachers, parents, and school administrators to understand student progress and provide early support.</div>',
    unsafe_allow_html=True
)

if model is None:
    st.stop()

# ---------------------------------------------------------
# Friendly Tabs
# ---------------------------------------------------------
tab_single, tab_analytics = st.tabs([
    "Check a Student",
    "What Helps Students Succeed?"
])

# =========================================================
# TAB 1: Individual Student Assessment
# =========================================================
with tab_single:
    st.info("**Tip:** Fill in the basic info and engagement scores below, then click **'Predict Student Outcome'** to see the forecast and recommendations.")

    with st.form("student_friendly_form"):
        # Section 1 & 2
        st.markdown("#### 1. Basic Information & School Details")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            gender = st.selectbox(
                "Gender:",
                ["M", "F"],
                index=0 if defaults["gender"] == "M" else 1,
                format_func=lambda x: "Male" if x == "M" else "Female"
            )
            stage = st.selectbox(
                "School Level:",
                STAGES,
                index=STAGES.index(defaults["stage"]) if defaults["stage"] in STAGES else 0,
                format_func=lambda x: {
                    "lowerlevel": "Primary / Elementary School",
                    "MiddleSchool": "Middle School",
                    "HighSchool": "High School"
                }.get(x, x)
            )

        with c2:
            grade = st.selectbox(
                "Grade Level:",
                GRADES,
                index=GRADES.index(defaults["grade"]) if defaults["grade"] in GRADES else 0,
                format_func=lambda x: f"Grade {x.replace('G-', '')}"
            )
            section = st.selectbox(
                "Class Section:",
                SECTIONS,
                index=SECTIONS.index(defaults["section"]) if defaults["section"] in SECTIONS else 0,
                format_func=lambda x: f"Section {x}"
            )

        with c3:
            topic = st.selectbox(
                "Subject / Course:",
                TOPICS,
                index=TOPICS.index(defaults["topic"]) if defaults["topic"] in TOPICS else 0
            )
            semester = st.selectbox(
                "Current Term / Semester:",
                ["F", "S"],
                index=0 if defaults["semester"] == "F" else 1,
                format_func=lambda x: "1st Term (Fall)" if x == "F" else "2nd Term (Spring)"
            )

        with st.expander("Optional: Origin & Demographics (click to view/edit)"):
            d1, d2, d3 = st.columns(3)
            with d1:
                nationality = st.selectbox("Nationality:", NATIONALITIES, index=NATIONALITIES.index(defaults["nationality"]) if defaults["nationality"] in NATIONALITIES else 0)
            with d2:
                pob = st.selectbox("Place of Birth:", PLACES_OF_BIRTH, index=PLACES_OF_BIRTH.index(defaults["pob"]) if defaults["pob"] in PLACES_OF_BIRTH else 0)
            with d3:
                relation = st.selectbox(
                    "Primary Guardian:",
                    ["Father", "Mum"],
                    index=0 if defaults["relation"] == "Father" else 1,
                    format_func=lambda x: "Mother" if x == "Mum" else "Father"
                )

        st.markdown("---")

        # Section 2: Attendance & Family Support
        st.markdown("#### 2. Attendance & Home Support")
        f1, f2, f3 = st.columns(3)
        
        with f1:
            absences = st.selectbox(
                "Days Absent This Term:",
                ["Under-7", "Above-7"],
                index=0 if defaults["absences"] == "Under-7" else 1,
                format_func=lambda x: "Good Attendance (Fewer than 7 days absent)" if x == "Under-7" else "Frequent Absences (More than 7 days absent)",
                help="Attendance is one of the highest predictors of academic performance."
            )
        
        with f2:
            survey = st.selectbox(
                "Parent Participates in School Surveys:",
                ["Yes", "No"],
                index=0 if defaults["survey"] == "Yes" else 1,
                format_func=lambda x: "Yes, actively participates" if x == "Yes" else "No / Rarely"
            )

        with f3:
            satisfaction = st.selectbox(
                "Parent School Satisfaction:",
                ["Good", "Bad"],
                index=0 if defaults["satisfaction"] == "Good" else 1,
                format_func=lambda x: "Satisfied with school" if x == "Good" else "Needs Improvement / Dissatisfied"
            )

        st.markdown("---")

        # Section 3: Daily Learning & Activity
        st.markdown("#### 3. Daily Learning & Class Participation")
        st.caption("How active is the student during lessons and on the digital school portal? (0 = Rare / Low, 100 = Frequent / Highly Active)")
        
        e1, e2 = st.columns(2)
        with e1:
            raised_hands = st.slider(
                "Asks & Answers Questions in Class:",
                min_value=0,
                max_value=100,
                value=int(defaults["raisedhands"]),
                help="How frequently does the student raise their hand or actively participate in teacher discussions?"
            )
            visited_resources = st.slider(
                "Reads Course Materials & Online Lessons:",
                min_value=0,
                max_value=100,
                value=int(defaults["visited"]),
                help="Number of times student reviewed reading materials, study guides, or lesson slides."
            )

        with e2:
            announcements = st.slider(
                "Checks School & Class Announcements:",
                min_value=0,
                max_value=100,
                value=int(defaults["announcements"]),
                help="Frequency of checking assignments, homework updates, and notice board announcements."
            )
            discussion = st.slider(
                "Participates in Group Study & Forum Discussions:",
                min_value=0,
                max_value=100,
                value=int(defaults["discussion"]),
                help="Contributions to group study boards, discussion topics, and peer peer learning."
            )

        st.markdown("")
        submit_button = st.form_submit_button("Predict Student Outcome", type="primary", use_container_width=True)

    # Prediction Processing
    if submit_button:
        student_data = {
            "gender": gender,
            "NationalITy": nationality,
            "PlaceofBirth": pob,
            "StageID": stage,
            "GradeID": grade,
            "SectionID": section,
            "Topic": topic,
            "Semester": semester,
            "Relation": relation,
            "ParentAnsweringSurvey": survey,
            "ParentschoolSatisfaction": satisfaction,
            "StudentAbsenceDays": absences,
            "raisedhands": raised_hands,
            "VisITedResources": visited_resources,
            "AnnouncementsView": announcements,
            "Discussion": discussion
        }

        # Convert to model features
        input_df = transform_to_model_features(model, student_data)
        
        prediction_num = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = probabilities[prediction_num] * 100

        st.markdown("---")
        st.markdown("### Assessment Results")

        res_col1, res_col2 = st.columns([1.1, 0.9])

        with res_col1:
            # Friendly outcome banner
            if prediction_num == 0:  # High
                st.markdown(
                    """
                    <div style="background-color: #ECFDF5; border-left: 6px solid #10B981; padding: 18px 20px; border-radius: 8px; margin-bottom: 16px;">
                        <h3 style="color: #065F46; margin: 0 0 6px 0;">Expected Outcome: High Achiever</h3>
                        <p style="color: #047857; margin: 0; font-size: 1.05rem;">
                            <strong>Estimated Score Range: 85% – 100%</strong><br>
                            This student demonstrates strong participation, solid attendance, and healthy learning habits.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif prediction_num == 2:  # Medium
                st.markdown(
                    """
                    <div style="background-color: #FFFBEB; border-left: 6px solid #F59E0B; padding: 18px 20px; border-radius: 8px; margin-bottom: 16px;">
                        <h3 style="color: #92400E; margin: 0 0 6px 0;">Expected Outcome: Average / Satisfactory</h3>
                        <p style="color: #B45309; margin: 0; font-size: 1.05rem;">
                            <strong>Estimated Score Range: 70% – 84%</strong><br>
                            The student is making steady progress, but a little extra guidance can help them excel into the top tier.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:  # Low
                st.markdown(
                    """
                    <div style="background-color: #FEF2F2; border-left: 6px solid #EF4444; padding: 18px 20px; border-radius: 8px; margin-bottom: 16px;">
                        <h3 style="color: #991B1B; margin: 0 0 6px 0;">⚠️ Needs Extra Support / At Risk</h3>
                        <p style="color: #B91C1C; margin: 0; font-size: 1.05rem;">
                            <strong>Estimated Score Range: Below 70%</strong><br>
                            Early intervention is strongly recommended to support this student before major exams.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write(f"**Confidence Level:** `{confidence:.1f}% certainty`")

            # Simple practical advice for teachers and parents
            st.markdown("#### Recommended Next Steps:")
            if prediction_num == 0:
                st.success("**Keep It Up**: Offer enrichment activities, encourage participation in academic clubs, or invite them to mentor peers.")
            elif prediction_num == 2:
                st.warning("**Boost Study Habit**: Encourage checking course resources weekly and participating more often during classroom discussions.")
            else:
                st.error("**Immediate Action**: Schedule a supportive check-in with the student and parents. Focus on reducing absences and creating an achievable weekly study plan.")

        with res_col2:
            st.markdown("#### Outcome Forecast")
            prob_high = probabilities[0] * 100
            prob_low = probabilities[1] * 100
            prob_med = probabilities[2] * 100

            chart_data = pd.DataFrame({
                "Category": ["High (85-100%)", "Average (70-84%)", "Needs Support (<70%)"],
                "Chance (%)": [prob_high, prob_med, prob_low],
                "Color": ["#10B981", "#F59E0B", "#EF4444"]
            })

            if HAS_PLOTLY:
                fig = go.Figure(go.Bar(
                    x=chart_data["Chance (%)"],
                    y=chart_data["Category"],
                    orientation="h",
                    marker_color=chart_data["Color"],
                    text=[f"{p:.1f}%" for p in chart_data["Chance (%)"]],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="<b>Chances for each performance level</b>",
                    xaxis_title="Likelihood (%)",
                    xaxis_range=[0, 100],
                    height=280,
                    margin=dict(l=10, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                for _, r in chart_data.iterrows():
                    st.write(f"**{r['Category']}**: {r['Chance (%)']:.1f}%")
                    st.progress(min(max(float(r['Chance (%)']) / 100.0, 0.0), 1.0))

            # Student's engagement summary
            st.markdown("##### Engagement Snapshot")
            e_col1, e_col2 = st.columns(2)
            e_col1.metric("Classroom Questions", f"{raised_hands}/100")
            e_col1.metric("Study Resources Visited", f"{visited_resources}/100")
            e_col2.metric("Notices Checked", f"{announcements}/100")
            e_col2.metric("Discussion Board Posts", f"{discussion}/100")

# =========================================================
# TAB 2: What Drives Success? (Friendly Insights)
# =========================================================
with tab_analytics:
    st.markdown("#### Key Factors Influencing Student Performance")
    st.markdown(
        "Based on extensive analysis of educational data, here are the most crucial factors that separate top-performing students from those needing support:"
    )

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("""
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 16px;">
            <h4 style="color: #166534; margin-top: 0;">1. Consistent Attendance</h4>
            <p style="color: #14532D; font-size: 0.95rem;">
                Students with fewer than 7 days absent per term have a significantly higher rate of achieving top grades. Regular presence in class is the single biggest predictor.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 16px;">
            <h4 style="color: #1E40AF; margin-top: 0;">2. Reading Study Materials</h4>
            <p style="color: #1E3A8A; font-size: 0.95rem;">
                Actively viewing learning resources and lecture notes between classes allows students to reinforce lessons and retain information much better.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div style="background: #FAF5FF; border: 1px solid #E9D5FF; border-radius: 10px; padding: 16px;">
            <h4 style="color: #6B21A8; margin-top: 0;">3. Active Classroom Questions</h4>
            <p style="color: #581C87; font-size: 0.95rem;">
                Raising hands and answering questions signals mental engagement and curiosity, which closely correlates with strong exam performance.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Visual Feature Importance in Friendly Terms
    if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
        importances = model.feature_importances_
        features = model.feature_names_in_
        
        # Friendly translation of feature names
        name_map = {
            "cat__StudentAbsenceDays_Under-7": "Good Attendance (Under 7 Days Absent)",
            "cat__StudentAbsenceDays_Above-7": "Frequent Absences (Above 7 Days Absent)",
            "remainder__VisITedResources": "Accessing Course Study Materials",
            "remainder__raisedhands": "Asking & Answering Questions in Class",
            "remainder__AnnouncementsView": "Checking Class Announcements",
            "remainder__Discussion": "Participating in Group Discussions",
            "cat__ParentAnsweringSurvey_Yes": "Parent Active in School Surveys",
            "cat__ParentschoolSatisfaction_Good": "High Parent Satisfaction with School",
            "cat__Relation_Mum": "Primary Guardian: Mother",
            "cat__Relation_Father": "Primary Guardian: Father",
        }

        clean_features = []
        for f in features:
            clean = name_map.get(f, f.replace("cat__", "").replace("remainder__", "").replace("_", " "))
            clean_features.append(clean)

        imp_df = pd.DataFrame({
            "Factor": clean_features,
            "Impact": importances
        }).sort_values(by="Impact", ascending=False).reset_index(drop=True)

        top_factors = imp_df.head(8)

        st.markdown("##### Top Factors Driving Success in the Data")
        if HAS_PLOTLY:
            fig_imp = px.bar(
                top_factors,
                x="Impact",
                y="Factor",
                orientation="h",
                color="Impact",
                color_continuous_scale="Blues",
                labels={"Impact": "Relative Importance", "Factor": "Factor"}
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(l=10, r=20, t=20, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.bar_chart(top_factors.set_index("Factor")["Impact"])

    # Optional Technical Expander for data scientists
    with st.expander("Advanced Technical Specifications (For Developers & Data Scientists)"):
        st.markdown("""
        - **Algorithm**: Random Forest Classifier (`scikit-learn`)
        - **Training Data**: 480 Students, 16 Original Features
        - **Preprocessing**: 72 One-Hot and Numerical Features
        - **Evaluated Models**: Tested against Gradient Boosting, XGBoost, Logistic Regression, and Support Vector Machines.
        """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>"
    "Student Success & Performance Advisor • Designed for Educators & Parents"
    "</div>",
    unsafe_allow_html=True
)
