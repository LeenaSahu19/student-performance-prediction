
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Student Success Center",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 4px;
}
.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 20px;
}
.login-box {
    max-width: 650px;
    margin: auto;
}
.small-note {
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/student_performance.csv")

data = load_data()

# ============================================================
# MODEL
# ============================================================
FEATURES = [
    "Attendance",
    "Study_Hours",
    "Previous_Score",
    "Assignment_Score",
    "Internal_Marks",
    "Participation",
    "Previous_Failures"
]

@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df["Performance"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True
    )

    return model, encoder, X_test, y_test, y_pred, accuracy, cm, report

model, encoder, X_test, y_test, y_pred, accuracy, cm, report = train_model(data)

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "logged_in": False,
    "role": None,
    "student_id": None,
    "quiz_submitted": False,
    "quiz_score": None,
    "planner": [],
    "assignments": [],
    "progress": 0,
    "reminders": [
        "Revise today's class notes",
        "Complete pending assignments",
        "Check your attendance before the next class"
    ]
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# TEACHER NOTES STORAGE
# ============================================================
TEACHER_NOTES_FILE = "data/teacher_notes.csv"

def load_teacher_notes():
    if os.path.exists(TEACHER_NOTES_FILE):
        try:
            return pd.read_csv(TEACHER_NOTES_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["Topic", "Title", "Notes"])

def save_teacher_note(topic, title, notes):
    os.makedirs("data", exist_ok=True)
    current = load_teacher_notes()
    new_note = pd.DataFrame([{
        "Topic": topic,
        "Title": title,
        "Notes": notes
    }])
    updated = pd.concat([current, new_note], ignore_index=True)
    updated.to_csv(TEACHER_NOTES_FILE, index=False)

# ============================================================
# LOGIN
# ============================================================
if not st.session_state.logged_in:
    st.markdown('<div class="main-title">🎓 Student Success Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Student Performance Prediction & Early Warning System</div>',
        unsafe_allow_html=True
    )

    st.divider()

    left, center, right = st.columns([1, 2, 1])
    with center:
        st.subheader("🔐 Login")

        role = st.radio(
            "Select Login Type",
            ["Student", "Teacher"],
            horizontal=True
        )

        if role == "Student":
            student_ids = data["Student_ID"].astype(str).tolist()
            selected_id = st.selectbox("Student ID", student_ids)
            password = st.text_input("Password", type="password", placeholder="student123")

            st.caption("Demo password: student123")

            if st.button("🚀 Login as Student", use_container_width=True):
                if password == "student123":
                    st.session_state.logged_in = True
                    st.session_state.role = "Student"
                    st.session_state.student_id = selected_id
                    st.rerun()
                else:
                    st.error("Incorrect student password.")

        else:
            username = st.text_input("Username", placeholder="teacher")
            password = st.text_input("Password", type="password", placeholder="teacher123")

            st.caption("Demo login: teacher / teacher123")

            if st.button("🚀 Login as Teacher", use_container_width=True):
                if username == "teacher" and password == "teacher123":
                    st.session_state.logged_in = True
                    st.session_state.role = "Teacher"
                    st.session_state.student_id = None
                    st.rerun()
                else:
                    st.error("Incorrect teacher username or password.")

    st.info("This is a college-project demo login. Passwords are hard-coded for demonstration only.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🎓 Student Success Center")
    st.write(f"**Role:** {st.session_state.role}")

    if st.session_state.role == "Student":
        st.write(f"**Student ID:** {st.session_state.student_id}")

    st.divider()

    st.write("### 🤖 Model")
    st.write("Random Forest Classifier")
    st.metric("Test Accuracy", f"{accuracy * 100:.1f}%")

    st.write("### 🛠️ Technologies")
    st.write("Python • Pandas • Scikit-learn • Streamlit")

    st.divider()

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.student_id = None
        st.rerun()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def risk_from_performance(performance):
    if performance == "Low":
        return "High Risk"
    if performance == "Medium":
        return "Medium Risk"
    return "Low Risk"

def predict_row(row):
    values = row[FEATURES].to_frame().T
    pred_code = model.predict(values)[0]
    performance = encoder.inverse_transform([pred_code])[0]
    return performance, risk_from_performance(performance)

def recommendation_list(row):
    recs = []

    if row["Attendance"] < 75:
        recs.append("Improve attendance and attend classes regularly.")
    if row["Study_Hours"] < 3:
        recs.append("Increase daily focused study time.")
    if row["Assignment_Score"] < 60:
        recs.append("Complete assignments regularly and review mistakes.")
    if row["Internal_Marks"] < 50:
        recs.append("Give extra attention to internal assessments.")
    if row["Participation"] < 5:
        recs.append("Participate more actively in class.")
    if row["Previous_Failures"] > 0:
        recs.append("Focus on previously difficult subjects and ask for help.")

    if not recs:
        recs.append("Keep maintaining the current study habits and performance.")

    return recs

def add_assignment(title, due_date, priority):
    st.session_state.assignments.append({
        "Assignment": title,
        "Due Date": str(due_date),
        "Priority": priority,
        "Status": "Pending"
    })

# ============================================================
# STUDENT DASHBOARD
# ============================================================
if st.session_state.role == "Student":
    student = data[data["Student_ID"].astype(str) == str(st.session_state.student_id)].iloc[0]
    performance, risk = predict_row(student)
    recs = recommendation_list(student)

    st.markdown('<div class="main-title">🎓 My Student Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Personal academic monitoring, prediction and study support</div>',
        unsafe_allow_html=True
    )

    # Reminder Center at top
    st.subheader("🔔 Reminder Center")
    reminder_col1, reminder_col2 = st.columns([3, 1])
    with reminder_col1:
        for reminder in st.session_state.reminders:
            st.info("📌 " + reminder)
    with reminder_col2:
        new_reminder = st.text_input("Add reminder", key="new_reminder")
        if st.button("➕ Add", use_container_width=True):
            if new_reminder.strip():
                st.session_state.reminders.append(new_reminder.strip())
                st.rerun()

    st.divider()

    st.info(
        "🌟 **Student Success Center:** prediction + early warning + personalized "
        "study planning + assignments + resources + goals in one place."
    )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attendance", f"{student['Attendance']:.1f}%")
    c2.metric("Study Hours", f"{student['Study_Hours']:.1f}/day")
    c3.metric("Previous Score", f"{student['Previous_Score']:.1f}")
    c4.metric("Predicted Performance", performance)

    if risk == "High Risk":
        st.error(f"🚨 {risk} — extra academic support may be useful.")
    elif risk == "Medium Risk":
        st.warning(f"⚠️ {risk} — monitor improvement regularly.")
    else:
        st.success(f"✅ {risk} — keep maintaining your current progress.")

    # Manual input / what-if prediction
    st.subheader("🎯 Set Your Current Academic Inputs")
    st.caption(
        "Set Attendance, Study Hours, Marks, and other academic inputs to check your performance prediction."
    )

    i1, i2, i3 = st.columns(3)

    with i1:
        input_attendance = st.slider(
            "Attendance (%)", 0.0, 100.0, float(student["Attendance"]), 0.5
        )
        input_study = st.slider(
            "Study Hours / Day", 0.0, 12.0, float(student["Study_Hours"]), 0.5
        )
        input_previous = st.slider(
            "Previous Score", 0.0, 100.0, float(student["Previous_Score"]), 0.5
        )

    with i2:
        input_assignment = st.slider(
            "Assignment Score", 0.0, 100.0, float(student["Assignment_Score"]), 0.5
        )
        input_internal = st.slider(
            "Internal Marks", 0.0, 100.0, float(student["Internal_Marks"]), 0.5
        )

    with i3:
        input_participation = st.slider(
            "Participation", 0.0, 10.0, float(student["Participation"]), 0.5
        )
        input_failures = st.number_input(
            "Previous Failures",
            min_value=0,
            max_value=10,
            value=int(student["Previous_Failures"]),
            step=1
        )

    custom_row = pd.DataFrame([{
        "Attendance": input_attendance,
        "Study_Hours": input_study,
        "Previous_Score": input_previous,
        "Assignment_Score": input_assignment,
        "Internal_Marks": input_internal,
        "Participation": input_participation,
        "Previous_Failures": input_failures
    }])

    if st.button("🔮 Predict My Performance", use_container_width=True):
        custom_code = model.predict(custom_row[FEATURES])[0]
        custom_performance = encoder.inverse_transform([custom_code])[0]
        custom_risk = risk_from_performance(custom_performance)

        p1, p2 = st.columns(2)
        with p1:
            st.metric("Predicted Performance", custom_performance)
        with p2:
            st.metric("Predicted Risk", custom_risk)

        if custom_risk == "High Risk":
            st.error("🚨 High Risk — consider increasing attendance/study support.")
        elif custom_risk == "Medium Risk":
            st.warning("⚠️ Medium Risk — keep monitoring and improve weak areas.")
        else:
            st.success("✅ Low Risk — keep maintaining these academic inputs.")

    st.divider()

    tabs = st.tabs([
        "🏠 Overview",
        "🧠 Smart Study Plan",
        "📅 Study Planner",
        "📋 Assignment Tracker",
        "📊 My Progress",
        "📚 Notes",
        "📝 Quiz",
        "🎯 Goals & Achievements"
    ])

    # --------------------------------------------------------
    # SMART STUDY PLAN
    # --------------------------------------------------------
    with tabs[1]:
        st.subheader("🧠 Smart Personalized Study Plan")
        st.write(
            "This plan provides personalized suggestions based on your current academic inputs, weak areas, and risk level."
        )

        smart_tasks = []

        if input_attendance < 75:
            smart_tasks.append(("📅 Attendance", "Attend classes regularly and aim for at least 75% attendance.", "High"))
        if input_study < 3:
            smart_tasks.append(("📖 Study Hours", "Add at least 1 focused study session every day.", "High"))
        if input_previous < 60:
            smart_tasks.append(("📚 Previous Score", "Revise difficult topics and solve practice questions.", "Medium"))
        if input_assignment < 60:
            smart_tasks.append(("📝 Assignments", "Complete pending assignments and review mistakes.", "High"))
        if input_internal < 50:
            smart_tasks.append(("📊 Internal Marks", "Focus on internal-test preparation and class revision.", "High"))
        if input_participation < 5:
            smart_tasks.append(("🙋 Participation", "Ask questions and participate actively in class.", "Medium"))
        if input_failures > 0:
            smart_tasks.append(("⚠️ Previous Failures", "Give extra revision time to previously difficult subjects.", "High"))

        if not smart_tasks:
            smart_tasks = [
                ("🌟 Maintain Progress", "Your current inputs are healthy. Keep the same routine.", "Low"),
                ("📝 Practice", "Take a short quiz each week to monitor your understanding.", "Low"),
                ("📈 Review", "Check your progress every weekend and adjust your plan.", "Low")
            ]

        plan_df = pd.DataFrame(
            smart_tasks,
            columns=["Focus Area", "Recommended Action", "Priority"]
        )
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

        st.subheader("📅 Suggested Weekly Routine")
        routine = pd.DataFrame({
            "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "Suggested Activity": [
                "Revise class notes + 30 min practice",
                "Assignment work + weak-topic revision",
                "Practice quiz + mistake review",
                "Revise difficult topics",
                "Assignment completion + recap",
                "Long revision session + quiz",
                "Weekly progress review + next-week planning"
            ],
            "Recommended Time": ["1–2 hr", "1–2 hr", "1 hr", "1–2 hr", "1–2 hr", "2 hr", "30–45 min"]
        })
        st.dataframe(routine, use_container_width=True, hide_index=True)

        st.info(
            "Tip: Study-plan suggestions are support recommendations, not guarantees "
            "of a particular final score."
        )

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------
    with tabs[0]:
        st.subheader("💡 Personalized Recommendations")
        for rec in recs:
            st.info("💡 " + rec)

        st.subheader("📈 Academic Profile")
        profile_df = pd.DataFrame({
            "Metric": [
                "Attendance",
                "Study Hours",
                "Previous Score",
                "Assignment Score",
                "Internal Marks",
                "Participation"
            ],
            "Value": [
                student["Attendance"],
                student["Study_Hours"],
                student["Previous_Score"],
                student["Assignment_Score"],
                student["Internal_Marks"],
                student["Participation"]
            ]
        })
        st.bar_chart(profile_df.set_index("Metric"))

        st.subheader("⚠️ Weak Area Detection")
        weak_areas = []
        if student["Attendance"] < 75:
            weak_areas.append(("Attendance", student["Attendance"], "Target: 75%+"))
        if student["Study_Hours"] < 3:
            weak_areas.append(("Study Hours", student["Study_Hours"], "Target: 3+ hours/day"))
        if student["Previous_Score"] < 60:
            weak_areas.append(("Previous Score", student["Previous_Score"], "Target: 60+"))
        if student["Assignment_Score"] < 60:
            weak_areas.append(("Assignment Score", student["Assignment_Score"], "Target: 60+"))
        if student["Internal_Marks"] < 50:
            weak_areas.append(("Internal Marks", student["Internal_Marks"], "Target: 50+"))
        if student["Participation"] < 5:
            weak_areas.append(("Participation", student["Participation"], "Target: 5+"))

        if weak_areas:
            weak_df = pd.DataFrame(weak_areas, columns=["Area", "Current", "Suggested Target"])
            st.dataframe(weak_df, use_container_width=True, hide_index=True)
        else:
            st.success("🎉 No major weak area detected using the current project thresholds.")

    # --------------------------------------------------------
    # STUDY PLANNER
    # --------------------------------------------------------
    with tabs[2]:
        st.subheader("📅 Personal Study Planner")
        st.write("Create a simple daily study plan.")

        p1, p2, p3 = st.columns(3)
        with p1:
            task = st.text_input("Study Task", placeholder="e.g. Revise Python loops")
        with p2:
            hours = st.number_input("Hours", 0.5, 12.0, 1.0, 0.5)
        with p3:
            day = st.selectbox(
                "Day",
                ["Today", "Tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )

        if st.button("➕ Add Study Task", use_container_width=True):
            if task.strip():
                st.session_state.planner.append({
                    "Day": day,
                    "Task": task.strip(),
                    "Hours": hours,
                    "Status": "Pending"
                })
                st.success("Study task added.")

        if st.session_state.planner:
            planner_df = pd.DataFrame(st.session_state.planner)
            st.dataframe(planner_df, use_container_width=True, hide_index=True)

            task_no = st.number_input(
                "Task number to mark complete",
                min_value=1,
                max_value=len(st.session_state.planner),
                value=1
            )
            if st.button("✅ Mark Task Complete"):
                st.session_state.planner[int(task_no) - 1]["Status"] = "Completed"
                st.rerun()
        else:
            st.info("No study tasks added yet.")

    # --------------------------------------------------------
    # ASSIGNMENT TRACKER
    # --------------------------------------------------------
    with tabs[3]:
        st.subheader("📋 Assignment Tracker")

        a1, a2, a3 = st.columns(3)
        with a1:
            assignment_title = st.text_input("Assignment Name", placeholder="e.g. Python Mini Project")
        with a2:
            due_date = st.date_input("Due Date")
        with a3:
            priority = st.selectbox("Priority", ["High", "Medium", "Low"])

        if st.button("➕ Add Assignment", use_container_width=True):
            if assignment_title.strip():
                add_assignment(assignment_title.strip(), due_date, priority)
                st.success("Assignment added.")

        if st.session_state.assignments:
            assignments_df = pd.DataFrame(st.session_state.assignments)
            st.dataframe(assignments_df, use_container_width=True, hide_index=True)

            assignment_no = st.number_input(
                "Assignment number to mark complete",
                min_value=1,
                max_value=len(st.session_state.assignments),
                value=1
            )
            if st.button("✅ Mark Assignment Complete"):
                st.session_state.assignments[int(assignment_no) - 1]["Status"] = "Completed"
                st.rerun()
        else:
            st.info("No assignments added yet.")

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------
    with tabs[4]:
        st.subheader("📊 Progress Tracker")

        completed_tasks = sum(x["Status"] == "Completed" for x in st.session_state.planner)
        total_tasks = len(st.session_state.planner)
        task_progress = (completed_tasks / total_tasks) if total_tasks else 0

        completed_assignments = sum(x["Status"] == "Completed" for x in st.session_state.assignments)
        total_assignments = len(st.session_state.assignments)
        assignment_progress = (completed_assignments / total_assignments) if total_assignments else 0

        pc1, pc2 = st.columns(2)
        with pc1:
            st.metric("Study Plan Completion", f"{task_progress * 100:.0f}%")
            st.progress(task_progress)
        with pc2:
            st.metric("Assignment Completion", f"{assignment_progress * 100:.0f}%")
            st.progress(assignment_progress)

        st.subheader("🎯 Academic Progress")
        progress_score = (
            0.25 * student["Attendance"] +
            0.20 * (student["Study_Hours"] / 8 * 100) +
            0.20 * student["Previous_Score"] +
            0.15 * student["Assignment_Score"] +
            0.20 * student["Internal_Marks"]
        )
        progress_score = max(0, min(100, progress_score))
        st.metric("Current Academic Progress Index", f"{progress_score:.1f}/100")
        st.progress(progress_score / 100)

    # --------------------------------------------------------
    # NOTES
    # --------------------------------------------------------
    with tabs[5]:
        st.subheader("📚 Study Notes")
        notes = {
            "Python Basics": {
                "Variables": "A variable stores a value. Example: name = 'Riya' and age = 20.",
                "Lists": "A list stores multiple values in one collection. Example: marks = [70, 80, 90].",
                "Loops": "for and while loops repeat a block of code. Use loops when the same operation is needed multiple times.",
                "Functions": "A function is a reusable block of code created with def."
            },
            "Machine Learning": {
                "Supervised Learning": "The model learns from labelled examples where the target/output is known.",
                "Classification": "Classification predicts categories such as High, Medium or Low performance.",
                "Random Forest": "Random Forest combines many decision trees and aggregates their predictions.",
                "Train-Test Split": "A dataset is commonly divided into training and testing parts to evaluate model performance."
            },
            "Data Analytics": {
                "Data Cleaning": "Data cleaning handles missing values, duplicates, incorrect formats and inconsistent entries.",
                "EDA": "Exploratory Data Analysis uses summaries and charts to understand patterns in data.",
                "Correlation": "Correlation describes how two numeric variables move together; it does not by itself prove causation.",
                "Dashboard": "A dashboard presents important metrics and visualizations in one place."
            }
        }

        subject = st.selectbox("Choose Topic", list(notes.keys()))
        for topic, content in notes[subject].items():
            with st.expander(topic):
                st.write(content)

        st.divider()
        st.subheader("🎥 YouTube Learning & Notes")
        st.caption("Choose a topic, watch related lectures on YouTube, and review the quick notes below.")

        youtube_map = {
            "Python Basics": {
                "search": "https://www.youtube.com/results?search_query=Python+basics+variables+lists+loops+functions",
                "video": "https://www.youtube.com/watch?v=rfscVS0vtbw",
                "title": "Learn Python - Full Course for Beginners",
                "channel": "freeCodeCamp.org",
                "notes": [
                    "Variables store data, for example, name = 'Riya'.",
                    "A list is a collection that can store multiple values.",
                    "for and while loops are used to repeat tasks.",
                    "A function is a reusable block of code created using the def keyword."
                ]
            },
            "Machine Learning": {
                "search": "https://www.youtube.com/results?search_query=machine+learning+classification+random+forest+train+test+split",
                "video": "https://www.youtube.com/watch?v=PcbuKRNtCUc",
                "title": "Essential Machine Learning and AI Concepts Animated",
                "channel": "freeCodeCamp.org",
                "notes": [
                    "Supervised learning learns patterns from labelled data.",
                    "Classification predicts output categories such as High, Medium, and Low.",
                    "Random Forest combines predictions from multiple decision trees.",
                    "Train-test split helps evaluate a model on unseen test data."
                ]
            },
            "Data Analytics": {
                "search": "https://www.youtube.com/results?search_query=data+analytics+data+cleaning+EDA+correlation+dashboard",
                "video": "https://www.youtube.com/results?search_query=data+analytics+data+cleaning+EDA+correlation+dashboard",
                "title": "Data Analytics Learning Videos",
                "channel": "YouTube Search",
                "notes": [
                    "Data cleaning handles missing values, duplicates, and incorrect formats.",
                    "EDA uses charts and summaries to understand patterns in a dataset.",
                    "Correlation describes the relationship between two numeric variables.",
                    "A dashboard presents important metrics and visualizations in one place."
                ]
            }
        }

        selected_resource = youtube_map[subject]

        # YouTube resources are shown as links only. Clicking a link opens YouTube
        # in the browser instead of embedding/playing the video inside the app.
        st.subheader("🎥 Recommended Learning Videos")
        st.caption("Click a link to open the video on YouTube.")

        video_links = [
            ("▶️ Python for Beginners – freeCodeCamp", "https://www.youtube.com/watch?v=rfscVS0vtbw"),
            ("▶️ Machine Learning Basics", "https://www.youtube.com/watch?v=PcbuKRNtCUc"),
            ("▶️ Random Forest Explained", "https://www.youtube.com/results?search_query=random+forest+explained"),
            ("▶️ Pandas Tutorial", "https://www.youtube.com/results?search_query=pandas+tutorial+python")
        ]

        for label, url in video_links:
            st.link_button(label, url, use_container_width=False)

        st.subheader("📒 Quick Notes from This Topic")
        for note in selected_resource["notes"]:
            st.info("📝 " + note)

        st.caption("Videos will open on YouTube. Quick notes are provided for the learning topics in the app.")

        # Teacher-added notes are kept separate from the original study notes above.
        st.divider()
        st.subheader("👨‍🏫 Teacher Notes")
        teacher_notes = load_teacher_notes()
        if teacher_notes.empty:
            st.info("No teacher notes have been added yet.")
        else:
            for _, note in teacher_notes.iterrows():
                with st.expander(f"📌 {note['Title']} — {note['Topic']}"):
                    st.write(note["Notes"])

    # --------------------------------------------------------
    # QUIZ
    # --------------------------------------------------------
    with tabs[6]:
        st.subheader("📝 Practice Quiz — 10 Questions")

        questions = [
            ("What does ML stand for?", ["Machine Learning", "Manual Logic", "Model List", "Machine Language"], 0),
            ("Which library is commonly used for DataFrames in Python?", ["Pandas", "Matplotlib", "Streamlit", "Flask"], 0),
            ("Which algorithm is used in this project?", ["Random Forest", "K-Means", "Linear Regression", "Naive Bayes"], 0),
            ("What is classification?", ["Predicting categories", "Deleting data", "Drawing only charts", "Sorting files"], 0),
            ("What does train_test_split do?", ["Splits data for training/testing", "Creates a chart", "Deletes rows", "Exports Excel"], 0),
            ("Which metric measures correct predictions overall?", ["Accuracy", "Height", "Count", "Mean"], 0),
            ("Which feature can indicate student engagement?", ["Participation", "Student ID", "Gender", "Age only"], 0),
            ("What is a confusion matrix used for?", ["Evaluating classification predictions", "Writing Python comments", "Creating folders", "Sending email"], 0),
            ("What does Streamlit help create?", ["Interactive data apps", "Operating systems", "Antivirus software", "Computer hardware"], 0),
            ("Why is early warning useful?", ["It can help identify students needing support", "It guarantees final marks", "It replaces teachers", "It removes exams"], 0)
        ]

        answers = []
        for i, (question, options, correct) in enumerate(questions):
            answer = st.radio(
                f"{i + 1}. {question}",
                options,
                key=f"quiz_{i}"
            )
            answers.append(options.index(answer))

        if st.button("📤 Submit Quiz", use_container_width=True):
            score = sum(answers[i] == questions[i][2] for i in range(len(questions)))
            st.session_state.quiz_submitted = True
            st.session_state.quiz_score = score

        if st.session_state.quiz_submitted:
            score = st.session_state.quiz_score
            if score >= 8:
                st.success(f"🎉 Your score: {score}/10")
            elif score >= 5:
                st.warning(f"👍 Your score: {score}/10 — revise a little more.")
            else:
                st.error(f"📚 Your score: {score}/10 — review the notes and try again.")

            with st.expander("🔎 Review Correct Answers"):
                for i, (question, options, correct) in enumerate(questions):
                    st.write(f"**{i + 1}. {question}**")
                    st.write(f"Correct answer: **{options[correct]}**")

            if st.button("🔄 Retake Quiz"):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = None
                st.rerun()

    # --------------------------------------------------------
    # GOAL CALCULATOR
    # --------------------------------------------------------
    with tabs[7]:
        st.subheader("🎯 Goals & Achievements")

        current_score = float(student["Previous_Score"])
        target_score = st.number_input(
            "🎯 Target Score (%)",
            min_value=0.0,
            max_value=100.0,
            value=min(90.0, max(60.0, current_score + 10))
        )

        improvement = target_score - current_score
        if improvement > 0:
            st.info(
                f"You need about **{improvement:.1f} percentage points** improvement "
                "from your previous score."
            )
        else:
            st.success("Your target is at or below your previous score.")

        st.subheader("📅 Attendance Target")
        required_attendance = st.number_input(
            "Desired Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        attendance_gap = required_attendance - float(student["Attendance"])
        if attendance_gap > 0:
            st.warning(
                f"Current attendance is {attendance_gap:.1f} percentage points "
                "below your desired target."
            )
        else:
            st.success("Your current attendance is already at or above the selected target.")

        st.divider()
        st.subheader("🏆 Achievement Center")

        completed_tasks = sum(x["Status"] == "Completed" for x in st.session_state.planner)
        completed_assignments = sum(x["Status"] == "Completed" for x in st.session_state.assignments)

        achievements = []
        if completed_tasks >= 1:
            achievements.append(("📅", "Planner Starter", "Completed your first study task."))
        if completed_tasks >= 5:
            achievements.append(("🔥", "Study Planner Pro", "Completed 5 or more study tasks."))
        if completed_assignments >= 1:
            achievements.append(("📝", "Assignment Starter", "Completed your first assignment."))
        if completed_assignments >= 3:
            achievements.append(("🏆", "Assignment Champion", "Completed 3 or more assignments."))
        if st.session_state.quiz_score is not None and st.session_state.quiz_score >= 8:
            achievements.append(("🧠", "Quiz Master", "Scored 8/10 or above in the quiz."))
        if input_attendance >= 75:
            achievements.append(("🎯", "Attendance Goal", "Current attendance meets the 75% target."))
        if input_study >= 3:
            achievements.append(("📖", "Study Habit", "Current study time meets the 3-hour target."))

        if achievements:
            for icon, title, description in achievements:
                st.success(f"{icon} **{title}** — {description}")
        else:
            st.info(
                "Complete a study task, assignment or quiz to start earning achievements."
            )

        st.subheader("📈 Goal Progress")
        goal_progress = min(current_score / target_score, 1.0) if target_score > 0 else 0
        st.write(f"Previous Score: **{current_score:.1f}%**")
        st.write(f"Target Score: **{target_score:.1f}%**")
        st.progress(goal_progress)
        st.caption(f"Goal progress: {goal_progress * 100:.0f}%")

# ============================================================
# TEACHER DASHBOARD
# ============================================================
else:
    st.markdown('<div class="main-title">👨‍🏫 Teacher Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Student monitoring, risk analysis and model evaluation</div>',
        unsafe_allow_html=True
    )

    high_performance = (data["Performance"] == "High").sum()
    medium_performance = (data["Performance"] == "Medium").sum()
    low_performance = (data["Performance"] == "Low").sum()

    high_risk = (data["Risk_Level"] == "High Risk").sum()
    medium_risk = (data["Risk_Level"] == "Medium Risk").sum()
    low_risk = (data["Risk_Level"] == "Low Risk").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Students", len(data))
    c2.metric("🟢 High Performance", high_performance)
    c3.metric("🟡 Medium Performance", medium_performance)
    c4.metric("🔴 Low Performance", low_performance)

    st.divider()

    r1, r2, r3 = st.columns(3)
    r1.metric("🔴 High Risk", high_risk)
    r2.metric("🟡 Medium Risk", medium_risk)
    r3.metric("🟢 Low Risk", low_risk)

    st.divider()

    teacher_tabs = st.tabs([
        "📊 Analytics",
        "⚠️ At-Risk Students",
        "👤 Individual Prediction",
        "📚 Teacher Notes",
        "📂 Dataset Upload & Analysis",
        "🤖 Model Performance"
    ])

    with teacher_tabs[0]:
        st.subheader("📈 Performance & Risk Analysis")
        col1, col2 = st.columns(2)

        with col1:
            perf_counts = data["Performance"].value_counts()
            fig, ax = plt.subplots(figsize=(7, 4))
            perf_counts.plot(kind="bar", ax=ax, edgecolor="black")
            ax.set_title("Performance Distribution")
            ax.set_xlabel("Performance")
            ax.set_ylabel("Students")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            risk_counts = data["Risk_Level"].value_counts()
            fig, ax = plt.subplots(figsize=(7, 4))
            risk_counts.plot(kind="bar", ax=ax, edgecolor="black")
            ax.set_title("Risk Distribution")
            ax.set_xlabel("Risk")
            ax.set_ylabel("Students")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("📚 Attendance vs Final Score")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(
            data["Attendance"],
            data["Final_Score"],
            alpha=0.6,
            edgecolors="black",
            linewidths=0.4
        )
        ax.set_xlabel("Attendance (%)")
        ax.set_ylabel("Final Score")
        ax.set_title("Attendance vs Final Score")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with teacher_tabs[1]:
        st.subheader("⚠️ At-Risk Student Monitoring")

        risk_filter = st.selectbox(
            "Select Risk Level",
            ["High Risk", "Medium Risk", "Low Risk"]
        )

        filtered = data[data["Risk_Level"] == risk_filter].copy()

        display_cols = [
            "Student_ID",
            "Attendance",
            "Study_Hours",
            "Previous_Score",
            "Assignment_Score",
            "Internal_Marks",
            "Performance",
            "Risk_Level"
        ]

        filtered_display = filtered[display_cols].copy()
        filtered_display.insert(
            0,
            "Final Score",
            filtered["Final_Score"].values
        )
        filtered_display = filtered_display.sort_values("Final Score")

        st.dataframe(
            filtered_display,
            use_container_width=True,
            hide_index=True
        )

        st.write(f"**Students in selected category:** {len(filtered)}")

    with teacher_tabs[2]:
        st.subheader("👤 Individual Student Prediction")

        student_id = st.selectbox(
            "Select Student",
            data["Student_ID"].astype(str).tolist()
        )
        selected = data[data["Student_ID"].astype(str) == student_id].iloc[0]

        predicted, selected_risk = predict_row(selected)

        a, b, c = st.columns(3)
        a.metric("Actual Performance", selected["Performance"])
        b.metric("Predicted Performance", predicted)
        c.metric("Risk Level", selected_risk)

        st.dataframe(
            selected[FEATURES].to_frame().T,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("💡 Suggested Support")
        for rec in recommendation_list(selected):
            st.info("💡 " + rec)

    with teacher_tabs[3]:
        st.subheader("📚 Add Teacher Notes")
        st.caption("These notes are separate from the existing student Notes and will appear in the Student Dashboard under Teacher Notes.")

        n1, n2 = st.columns(2)
        with n1:
            teacher_topic = st.selectbox(
                "Topic",
                ["Python Basics", "Machine Learning", "Data Analytics", "Other"]
            )
        with n2:
            teacher_title = st.text_input(
                "Note Title",
                placeholder="e.g. Important points for Unit 2"
            )

        teacher_note_text = st.text_area(
            "Write Note",
            placeholder="Teachers can add class notes, important points, or revision instructions here...",
            height=180
        )

        if st.button("➕ Add Teacher Note", use_container_width=True):
            if not teacher_title.strip() or not teacher_note_text.strip():
                st.warning("Please enter both a note title and note content.")
            else:
                save_teacher_note(
                    teacher_topic,
                    teacher_title.strip(),
                    teacher_note_text.strip()
                )
                st.success("Teacher note added successfully.")
                st.rerun()

        st.divider()
        st.subheader("📝 Added Teacher Notes")
        teacher_notes = load_teacher_notes()
        if teacher_notes.empty:
            st.info("No teacher notes have been added yet.")
        else:
            st.dataframe(
                teacher_notes,
                use_container_width=True,
                hide_index=True
            )

    with teacher_tabs[4]:
        st.subheader("📂 Upload Your Dataset")
        st.caption(
            "Upload a CSV or Excel file to analyze your own student dataset. "
            "The existing project dataset and all other features will remain unchanged."
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="custom_dataset_upload"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    uploaded_data = pd.read_csv(uploaded_file)
                else:
                    uploaded_data = pd.read_excel(uploaded_file)

                st.success(
                    f"Dataset uploaded successfully: {len(uploaded_data):,} rows and {len(uploaded_data.columns)} columns."
                )

                st.subheader("📋 Dataset Preview")
                st.dataframe(uploaded_data.head(10), use_container_width=True, hide_index=True)

                numeric_columns = uploaded_data.select_dtypes(include=np.number).columns.tolist()

                if numeric_columns:
                    st.subheader("📊 Numeric Summary")
                    st.dataframe(
                        uploaded_data[numeric_columns].describe().round(2),
                        use_container_width=True
                    )

                    chart_column = st.selectbox(
                        "Select a numeric column for distribution analysis",
                        numeric_columns,
                        key="uploaded_numeric_column"
                    )
                    fig_upload, ax_upload = plt.subplots(figsize=(9, 4))
                    ax_upload.hist(uploaded_data[chart_column].dropna(), bins=20, edgecolor="black")
                    ax_upload.set_title(f"Distribution of {chart_column}")
                    ax_upload.set_xlabel(chart_column)
                    ax_upload.set_ylabel("Count")
                    plt.tight_layout()
                    st.pyplot(fig_upload)
                else:
                    st.info("No numeric columns were found for statistical analysis.")

                if "Performance" in uploaded_data.columns:
                    st.subheader("🎯 Performance Distribution")
                    perf = uploaded_data["Performance"].value_counts()
                    st.bar_chart(perf)

                if "Risk_Level" in uploaded_data.columns:
                    st.subheader("⚠️ Risk Distribution")
                    risk = uploaded_data["Risk_Level"].value_counts()
                    st.bar_chart(risk)

                missing_features = [feature for feature in FEATURES if feature not in uploaded_data.columns]
                st.subheader("🤖 Machine Learning Prediction")

                if not missing_features:
                    prediction_data = uploaded_data[FEATURES].copy()
                    valid_rows = prediction_data.notna().all(axis=1)

                    if valid_rows.any():
                        predicted_codes = model.predict(prediction_data.loc[valid_rows, FEATURES])
                        predicted_labels = encoder.inverse_transform(predicted_codes)
                        prediction_result = uploaded_data.loc[valid_rows].copy()
                        prediction_result["Predicted Performance"] = predicted_labels

                        st.success("The uploaded dataset contains all required model features. Predictions are available.")
                        st.dataframe(
                            prediction_result[[c for c in ["Student_ID", "Predicted Performance", "Performance", "Final_Score", "Risk_Level"] if c in prediction_result.columns]],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.warning("The required model features are present, but some rows contain missing values. No prediction was generated for those rows.")
                else:
                    st.info(
                        "Machine-learning prediction is unavailable because these required columns are missing: "
                        + ", ".join(missing_features)
                    )

            except Exception as e:
                st.error(f"Unable to analyze this dataset. Please check the file format and column values. Details: {e}")

    with teacher_tabs[5]:
        st.subheader("🤖 Machine Learning Model Performance")

        a, b, c = st.columns(3)
        a.metric("Model", "Random Forest")
        b.metric("Test Accuracy", f"{accuracy * 100:.1f}%")
        c.metric("Trees", "300")

        st.subheader("📊 Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=encoder.classes_
        )
        disp.plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title("Model Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig_cm)

        st.subheader("📋 Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

        st.subheader("📌 Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
        ax_imp.barh(
            importance_df["Feature"],
            importance_df["Importance"],
            edgecolor="black"
        )
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title("Random Forest Feature Importance")
        plt.tight_layout()
        st.pyplot(fig_imp)

        st.info(
            "Feature importance shows the relative contribution of input features "
            "to this trained Random Forest model. It should not be interpreted as causation."
        )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Student Performance Prediction & Early Warning System | "
    "Python • Machine Learning • Streamlit | College Minor Project"
)
