from flask import Flask, request, render_template, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from groq import Groq
import os

# -------------------------------------------------
# Brevo Configuration
# -------------------------------------------------

brevo_configuration = sib_api_v3_sdk.Configuration()
brevo_configuration.api_key['api-key'] = os.environ.get('BREVO_API_KEY')

sender_email = "bharath.98458@gmail.com"
sender_name = "Cancer Risk System"

# -------------------------------------------------
# Groq Configuration
# -------------------------------------------------

groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

# -------------------------------------------------
# Flask App Initialization
# -------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'cancer_risk_secret_key')

# -------------------------------------------------
# Load Dataset & Train Model
# -------------------------------------------------

df = pd.read_csv('cancer.csv')

selected_features = [
    'Air Pollution',
    'Genetic Risk',
    'Obesity',
    'Balanced Diet',
    'OccuPational Hazards',
    'Coughing of Blood'
]

X = df[selected_features]
y = df['Level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy * 100)
print("Confusion Matrix:\n", conf_matrix)

# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    air_pollution        = float(request.form['Air Pollution'])
    genetic_risk         = float(request.form['Genetic Risk'])
    obesity              = float(request.form['Obesity'])
    balanced_diet        = float(request.form['Balanced Diet'])
    occupational_hazards = float(request.form['Occupational Hazards'])
    coughing_of_blood    = float(request.form['coughing_of_blood'])
    patient_name         = str(request.form['Patient_Name'])
    receiver_email       = str(request.form['receiver_email'])

    input_data = [[
        air_pollution,
        genetic_risk,
        obesity,
        balanced_diet,
        occupational_hazards,
        coughing_of_blood
    ]]

    prediction       = rf_clf.predict(input_data)
    prediction_proba = rf_clf.predict_proba(input_data)
    predicted_class  = prediction[0]
    predicted_prob   = max(prediction_proba[0]) * 100

    # Store RF result in session for Step 2
    session['rf_result'] = {
        'predicted_class': predicted_class,
        'predicted_prob': predicted_prob,
        'predicted_probability': dict(zip(rf_clf.classes_, [float(p) for p in prediction_proba[0]])),
        'inputs': {
            'air_pollution': air_pollution,
            'genetic_risk': genetic_risk,
            'obesity': obesity,
            'balanced_diet': balanced_diet,
            'occupational_hazards': occupational_hazards,
            'coughing_of_blood': coughing_of_blood
        }
    }
    session['patient_name']    = patient_name
    session['receiver_email']  = receiver_email

    return render_template('step2.html', patient_name=patient_name, predicted_class=predicted_class)


@app.route('/analyze', methods=['POST'])
def analyze():

    # Pull Step 1 data from session
    rf_result      = session.get('rf_result', {})
    patient_name   = session.get('patient_name', 'Patient')
    receiver_email = session.get('receiver_email', '')

    # Step 2 descriptive answers
    desc_air_pollution  = request.form.get('desc_air_pollution', '').strip()
    desc_alcohol        = request.form.get('desc_alcohol', '').strip()
    desc_coughing_blood = request.form.get('desc_coughing_blood', '').strip()
    desc_genetic_risk   = request.form.get('desc_genetic_risk', '').strip()

    predicted_class = rf_result.get('predicted_class', 'Unknown')
    predicted_prob  = rf_result.get('predicted_prob', 0)
    inputs          = rf_result.get('inputs', {})

    # -------------------------------------------------
    # Build Groq Prompt
    # -------------------------------------------------

    groq_prompt = f"""
You are a medical AI assistant specializing in cancer risk analysis.

A patient named {patient_name} has been assessed by a Random Forest ML model which predicted their cancer risk level as: "{predicted_class}" with {predicted_prob:.2f}% confidence.

Their numeric health scores were:
- Air Pollution Exposure: {inputs.get('air_pollution')}
- Genetic Risk: {inputs.get('genetic_risk')}
- Obesity: {inputs.get('obesity')}
- Balanced Diet: {inputs.get('balanced_diet')}
- Occupational Hazards: {inputs.get('occupational_hazards')}
- Coughing of Blood: {inputs.get('coughing_of_blood')}

The patient has also provided the following descriptive responses:

1. Air Pollution Exposure (describe your daily environment):
"{desc_air_pollution}"

2. Alcohol Usage (describe your drinking habits):
"{desc_alcohol}"

3. Coughing of Blood (describe any symptoms):
"{desc_coughing_blood}"

4. Genetic Risk (family history of cancer):
"{desc_genetic_risk}"

Based on both the ML prediction and the patient's own description, provide a detailed, empathetic, and medically informed analysis that covers:
1. A summary of the patient's overall risk profile
2. Key risk factors identified from their descriptions
3. How their lifestyle and environment contribute to their risk level
4. Specific, actionable recommendations to reduce their risk
6. Your answer should support the predicted cancer risk level by Random forest model

Be clear, compassionate, and thorough. Do not use bullet points in a mechanical way — write in flowing paragraphs.
"""

    # -------------------------------------------------
    # Call Groq
    # -------------------------------------------------

    groq_analysis = ""

    try:
        chat_completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": groq_prompt
                }
            ],
            max_tokens=1024
        )
        groq_analysis = chat_completion.choices[0].message.content

    except Exception as e:
        groq_analysis = f"AI analysis could not be completed at this time. Error: {str(e)}"
        print(f"Groq error: {e}")

    # -------------------------------------------------
    # Build Combined Email Report
    # -------------------------------------------------

    report_html = f"""
<html><body style="font-family: Arial, sans-serif; max-width: 700px; margin: auto; padding: 30px; color: #111;">

<h2 style="color: #b42828;">Cancer Risk Assessment Report</h2>
<p>Hi <strong>{patient_name}</strong>, below is your complete health assessment report.</p>

<hr style="border-color: #ddd; margin: 20px 0;">

<h3 style="color: #b42828;"> ML Model Prediction</h3>
<table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
  <tr>
    <td style="padding: 8px; background: #f7f4ef; font-weight: 600;">Risk Level</td>
    <td style="padding: 8px; background: #f7f4ef;">{predicted_class}</td>
  </tr>
  <tr>
    <td style="padding: 8px; font-weight: 600;">Confidence</td>
    <td style="padding: 8px;">{predicted_prob:.2f}%</td>
  </tr>
  <tr>
    <td style="padding: 8px; background: #f7f4ef; font-weight: 600;">Air Pollution</td>
    <td style="padding: 8px; background: #f7f4ef;">{inputs.get('air_pollution')}</td>
  </tr>
  <tr>
    <td style="padding: 8px; font-weight: 600;">Genetic Risk</td>
    <td style="padding: 8px;">{inputs.get('genetic_risk')}</td>
  </tr>
  <tr>
    <td style="padding: 8px; background: #f7f4ef; font-weight: 600;">Obesity</td>
    <td style="padding: 8px; background: #f7f4ef;">{inputs.get('obesity')}</td>
  </tr>
  <tr>
    <td style="padding: 8px; font-weight: 600;">Balanced Diet</td>
    <td style="padding: 8px;">{inputs.get('balanced_diet')}</td>
  </tr>
  <tr>
    <td style="padding: 8px; background: #f7f4ef; font-weight: 600;">Occupational Hazards</td>
    <td style="padding: 8px; background: #f7f4ef;">{inputs.get('occupational_hazards')}</td>
  </tr>
  <tr>
    <td style="padding: 8px; font-weight: 600;">Coughing of Blood</td>
    <td style="padding: 8px;">{inputs.get('coughing_of_blood')}</td>
  </tr>
</table>

<hr style="border-color: #ddd; margin: 20px 0;">

<h3 style="color: #b42828;">🩺 AI Descriptive Analysis</h3>
<div style="background: #f7f4ef; padding: 20px; border-radius: 8px; line-height: 1.8; white-space: pre-wrap;">{groq_analysis}</div>

<hr style="border-color: #ddd; margin: 20px 0;">

<p style="font-size: 0.8rem; color: #999;">This is an AI-based assessment and is not a substitute for professional medical advice. Please consult a qualified healthcare provider for any health concerns.</p>

</body></html>
"""

    report_text = f"""
Cancer Risk Assessment Report

Hi {patient_name}, below is your complete health assessment.

--- ML MODEL PREDICTION ---
Risk Level:            {predicted_class}
Confidence:            {predicted_prob:.2f}%
Air Pollution:         {inputs.get('air_pollution')}
Genetic Risk:          {inputs.get('genetic_risk')}
Obesity:               {inputs.get('obesity')}
Balanced Diet:         {inputs.get('balanced_diet')}
Occupational Hazards:  {inputs.get('occupational_hazards')}
Coughing of Blood:     {inputs.get('coughing_of_blood')}

--- AI DESCRIPTIVE ANALYSIS ---
{groq_analysis}

---
This is an AI-based assessment. Please consult a medical professional for confirmation.
"""

    # -------------------------------------------------
    # Send Email via Brevo
    # -------------------------------------------------

    try:
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
            sib_api_v3_sdk.ApiClient(brevo_configuration)
        )

        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
            to=[{"email": receiver_email, "name": patient_name}],
            sender={"email": sender_email, "name": sender_name},
            subject="Your Cancer Risk Assessment Report",
            html_content=report_html,
            text_content=report_text
        )

        api_instance.send_transac_email(send_smtp_email)
        print("Email sent successfully to", receiver_email)

    except ApiException as e:
        print(f"Failed to send email: {e}")

    # -------------------------------------------------
    # Return Final Result
    # -------------------------------------------------

    result = {
        'predicted_class': predicted_class,
        'predicted_prob': predicted_prob,
        'predicted_probability': rf_result.get('predicted_probability', {}),
        'groq_analysis': groq_analysis
    }

    return render_template('index.html', result=result)


# -------------------------------------------------
# Run App
# -------------------------------------------------

if __name__ == '__main__':
    app.run(debug=False)