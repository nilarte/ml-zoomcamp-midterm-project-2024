import pickle
import gradio as gr

# Load the model and vectorizer
model_file = 'rf.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Define the prediction function
def predict_heart_disease(age, sex, chestpaintype, restingbp, cholesterol, fastingbs, restingecg, maxhr, exerciseangina, oldpeak, st_slope):
    # Create patient dictionary
    patient = {
        "age": age,
        "sex": sex,
        "chestpaintype": chestpaintype,
        "restingbp": restingbp,
        "cholesterol": cholesterol,
        "fastingbs": fastingbs,
        "restingecg": restingecg,
        "maxhr": maxhr,
        "exerciseangina": exerciseangina,
        "oldpeak": oldpeak,
        "st_slope": st_slope,
    }
    # Validate and preprocess inputs
    # Convert numerical fields to float
    patient['age'] = float(patient['age'])
    patient['restingbp'] = float(patient['restingbp'])
    patient['cholesterol'] = float(patient['cholesterol'])
    patient['fastingbs'] = int(patient['fastingbs'])
    patient['maxhr'] = float(patient['maxhr'])
    patient['oldpeak'] = float(patient['oldpeak'])
    # Transform input and predict
    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    #heartdisease = y_pred >= 0.5

    return {
        "Heart Disease Probability": round(y_pred, 2),
        #"Heart Disease Prediction": "High Risk" if heartdisease else "Low Risk"
    }

# Gradio interface
interface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(choices=["M", "F"], label="Sex"),
        gr.Radio(choices=["ATA", "NAP", "ASY", "TA"], label="Chest Pain Type"),
        gr.Number(label="Resting Blood Pressure"),
        gr.Number(label="Cholesterol"),
        gr.Radio(choices=[0, 1], label="Fasting Blood Sugar (1=True, 0=False)"),
        gr.Radio(choices=["Normal", "ST", "LVH"], label="Resting ECG"),
        gr.Number(label="Maximum Heart Rate Achieved"),
        gr.Radio(choices=["Y", "N"], label="Exercise-Induced Angina"),
        gr.Number(label="Old Peak (ST depression)"),
        gr.Radio(choices=["Up", "Flat", "Down"], label="Slope of the ST Segment"),
    ],
    outputs=[
        gr.Label(label="Prediction Results"),
    ],
    title="Heart Disease Prediction",
    description="Predict the likelihood of heart disease based on patient data.",
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
