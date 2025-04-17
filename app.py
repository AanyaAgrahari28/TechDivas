from fastapi import FastAPI
from pydantic import BaseModel
import joblib


model = joblib.load("model/job_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")


core_skills = ['Python', 'Java', 'SQL', 'C++', 'Machine Learning', 'Data Analysis', 'Communication', 'Leadership']

app = FastAPI()


class SkillRequest(BaseModel):
    skills: list[str]


from fastapi.responses import JSONResponse

@app.post("/predict")
def predict_job(request: SkillRequest):
    try:
        input_vector = [1 if skill in request.skills else 0 for skill in core_skills]
        prediction = model.predict([input_vector])[0]
        predicted_title = label_encoder.inverse_transform([prediction])[0]
        return {"predicted_job_title": predicted_title}
    except Exception as e:
        print(" INTERNAL ERROR:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
