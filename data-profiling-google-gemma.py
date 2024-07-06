from fastapi import FastAPI,Request,UploadFile,File
import json
import pandas as pd
import os
import google.generativeai as genai
from ydata_profiling import ProfileReport
import jinja2
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:4200",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from dotenv import load_dotenv
load_dotenv("dot.env")

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name='gemini-1.5-flash')
agent = {"designation": "Business Analyst", "department": "Finanace", "workplace": "Test Hospital, Abu Dhabi"}
sk = {"designation": "CEO", "department": "Finanace", "workplace": "Test Hospital, Abu Dhabi"}

agent_prompt_str = """
  You are a {{agent.designation}} at {{agent.department}} department of {{agent.workplace}}.
  Here are a list of alerts, raised by an automated tool from dataset:
  
  {% for alert in alerts %}
    - {{ alert }}
  {% endfor %}
  
  For each alert your task is to:
  1. Provide a summary of the alert
  2. Provide decision making insights for the Business 
    - if applicable suggest with example on impact of inflation, geo political events, macro economic events
  3. Provide actionable insights from the alerts
  
  Specific Instructions:
  - Remove unnecessary details: The {{sk.designation}} likely isn't concerned either specific numbers or technical terms.
  - Actionable insights: Rephrase alerts to highlight potential decisions and next steps.
  - Conciseness: Focused on key takeaways and offer a clear path forward.
  - Removed roleplay element: {{sk.designation}} likely doesn't need to be reminded of your role.
  - Focus on Decision Making: These findings offer valuable insights for potential actions.
  - Don't use the word "Alert" in your response, you are providing actionable insights
"""


@app.post("/llm_profiling_excel")
async def llm_profiling_excel(file: UploadFile = File(...)):
    spreadsheet = file
    excel_file = spreadsheet.file.read()
    df = pd.read_excel(excel_file,skiprows=2)
    print(df.columns)
    df = df.astype({2021: "int64", 2022: "int64", 2023: "int64"})
    return profiling(df)

@app.post("/llm_profiling_json")
async def llm_profiling_excel(payload: Request):
    json_arr = await payload.json()
    df = pd.DataFrame(json_arr)
    df = df.astype({'2021': "int64", '2022': "int64", '2023': "int64"})
    return profiling(df)


def profiling(df):
  profile = ProfileReport(df, title="Profiling Report")
  analysis = profile.to_json()
  alerts = json.loads(analysis)["alerts"]
  environment = jinja2.Environment()

  template = environment.from_string(agent_prompt_str)
  prompt = template.render(name="Alerts Analysis", agent=agent, sk=sk, alerts=alerts)
  #
  response = model.generate_content(prompt)
  return response.text

# Run the application (for development purposes)
if __name__ == "__main__":
  import uvicorn
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)