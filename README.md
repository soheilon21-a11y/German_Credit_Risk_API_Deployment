# 🚀 German Credit Risk Prediction API (FastAPI Deployment)

This project demonstrates the successful deployment of a Machine Learning classification model using a *RESTful API* developed with *FastAPI*. The service predicts the credit risk (Good/Bad) of a customer based on 48 processed features derived from the German Credit Data.

---

## 🛠 Key Technologies & MLOps Tools

This project showcases competence in critical MLOps (Machine Learning Operations) and deployment concepts:

* *API Framework:* FastAPI (Chosen for high performance and automatic documentation).
* *Server:* Uvicorn (ASGI server used for running the FastAPI application).
* *Model:* Logistic Regression (Trained using Scikit-learn).
* *Model Serialization:* Joblib (Used to serialize/save the trained model: german_credit_model.pkl).
* *MLOps Focus:* Model Deployment, API Development, and Dependency Management (requirements.txt).

---

## 💻 How to Run the API Locally

To test the deployment locally, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/soheilon21-a11y/German_Credit_Risk_API_Deployment.git](https://github.com/soheilon21-a11y/German_Credit_Risk_API_Deployment.git)
cd German_Credit_Risk_API_Deployment

2. Install Dependencies
All required packages (FastAPI, Uvicorn, Pandas, etc.) are listed in requirements.txt.
pip install -r requirements.txt

3. Start the Server
Run the Uvicorn server from the project directory:
python -m uvicorn main:app --reload
(The server will start running on http://127.0.0.1:8000)

🧪 How to Test the Prediction Endpoint
You can interact with the deployed model using the automatic documentation interface (Swagger UI):

Access Docs: Open your web browser and navigate to: http://127.0.0.1:8000/docs

Perform Prediction:

Click on the /predict_risk endpoint.

Click Try it out.

Paste the following sample of 48 processed features into the Request body:
{
  "features": [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
Click Execute to receive the prediction result (e.g., "Good Credit Risk").
