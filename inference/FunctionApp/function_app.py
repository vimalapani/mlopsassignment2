import os
import joblib
import azure.functions as func
import logging

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="mlops_assignmet2")
def mlops_assignmet2(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    data = req.params.get('data')
    if not data:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            data = req_body.get('data')

    if data:
        # Load the pre-trained model
        model_path = os.path.join(os.getcwd(), 'best_model_mlops.joblib')
        model = joblib.load(model_path)
        data1= eval(data)
        prediction = model.predict([data1])
        return func.HttpResponse(f"Prediction: {prediction}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass valid data in the query string or in the request body for appropriate model predictions.",
             status_code=200
        )