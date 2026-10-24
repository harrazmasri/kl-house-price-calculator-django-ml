from django.shortcuts import render
import joblib

# Create your views here.
model = joblib.load('houseApp/ml_model/house_price_model.pkl')

def predict_price(request):
    predicted_price = None

    if request.method == 'POST':
        bedrooms = float(request.POST['bedrooms'])
        bathrooms = float(request.POST['bathrooms'])
        size_sqft = float(request.POST['size_sqft'])
        age = float(request.POST['age'])

        house_data = [[
            bedrooms,
            bathrooms,
            size_sqft,
            age,
        ]]

        prediction = model.predict(house_data)

        predicted_price = round(prediction[0], 2)

    return render(
        request,
        'houseApp/index.html',
        {'predicted_price': predicted_price},
    )