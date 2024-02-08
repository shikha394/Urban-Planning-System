from flask import Flask, render_template, request
from Model import combine_models

app = Flask(__name__)

# Assuming you have a list of cities (city_names) from your dataset
# Replace this with your actual list of city names
city_names = ['Agartala', 'Agra', 'Ahmedabad', 'Aizwal', 'Ajmer', 'Allahabad', 'Amritsar', 'Andaman and Nicobar',
              'Asansol', 'Bangalore', 'Begusarai', 'Bhopal', 'Bhubaneswar', 'Bilaspur', 'Bokaro', 'Buxar', 'Chandigarh',
              'Chennai', 'Coimbatore', 'Cuttack', 'Dadar-Nagar Haveli', 'Daman & diu', 'Darbhanga', 'Dehradun', 'Delhi',
              'Dhanbad', 'Dibugarh', 'Durgapur', 'Faridabad', 'Gandhinagar', 'Gaya', 'Ghaziabad', 'Gorakhpur',
              'Gurgaon', 'Guwahati', 'Hajipur', 'Haldia', 'Hissar', 'Howrah', 'Hydrabad', 'Imphal', 'Indore',
              'Jabalpur', 'Jaipur', 'Jammu', 'Jamshedpur', 'Jhansi', 'Jodhpur', 'Kanpur', 'Kochi', 'Kohima', 'Kolkata',
              'Korba', 'Kota', 'Kurnool', 'Lakshadweep', 'Lucknow', 'Madurai', 'Manali', 'Mangalore', 'Mathura',
              'Meerut', 'Meghalaya', 'Mirzapur', 'Mumbai', 'Muzaffarpur', 'Mysore', 'Nagpur', 'Nasik', 'Navi Mumbai',
              'Panaji', 'Panipat', 'Pathankot', 'Patiala', 'Patna', 'Puducherry', 'Pune', 'Puri', 'Rae Bareli',
              'Raipur', 'Rajkot', 'Ranchi', 'Rishikesh', 'Rourkela', 'Sambalpur', 'Shimla', 'Silachar', 'Siliguri',
              'Srinagar', 'Surat', 'Thane', 'Thiruvananathpuram', 'Tirupati', 'Ujjain', 'Vadodra', 'Varanasi',
              'Vellore', 'Vishakhapatnam', 'Warangal']


@app.route('/')
def index():
    return render_template('index.html', city_names=city_names)


app.config['STATIC_FOLDER'] = 'static'


@app.route('/result', methods=['POST'])
def result():
    city_input = request.form['city']
    year_input = int(request.form['year'])
    result, img, img_pop = combine_models(city_input, year_input)
    return render_template('result.html', result=result, city=city_input, year=year_input, img=img, img_pop=img_pop)


if __name__ == '__main__':
    app.run(debug=True)
