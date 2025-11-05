# Disease Prediction & Medical Recommendation System

A comprehensive machine learning-based web application for predicting diseases and providing medical recommendations.

## Features

- **Diabetes Prediction**: Predict diabetes risk based on medical parameters
- **Heart Disease Prediction**: Assess cardiovascular disease risk
- **Symptom-Based Prediction**: Identify diseases based on symptoms
- **Medical Recommendations**: Get treatment suggestions and precautions

## Installation

1. Extract the zip file
2. Open terminal/command prompt in the project directory
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Navigate to the project directory
2. Run the following command:
   ```bash
   streamlit run app.py
   ```
3. The app will open in your default web browser at `http://localhost:8501`

## Project Structure

```
Disease_Prediction_App/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/                          # Dataset folder
│   ├── diabetes.csv               # Diabetes dataset
│   ├── heart.csv                  # Heart disease dataset
│   ├── symptom_disease.csv        # Symptom-disease mapping
│   └── medicine_recommendations.csv # Treatment recommendations
├── models/                        # (Models are trained on-the-fly)
└── utils/                         # Utility scripts

```

## How to Use

### Diabetes Prediction
1. Select "Diabetes Prediction" from the sidebar
2. Enter medical parameters (glucose, BMI, age, etc.)
3. Click "Predict Diabetes" to get results
4. View recommendations based on prediction

### Heart Disease Prediction
1. Select "Heart Disease Prediction" from the sidebar
2. Enter cardiac parameters
3. Click "Predict Heart Disease" to get results
4. Follow the provided recommendations

### Symptom-Based Prediction
1. Select "Symptom-Based Disease Prediction" from the sidebar
2. Check the symptoms you're experiencing
3. Click "Predict Disease" to identify potential conditions
4. View medicine recommendations and precautions

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy
- **ML Algorithm**: Random Forest Classifier

## Models

The application uses Random Forest Classifier for all predictions:
- **Diabetes Model**: Trained on 200+ samples
- **Heart Disease Model**: Trained on 200+ samples  
- **Symptom Model**: Trained on 300+ symptom-disease mappings

## Datasets

All datasets are included in the `data/` folder:
- `diabetes.csv`: Medical parameters for diabetes prediction
- `heart.csv`: Cardiac parameters for heart disease prediction
- `symptom_disease.csv`: Symptom-disease associations
- `medicine_recommendations.csv`: Treatment and precaution data

## Important Notes

⚠️ **Disclaimer**: This application is for educational and informational purposes only. 
It should NOT replace professional medical advice, diagnosis, or treatment. 
Always consult qualified healthcare professionals for medical concerns.

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Internet connection (for initial package installation)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Troubleshooting

### Issue: Module not found error
**Solution**: Run `pip install -r requirements.txt` to install all dependencies

### Issue: Data files not found
**Solution**: Ensure you're running the app from the project root directory

### Issue: Port already in use
**Solution**: Stop other Streamlit apps or specify a different port:
```bash
streamlit run app.py --server.port 8502
```

## Features in Detail

### 1. User-Friendly Interface
- Clean and intuitive design
- Easy navigation through sidebar
- Color-coded predictions (red for high risk, green for low risk)

### 2. Real-Time Predictions
- Instant results upon entering data
- Probability scores for transparency
- Multiple disease predictions from symptoms

### 3. Medical Recommendations
- Disease-specific treatment suggestions
- Dietary recommendations
- Precautionary measures
- Lifestyle advice

### 4. Data Privacy
- All processing done locally
- No data stored or transmitted
- Complete privacy of medical information

## Future Enhancements

- Add more disease prediction models
- Integration with medical databases
- User authentication and history tracking
- Doctor appointment booking
- Multi-language support

## Support

For issues or questions, please ensure:
1. All dependencies are correctly installed
2. You're running Python 3.8 or higher
3. All data files are in the correct location

## Version

**Current Version**: 1.0.0  
**Release Date**: November 2025

## License

This project is for educational purposes. Use responsibly and always consult healthcare professionals for medical decisions.

---

**Built with ❤️ using Streamlit and Machine Learning**
