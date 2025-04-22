import pandas as pd

class Patient:
    """Represents a patient with attributes relevant for heart disease prediction."""

    def __init__(self, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
        
        # Basic type conversion 
        self.age = int(age)
        self.sex = int(sex)
        self.cp = int(cp)
        self.trestbps = int(trestbps)
        self.chol = int(chol)
        self.fbs = int(fbs)
        self.restecg = int(restecg)
        self.thalach = int(thalach)
        self.exang = int(exang)
        self.oldpeak = float(oldpeak)
        self.slope = int(slope)
        self.ca = int(ca) 
        self.thal = int(thal) 

    def to_dataframe(self):
        """Converts the patient's attributes into a single-row Pandas DataFrame
           suitable for PyCaret's predict_model function."""

        # Create dictionary ensuring keys match the model's expected feature names
        data_dict = {
            'age': [self.age],
            'sex': [self.sex],
            'cp': [self.cp],
            'trestbps': [self.trestbps],
            'chol': [self.chol],
            'fbs': [self.fbs],
            'restecg': [self.restecg],
            'thalach': [self.thalach],
            'exang': [self.exang],
            'oldpeak': [self.oldpeak],
            'slope': [self.slope],
            'ca': [self.ca],
            'thal': [self.thal]
        }
       
        return pd.DataFrame.from_dict(data_dict)