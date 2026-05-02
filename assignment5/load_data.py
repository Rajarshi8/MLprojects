import pandas as pd
import os

def load_energy_data():
    base_path = os.path.dirname(__file__)

    file_path = os.path.join(base_path, "ENB2012_data.xlsx")

    df = pd.read_excel(file_path)

    df.columns = [
        "Relative_Compactness","Surface_Area","Wall_Area","Roof_Area",
        "Overall_Height","Orientation","Glazing_Area","Glazing_Distribution",
        "Heating_Load","Cooling_Load"
    ]

    return df