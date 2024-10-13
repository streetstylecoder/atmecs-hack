# main.py
import pandas as pd

from utils import (
    fetch_bioactivity_data,
    calculate_descriptors,
    train_qsar_model,
    predict_activity,
    fetch_admet_properties
)
# import pandas as pd

def main():
    # Step 1: Fetch bioactivity data for Alzheimer's Tau protein
    target_name = 'Tau protein'
    data = fetch_bioactivity_data(target_name)
    if data is None or data.empty:
        print("No data retrieved. Exiting.")
        return

    # Step 2: Calculate descriptors for all molecules
    descriptor_list = data['SMILES'].apply(calculate_descriptors)
    descriptor_df = pd.DataFrame(descriptor_list.tolist())
    activity_series = data['IC50']

    # Step 3: Train the QSAR model
    model, scaler, to_drop, feature_names = train_qsar_model(descriptor_df, activity_series)

    #generative model 
    #input - list of smile data from fetc-bioactivity-data
    #output - list of new smiles 


    # Step 4: Predict activity of a new molecule
    new_smiles = 'CC(=O)OCCC1=CNc2c1cc(OC)cc2'  # Example SMILES
    predicted_activity = predict_activity(new_smiles, model, scaler, to_drop, feature_names)
    print(f'Predicted Activity (IC50) for {new_smiles}: {predicted_activity}')

    #filters will be added here
    # Step 5: Fetch ADMET properties (optional)
    admet_props = fetch_admet_properties(new_smiles)
    print(f'ADMET Properties for {new_smiles}: {admet_props}')

if __name__ == '__main__':
    main()
