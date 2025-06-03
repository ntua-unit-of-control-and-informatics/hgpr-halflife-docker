from typing import Any

import joblib
import pandas as pd
import numpy as np
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.descriptors import RDKitDescriptors, TopologicalFingerprint
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_response import PredictionResponse


class ModelService:
    def __init__(self):
        self.model = joblib.load("hgpr_model.pkl")
        self.train_data = pd.read_csv("src/train_data.csv")

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        # Convert input list to DataFrame
        input_data = pd.DataFrame(request.dataset.input)
        # Get feature columns (excluding jaqpotRowId)
        feature_cols = [col for col in input_data.columns if col != "jaqpotRowId"]
        input_values = input_data[feature_cols]
        jaqpot_dataset = JaqpotTabularDataset(
            df=input_values,
            y_cols=None,
            x_cols=["LogKa"],
            smiles_cols="SMILES",
            featurizer=[RDKitDescriptors(), TopologicalFingerprint()],
            task="regression",
        )
        selected_features = [
            "LogKa",
            "PEOE_VSA4",
            "Bit_1198",
            "Bit_1720",
            "Bit_486",
            "Bit_1840",
        ]
        X = jaqpot_dataset.X[selected_features]

        # Make predictions for all rows at once
        predictions = self.model.predict(X, return_std=True)

        # DOA calculations and results
        train_pfas = JaqpotTabularDataset(
            df=self.train_data,
            y_cols=None,
            x_cols=["LogKa"],
            smiles_cols="SMILES",
            featurizer=[RDKitDescriptors(), TopologicalFingerprint()],
            task="regression",
        )
        train_pfas = self.model.X_scaler_.transform(train_pfas.X[selected_features])
        train_dists = self.model._compute_kernel(train_pfas, train_pfas)
        # Calculate the 95th percentile of training distances for DOA threshold
        # Extract maximum value from each row in train_dists
        # Get maximum distances excluding self-similarities (diagonal)
        np.fill_diagonal(train_dists, 0)  # Set diagonal values to 0
        max_train_dists = np.max(train_dists, axis=1)
        doa_threshold = np.percentile(max_train_dists, 5)

        test_distances_from_train = []
        for i in range(len(X)):
            # Get distances from this test point to all training points
            test_point = self.model.X_scaler_.transform(
                X[i : i + 1]
            )  # Keep as 2D array
            distances_to_train = self.model._compute_kernel(
                test_point, train_pfas
            )  # Calculate mean distance for this test point
            mean_distance = np.mean(distances_to_train)
            test_distances_from_train.append(mean_distance)

        # Get dependent feature keys
        dependent_feature_keys = [
            feature.key for feature in request.model.dependent_features
        ]
        # Create prediction results
        prediction_results = []
        for i in range(len(input_data)):
            # Create dict with output features as keys
            prediction_dict = {}
            for j, feature_key in enumerate(dependent_feature_keys):
                if feature_key == "DOA":
                    doa_value = (
                        "In DOA"
                        if test_distances_from_train[i] >= doa_threshold
                        else "Not in DOA"
                    )
                    prediction_dict[feature_key] = doa_value
                else:
                    prediction_dict[feature_key] = predictions[j][i]
            prediction_dict["jaqpotMetadata"] = {
                "jaqpotRowId": int(input_data["jaqpotRowId"].iloc[i]),
            }

            prediction_results.append(prediction_dict)
        return PredictionResponse(predictions=prediction_results)
