import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA

class BEpiQ:
    def __init__(self, 
                 model_path=os.path.join("qmodel"),
                 scaler_path=os.path.join("scaler.pkl"),
                 feature_map=None,
                 ansatz=None,
                 optimizer=None):
        """
        Inisialisasi BEpiQ: Memuat model dan scaler serta menerima konfigurasi untuk VQC.
        
        :param feature_map: (optional) Feature map untuk VQC (default: ZZFeatureMap)
        :param ansatz: (optional) Ansatz untuk VQC (default: RealAmplitudes)
        :param optimizer: (optional) Optimizer untuk VQC (default: COBYLA)
        """
        self.model, self.scaler = self.load_model_and_scaler(model_path, scaler_path)

        # Set konfigurasi VQC
        self.feature_map = feature_map or ZZFeatureMap(3, reps=2, entanglement="full")  # Default ZZFeatureMap
        self.ansatz = ansatz or RealAmplitudes(3, entanglement='full', reps=3)  # Default RealAmplitudes
        self.optimizer = optimizer or COBYLA(maxiter=100)  # Default COBYLA

    def preprocess(self, input_string):
        """
        Preprocessing input string: mapping amino acids to numeric and normalizing.
        """
        # Mapping asam amino menjadi nilai numerik
        amino_mapping = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
            'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
        }

        # Memetakan input string menjadi urutan angka
        ex_list = list(input_string)
        ex_len = len(ex_list)
        ex_numeric = [amino_mapping.get(str(amino).upper(), 0) for amino in ex_list]

        # Membuat fitur untuk prediksi
        ex_feature = np.array([[i, ex_len, amino] for i, amino in enumerate(ex_numeric)])

        # Normalisasi fitur dengan scaler yang telah dimuat
        ex_feature_scaled = self.scaler.transform(ex_feature)

        return ex_feature_scaled

    def predict(self, input_string):
        """
        Melakukan prediksi dengan model VQC
        """
        processed_input = self.preprocess(input_string)
        predictions = self.model.predict(processed_input)
        return predictions

    @staticmethod
    def load_model_and_scaler(model_path='qmodel', scaler_path='scaler.pkl'):
        """
        Memuat model VQC dan scaler.
        """
        model = VQC.load(model_path)  # Memuat model VQC
        scaler = joblib.load(scaler_path)  # Memuat scaler
        return model, scaler

    def train(self, peptides, labels, model_save_path='qmodel', scaler_save_path='scaler.pkl'):
        """
        Fungsi untuk melatih model VQC dengan data peptida dan label.
        """
        # Membuat dataset untuk pelatihan
        X, y = self.create_training_data(peptides, labels)

        # Normalisasi data dengan StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Menyimpan scaler untuk digunakan saat prediksi
        joblib.dump(self.scaler, scaler_save_path)

        # Mendefinisikan VQC model
        num_qubits = X_scaled.shape[1]
        feature_map = self.feature_map
        ansatz = self.ansatz
        optimizer = self.optimizer

        # Melatih model
        self.model.fit(X_scaled, y)

        # Menyimpan model
        self.save_model(model_save_path)
        print(f"Model telah disimpan di {model_save_path}")

    def create_training_data(self, peptides, labels):
        """
        Membuat dataset untuk pelatihan, mengonversi string peptida dan label ke format yang sesuai.
        """
        amino_mapping = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
            'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
        }

        # Menyiapkan data fitur dan label
        X = []
        y = []

        for peptide, label in zip(peptides, labels):
            ex_list = list(peptide)
            ex_len = len(ex_list)
            ex_numeric = [amino_mapping.get(str(amino).upper(), 0) for amino in ex_list]
            ex_feature = np.array([[i, ex_len, amino] for i, amino in enumerate(ex_numeric)])
            label = [label] * len(ex_feature)
            X.append(ex_feature)
            y.extend(label)

        X = np.concatenate(X)  # Menggabungkan data fitur dari seluruh peptida
        y = np.array(y)

        return X, y

    def save_model(self, model_save_path='qmodel'):
        """
        Menyimpan model VQC ke path yang ditentukan.
        """
        self.model.save(model_save_path)
        print(f"Model telah disimpan di {model_save_path}")
