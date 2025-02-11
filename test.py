from BEpiQ.BEpiQPipeline import BEpiQ
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA

# Membuat konfigurasi khusus
feature_map = ZZFeatureMap(3, reps=2, entanglement="full")
optimizer = COBYLA(maxiter=200)

# Membuat objek BEpiQ
bepiq_pipeline = BEpiQ(feature_map=feature_map, optimizer=optimizer)

# Latih dan prediksi
peptides = ["ABCD", "EFGH"]
labels = [0, 1]
bepiq_pipeline.train(peptides, labels)

input_data = "ABCD"
prediction = bepiq_pipeline.predict(input_data)
print(f"Prediksi untuk data '{input_data}': {prediction}")
