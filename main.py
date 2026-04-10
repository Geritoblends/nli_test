import sys
import csv
import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def process_entailment(hypothesis, file_paths):
    model_path = "model.onnx"
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra {model_path} en el directorio actual.")
        return

    # Cargamos recursos
    print("Cargando modelo y tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    csv_data = []
    relevant_chunks = []
    threshold = 0.7

    print(f"Analizando {len(file_paths)} archivos...\n")

    for filepath in file_paths:
        if not os.path.exists(filepath):
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Preparar input para el modelo (Formato NLI)
        # <s> Premise </s></s> Hypothesis </s>
        text = f"{content} </s></s> {hypothesis}"
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=1024)
        
        # Inferencia ONNX
        ort_inputs = {k: v for k, v in inputs.items()}
        logits = session.run(None, ort_inputs)[0]
        
        # Los modelos BART-MNLI suelen tener etiquetas: [contradiction, neutral, entailment]
        # El índice 2 (último) es el score de 'entailment'
        probs = softmax(logits[0])
        score = float(probs[-1])

        # Guardar para el CSV
        csv_data.append({'filepath': filepath, 'value': f"{score:.4f}"})

        # Si supera el umbral, lo guardamos para el TXT
        if score >= threshold:
            relevant_chunks.append(f"--- Archivo: {filepath} (Score: {score:.4f}) ---\n{content}\n")
            status = " [RELEVANTE]"
        else:
            status = ""

        print(f"Procesado: {filepath} | Score: {score:.4f}{status}")

    # 1. Guardar CSV
    with open("resultados.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filepath', 'value'])
        writer.writeheader()
        writer.writerows(csv_data)

    # 2. Guardar TXT Consolidado
    with open("relevantes.txt", 'w', encoding='utf-8') as f:
        if relevant_chunks:
            f.write(f"Hipótesis evaluada: {hypothesis}\n")
            f.write("="*50 + "\n\n")
            f.write("\n\n".join(relevant_chunks))
        else:
            f.write("No se encontraron fragmentos con score >= 0.7")

    print(f"\nListo. CSV generado: resultados.csv")
    print(f"TXT de fragmentos relevantes: relevantes.txt ({len(relevant_chunks)} encontrados)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Uso: python entail.py "Mi hipótesis" chunk_*.txt')
        sys.exit(1)

    # El primer argumento es la hipótesis, el resto son los archivos
    hypo = sys.argv[1]
    files = sys.argv[2:]
    process_entailment(hypo, files)
