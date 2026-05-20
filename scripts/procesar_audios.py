import os
import torchaudio.transforms as T
import torch
import av
import numpy as np
import soundfile as sf

# Config de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "01_raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "03_processed")
TARGET_SAMPLE_RATE = 16000  # Frecuencia de 16KHz

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_audio_with_av(file_path):
    container = av.open(file_path)
    audio_stream = container.streams.audio[0]
    sample_rate = audio_stream.rate

    audio_data = []
    for frame in container.decode(audio_stream):
        audio_data.append(frame.to_ndarray())

    waveform_np = np.concatenate(audio_data, axis=-1)
    waveform = torch.from_numpy(waveform_np).float()

    if waveform.abs().max() > 0:
        waveform /= waveform.abs().max()

    return waveform, sample_rate

def preprocess_audio_file(file_path, output_path):
    try:
        # Carga de audio
        waveform, sample_rate = load_audio_with_av(file_path)

        # Estandarizacion a mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Convertir a 16KHz
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)

        # Escribir archivo
        sf.write(output_path, waveform.numpy().T, TARGET_SAMPLE_RATE)
        print(f"Procesado con exito: {os.path.basename(output_path)}")

    except Exception as e:
        print(f"Error procesando {file_path}: {str(e)}")

def main():
    os.system('cls') 
    print("Iniciando preprocesamiento de audios Mam...")
    ensure_dir(PROCESSED_DATA_DIR)

    valid_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac')

    # Procesamiento por lotes
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.lower().endswith(valid_extensions):
            raw_path = os.path.join(RAW_DATA_DIR, filename)

            # Forzar extensión a .wav
            output_filename = os.path.splitext(filename)[0] + ".wav"
            processed_path = os.path.join(PROCESSED_DATA_DIR, output_filename)

            preprocess_audio_file(raw_path, processed_path)

    print("\nProceso finalizado. Audios guardados en 03_processed.")

if __name__ == "__main__":
    main()
