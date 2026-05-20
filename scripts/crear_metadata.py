import re
import unicodedata
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
FILES_DIR = DATA_DIR / "02_files"
PROCESSED_DATA_DIR = DATA_DIR / "03_processed"
OUTPUT_PATH = PROCESSED_DATA_DIR / "metadata.csv"

OUTPUT_COLUMNS = [
    "file_name",
    "transcription",
    "speaker_id",
    "dialect_variant",
    "gender",
]

APOSTROPHE_TRANSLATION = str.maketrans({
    "’": "'",
    "‘": "'",
    "ʼ": "'",
    "`": "'",
    "´": "'",
})

GENDER_MAP = {
    "masculino": "M",
    "m": "M",
    "femenino": "F",
    "f": "F",
}


def read_csv_auto(path):
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def clean_transcription(text):
    if pd.isna(text):
        return ""

    text = str(text).translate(APOSTROPHE_TRANSLATION)
    text = unicodedata.normalize("NFC", text).lower()

    cleaned_chars = []
    for char in text:
        if char == "'" or char.isspace() or char.isalnum():
            cleaned_chars.append(char)
        elif unicodedata.category(char).startswith("M"):
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")

    return re.sub(r"\s+", " ", "".join(cleaned_chars)).strip()


def normalize_gender(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return GENDER_MAP.get(value.lower(), value)


def normalize_dialect(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "_", str(value).strip())


def find_source_files():
    csv_files = sorted(FILES_DIR.glob("*.csv"))
    if len(csv_files) < 2:
        raise FileNotFoundError(f"Se esperaban al menos 2 CSV en {FILES_DIR}")

    labels_path = None
    phrases_path = None

    for path in csv_files:
        sample = read_csv_auto(path)
        columns = set(sample.columns)

        if {"speaker_id", "gender", "dialect_variant", "codigo_frase", "audio_file"}.issubset(columns):
            labels_path = path
        elif {"codigo_frase", "transcripcion"}.issubset(columns):
            phrases_path = path

    if labels_path is None:
        raise FileNotFoundError("No se encontro el CSV de labels con audio_file y speaker_id.")
    if phrases_path is None:
        raise FileNotFoundError("No se encontro el CSV de frases con codigo_frase y transcripcion.")

    return labels_path, phrases_path


def main():
    labels_path, phrases_path = find_source_files()

    labels = read_csv_auto(labels_path)
    phrases = read_csv_auto(phrases_path)

    if phrases["codigo_frase"].duplicated().any():
        duplicated = phrases.loc[phrases["codigo_frase"].duplicated(), "codigo_frase"].unique()
        raise ValueError(f"Hay codigos de frase duplicados en frases: {duplicated}")

    merged = labels.merge(
        phrases[["codigo_frase", "transcripcion"]],
        on="codigo_frase",
        how="left",
        validate="many_to_one",
    )

    missing_transcriptions = merged["transcripcion"].isna().sum()
    if missing_transcriptions:
        raise ValueError(f"Hay {missing_transcriptions} filas sin transcripcion despues del merge.")

    merged["file_name"] = merged["audio_file"].map(lambda value: f"{Path(str(value)).stem}.wav")
    merged["transcription"] = merged["transcripcion"].map(clean_transcription)
    merged["speaker_id"] = merged["speaker_id"].astype(str).str.strip()
    merged["dialect_variant"] = merged["dialect_variant"].map(normalize_dialect)
    merged["gender"] = merged["gender"].map(normalize_gender)

    processed_files = {path.name for path in PROCESSED_DATA_DIR.glob("*.wav")}
    metadata_files = set(merged["file_name"])

    missing_audio = sorted(metadata_files - processed_files)
    if missing_audio:
        raise FileNotFoundError(
            "Hay audios en metadata que no existen en 03_processed: "
            + ", ".join(missing_audio[:10])
        )

    if merged["file_name"].duplicated().any():
        duplicated = merged.loc[merged["file_name"].duplicated(), "file_name"].unique()
        raise ValueError(f"Hay file_name duplicados: {duplicated[:10]}")

    output = merged[OUTPUT_COLUMNS].sort_values(["speaker_id", "file_name"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Labels: {len(labels)}")
    print(f"Frases: {len(phrases)}")
    print(f"Filas consolidadas: {len(output)}")
    print(f"Archivo generado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
