#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from google.cloud import translate
import pandas as pd
import random
import fire

# ============================================================= #
# ================== CONFIGURATION ============================= #
# ============================================================= #

GENDERED_LANGS = {
    'Bulgarian': 'bg', 'French': 'fr', 'German': 'de', 'Greek': 'el',
    'Italian': 'it', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Maltese': 'mt',
    'Portuguese': 'pt', 'Romanian': 'ro', 'Spanish': 'es', 'Catalan': 'ca',
    'Galician': 'gl', 'Croatian': 'hr', 'Czech': 'cs', 'Polish': 'pl',
    'Slovak': 'sk', 'Slovenian': 'sl', 'Russian': 'ru', 'Ukrainian': 'uk'
}

NEUTRAL_LANGS = {
    'Danish': 'da', 'Dutch': 'nl', 'Estonian': 'et', 'Finnish': 'fi',
    'Hungarian': 'hu', 'Irish': 'ga', 'Swedish': 'sv',
    'Norwegian': 'no', 'Turkish': 'tr', 'Indonesian': 'id',
    'Chinese': 'zh'
}

ALL_LANGS = {**GENDERED_LANGS, **NEUTRAL_LANGS}

# ===================== MAIN FUNCTION ===================== #

def main(
    data_path,
    json_key,
    project_id,
    out_dir,
    sample_size,
    languages
):
    """
    Translate English GEST sentences into multiple European languages.

    :param data_path: Path to the source GEST CSV file.
    :param json_key: Path to your Google Cloud Service Account JSON key.
    :param project_id: Your Google Cloud Project ID.
    :param out_dir: Folder where translations will be saved.
    :param sample_size: If 1 (default), runs full dataset. If > 1, picks a random sample.
    :param languages: Optional comma-separated list of languages to process.
    """
    
    # 1. Setup Paths
    data_file = Path(data_path)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Dataset
    gest_df = pd.read_csv(data_file)

    # 3. Determine Languages and Sample
    if languages:
        # If user provides a specific list: "Spanish,French"
        if isinstance(languages, str):
            languages = [l.strip() for l in languages.split(',')]
        test_langs = languages
        sampled_df = gest_df if sample_size == 1 else gest_df.sample(n=sample_size)
    elif sample_size != 1:
        # Original logic: Pick one random gendered and one random neutral for testing
        test_langs = [
            random.choice(list(GENDERED_LANGS.keys())),
            random.choice(list(NEUTRAL_LANGS.keys()))
        ]
        sampled_df = gest_df.sample(n=sample_size)
    else:
        # Full run
        test_langs = list(ALL_LANGS.keys())
        sampled_df = gest_df

    print(f"Processing languages: {test_langs}")
    print(f"Sentences per language: {len(sampled_df)}")

    # 4. Initialize Google Translate Client
    client = translate.TranslationServiceClient.from_service_account_json(str(json_key))

    def translate_text(text, target_code):
        parent = f"projects/{project_id}/locations/global"
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": "en",
                "target_language_code": target_code,
            }
        )
        return response.translations[0].translated_text

    # 5. Translation Loop
    for language in test_langs:
        if language not in ALL_LANGS:
            print(f"Warning: {language} not in supported list. Skipping.")
            continue
            
        code = ALL_LANGS[language]
        is_gendered = language in GENDERED_LANGS
        save_file = output_path / f"{language.lower().replace(' ', '_')}.csv"

        print("save_file", save_file)
        

        if is_gendered:
            cols = ["GEST_sentence", "the man said", "the woman said", "original_stereotype"]
        else:
            cols = ["GEST_sentence", "translation", "original_stereotype"]

        df_out = pd.DataFrame(index=sampled_df.index, columns=cols)
        df_out["GEST_sentence"] = sampled_df["sentence"]
        df_out["original_stereotype"] = sampled_df["stereotype"]

        print(f"Translating for: {language}...")

        for idx, row in sampled_df.iterrows():
            sentence = row["sentence"]
            if is_gendered:
                df_out.at[idx, "the man said"] = translate_text(f"The man said, '{sentence}'", code)
                df_out.at[idx, "the woman said"] = translate_text(f"The woman said, '{sentence}'", code)
            else:
                df_out.at[idx, "translation"] = translate_text(sentence, code)
            
            # Save every iteration for safety (££)
            df_out.to_csv(save_file)

    print("All translations complete.")

if __name__ == "__main__":
    fire.Fire(main)