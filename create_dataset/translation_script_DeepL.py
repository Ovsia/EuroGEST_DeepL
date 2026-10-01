#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import time
import random

import pandas as pd
import fire
import deepl

# ============================================================= #
# ================== CONFIGURATION ============================= #
# ============================================================= #

GENDERED_LANGS = {
    'Bulgarian': 'BG', 'French': 'FR', 'German': 'DE', 'Greek': 'EL',
    'Italian': 'IT', 'Latvian': 'LV', 'Lithuanian': 'LT', 'Maltese': 'MT',
    'Portuguese': 'PT', 'Romanian': 'RO', 'Spanish': 'ES', 'Catalan': 'CA',
    'Galician': 'GL', 'Croatian': 'HR', 'Czech': 'CS', 'Polish': 'PL',
    'Slovak': 'SK', 'Slovenian': 'SL', 'Russian': 'RU', 'Ukrainian': 'UK'
}

NEUTRAL_LANGS = {
    'Danish': 'DA', 'Dutch': 'NL', 'Estonian': 'ET', 'Finnish': 'FI',
    'Hungarian': 'HU', 'Irish': 'GA', 'Swedish': 'SV',
    'Norwegian': 'NB', 'Turkish': 'TR', 'Chinese': 'ZH', 'Indonesian': 'ID'
}

ALL_LANGS = {**GENDERED_LANGS, **NEUTRAL_LANGS}

# ============================================================= #
# ======================= MAIN FUNCTION ======================== #
# ============================================================= #

def main(
    data_path,
    json_key=None,
    project_id=None,
    out_dir='.',
    sample_size=1,
    languages=None,
    auth_key=None
):
    """
    Translate English GEST sentences into multiple European languages using DeepL.
    """

    api_key = auth_key or os.getenv("DEEPL_AUTH_KEY") or json_key

    if not api_key:
        raise ValueError(
            "DeepL API key is required. Pass --auth_key or set the "
            "DEEPL_AUTH_KEY environment variable."
        )

    translator = deepl.Translator(api_key)

    # ========================================================= #
    # ===================== CONFIGURATION ====================== #
    # ========================================================= #

    MAX_RETRIES = 8
    INITIAL_BACKOFF = 2
    MAX_BACKOFF = 60

    # Delay between successful API requests.
    # Increase this if DeepL still returns 429 errors.
    REQUEST_DELAY = 0.5

    # ========================================================= #
    # ======================== PATHS =========================== #
    # ========================================================= #

    data_file = Path(data_path)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========================================================= #
    # ===================== LOAD DATASET ======================= #
    # ========================================================= #

    gest_df = pd.read_csv(data_file)

    # ========================================================= #
    # ================= DETERMINE LANGUAGES =================== #
    # ========================================================= #

    if languages:
        if isinstance(languages, str):
            languages = [l.strip() for l in languages.split(',')]

        normalized_langs = []
        lookup = {name.lower(): name for name in ALL_LANGS}

        for language in languages:
            resolved = lookup.get(language.strip().lower())

            if resolved is not None:
                normalized_langs.append(resolved)
            else:
                print(
                    f"Warning: {language} not in supported list. Skipping."
                )

        test_langs = normalized_langs

        sampled_df = (
            gest_df
            if sample_size == 1
            else gest_df.sample(n=sample_size)
        )

    elif sample_size != 1:
        test_langs = [
            random.choice(list(GENDERED_LANGS.keys())),
            random.choice(list(NEUTRAL_LANGS.keys()))
        ]

        sampled_df = gest_df.sample(n=sample_size)

    else:
        test_langs = list(ALL_LANGS.keys())
        sampled_df = gest_df

    print(f"Processing languages: {test_langs}")
    print(f"Sentences per language: {len(sampled_df)}")

    # ========================================================= #
    # ==================== TRANSLATION ========================= #
    # ========================================================= #

    def translate_text(text, target_code):
        for attempt in range(MAX_RETRIES):
            try:
                result = translator.translate_text(
                    text=text,
                    target_lang=target_code,
                    source_lang="EN",
                )

                time.sleep(REQUEST_DELAY)

                return result.text

            except deepl.TooManyRequestsException as e:
                if attempt == MAX_RETRIES - 1:
                    raise

                backoff = min(
                    INITIAL_BACKOFF * (2 ** attempt),
                    MAX_BACKOFF
                )

                jitter = random.uniform(0, 1)

                wait_time = backoff + jitter

                print(
                    f"DeepL rate limit reached. "
                    f"Retry {attempt + 1}/{MAX_RETRIES} "
                    f"in {wait_time:.1f}s..."
                )

                time.sleep(wait_time)

            except deepl.DeepLException as e:
                print(f"DeepL error: {e}")

                if attempt == MAX_RETRIES - 1:
                    raise

                wait_time = min(
                    INITIAL_BACKOFF * (2 ** attempt),
                    MAX_BACKOFF
                )

                print(
                    f"Retry {attempt + 1}/{MAX_RETRIES} "
                    f"in {wait_time}s..."
                )

                time.sleep(wait_time)

    # ========================================================= #
    # =================== TRANSLATION LOOP ===================== #
    # ========================================================= #

    for language in test_langs:

        if language not in ALL_LANGS:
            print(
                f"Warning: {language} not in supported list. Skipping."
            )
            continue

        code = ALL_LANGS[language]
        is_gendered = language in GENDERED_LANGS

        save_file = (
            output_path /
            f"{language.lower().replace(' ', '_')}.csv"
        )

        print(f"\nSave file: {save_file}")

        if is_gendered:
            cols = [
                "GEST_sentence",
                "the man said",
                "the woman said",
                "original_stereotype"
            ]
        else:
            cols = [
                "GEST_sentence",
                "translation",
                "original_stereotype"
            ]

        # ===================================================== #
        # =============== LOAD EXISTING RESULTS =============== #
        # ===================================================== #

        if save_file.exists():
            print(f"Existing file found: {save_file}")
            print("Resuming previous translation...")

            df_out = pd.read_csv(
                save_file,
                index_col=0
            )

            # Make sure required columns exist
            for col in cols:
                if col not in df_out.columns:
                    df_out[col] = None

        else:
            df_out = pd.DataFrame(
                index=sampled_df.index,
                columns=cols
            )

            df_out["GEST_sentence"] = sampled_df["sentence"]
            df_out["original_stereotype"] = sampled_df["stereotype"]

        print(f"Translating for: {language}...")

        # ===================================================== #
        # ================= TRANSLATE ROWS ===================== #
        # ===================================================== #

        for idx, row in sampled_df.iterrows():

            sentence = row["sentence"]

            if is_gendered:

                # -------------------------------------------------
                # Man
                # -------------------------------------------------

                if (
                    pd.isna(df_out.at[idx, "the man said"])
                    or df_out.at[idx, "the man said"] == ""
                ):

                    text = f"The man said, '{sentence}'"

                    print(
                        f"[{language}] "
                        f"{idx}: translating man..."
                    )

                    df_out.at[idx, "the man said"] = translate_text(
                        text,
                        code
                    )

                    df_out.to_csv(save_file)

                # -------------------------------------------------
                # Woman
                # -------------------------------------------------

                if (
                    pd.isna(df_out.at[idx, "the woman said"])
                    or df_out.at[idx, "the woman said"] == ""
                ):

                    text = f"The woman said, '{sentence}'"

                    print(
                        f"[{language}] "
                        f"{idx}: translating woman..."
                    )

                    df_out.at[idx, "the woman said"] = translate_text(
                        text,
                        code
                    )

                    df_out.to_csv(save_file)

            else:

                if (
                    pd.isna(df_out.at[idx, "translation"])
                    or df_out.at[idx, "translation"] == ""
                ):

                    print(
                        f"[{language}] "
                        f"{idx}: translating..."
                    )

                    df_out.at[idx, "translation"] = translate_text(
                        sentence,
                        code
                    )

                    df_out.to_csv(save_file)

        print(f"Finished: {language}")

    print("\nAll translations complete.")


if __name__ == "__main__":
    fire.Fire(main)