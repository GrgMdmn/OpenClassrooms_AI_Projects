import nltk
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator
from googletrans import Translator
from concurrent.futures import ThreadPoolExecutor, as_completed

def select_key_words(
    text,
    keep_between_separators=True,
    regex_pattern=r'[a-zA-Z\u00C0-\u017F]+',  # Default regex pattern: keeps words with letters and accents
    stopword_languages=['english'],  # List of languages for stopwords
    separator=",",  # Default separator
    keep_duplicates = False # Return unique list
):
    if pd.isna(text):
        return np.nan
    else:
        # Compile the custom regex pattern
        word_pattern = re.compile(regex_pattern, re.UNICODE)

        # Build the stop words set based on the provided languages
        stop_words = set()
        for language in stopword_languages:
            try:
                stop_words.update(stopwords.words(language))
            except OSError:
                print(f"Warning: Could not load stopwords for language '{language}'. "
                      f"Ensure the NLTK stopword corpus is downloaded using 'nltk.download('stopwords')'.")

        # Mode 1: Keep elements separated by the custom separator as single keywords
        if keep_between_separators:
            # Split the text using the provided separator while keeping the segments
            segments = re.split(rf'({re.escape(separator)})', text.lower())  # Split using custom separator
            segments = [seg.strip() for seg in segments if seg.strip() and seg != separator]  # Clean empty segments
            
            # Apply regex to extract words and clean each segment
            filtered_segments = [
                ' '.join(word_pattern.findall(seg))
                for seg in segments
            ]

            if keep_duplicates:
                # Remove duplicates while preserving order
                seen = set()
                ordered_segments = [seg for seg in filtered_segments if seg not in seen and not seen.add(seg) and len(seg)>0]
            else:
                ordered_segments = [seg for seg in filtered_segments if len(seg)>0]

            return ordered_segments

        # Mode 2: Extract individual words (ignoring the separator)
        else:
            # Extract words using the regex pattern
            keywords = word_pattern.findall(text.lower())
            
            if not keep_duplicates:
                # Remove stop words while preserving the order and remove duplicates
                seen = set()
                filtered_words = [word for word in keywords if word not in stop_words and word not in seen and not seen.add(word) and len(word)>0]
            else:
                filtered_words = [word for word in keywords if word not in stop_words and len(word)>0]


            return filtered_words

# Function to count occurrences of the first ingredient in a string
def count_words_occurrences(text, keep_duplicates=False):
    if pd.isna(text):
      result = np.nan
    else:
      result = {}
      for word in select_key_words(text, keep_duplicates=keep_duplicates):
        count = text.lower().count(word.lower())
        result[word] = count
    return result
    

def translate_word(word, source_language="auto", target_language="en", translator_lib='deep_translator'):
    """
    Traduction d'un mot en utilisant soit `deep_translator`, soit `googletrans`.
    Args:
        word (str): Texte à traduire
        source_language (str): Langue source (auto par défaut)
        target_language (str): Langue cible (en par défaut)
        translator_lib (str): Utilise 'deep_translator' ou 'googletrans'.
    
    Returns:
        dict: 'original' contient le mot d'origine, 'translated' la traduction ou None en cas d'erreur.
    """
    try:
        translated = None
        error_in_text = None
        
        if translator_lib == 'deep_translator':
            # Traduction avec deep_translator
            translated = GoogleTranslator(source=source_language, target=target_language).translate(word)
            # Vérification si le résultat contient des erreurs textuelles inattendues
            # Adapté pour détecter les messages comme "Error 500..." ou similaires
            if "Error" in translated or "error" in translated.lower():
                error_in_text = f"Erreur détectée dans la réponse traduite : '{translated}'"
                raise ValueError(error_in_text)
        
        elif translator_lib == 'googletrans':
            # Traduction avec googletrans
            from googletrans import Translator
            translator = Translator()
            result = translator.translate(word, src=source_language, dest=target_language)
            translated = result.text.lower()
        
        else:
            raise ValueError("Bibliothèque non supportée. Utilisez 'deep_translator' ou 'googletrans'.")
        
        return {"original": word, "translated": translated}
    
    except Exception as e:
        if e==error_in_text:
          raise ValueError(error_in_text)
        # Gestion explicite des erreurs avec diagnostic
        else:
          raise ValueError(f"Other Error: {e}")
        # return {"original": word, "translated": None, "error": str(e)}




def translate_all_words_parallel(words, source_language="auto", target_language="en", max_workers=5, translator_lib="deep_translator"):
    """
    Translates a list of words in parallel, returning a dictionary where 
    the key is the original word, and the value is the translated word.

    Args:
        words (list): The list of words to be translated.
        target_language (str): Target language for translation (default is English "en").
        max_workers (int): Maximum number of threads for parallel processing.

    Returns:
        dict: A dictionary with original words as keys and translations as values.
    """
    translations = {}  # Dictionary to store the translations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit parallel tasks for translation
        future_to_word = {executor.submit(translate_word, word, source_language, target_language, translator_lib): word for word in words}

        print(future_to_word)
        # Process the completed tasks
        for future in as_completed(future_to_word):
            result = future.result()  # Get the result of the translation
            if result.get("translated"):  # If translation is successful
                translations[result["original"]] = result["translated"]
            else:
                # If there's an error, store None or the error message as the translation
                translations[result["original"]] = None
    
    return translations  # Return all translations
    
    
def translate_all_words_batch(words, source_language="auto", target_language="en"):
    """
    Translates a list of words in a batch, returning a dictionary where the key
    is the original word, and the value is the translated word.
    This function uses the translate_batch function of deep_translator for better performance
    on large datasets.
    
    Args:
        words (list): The list of words to be translated.
        source_language (str): Source language for translation (default is "auto").
        target_language (str): Target language for translation (default is English "en").
    
    Returns:
        dict: A dictionary with original words as keys and translations as values.
    """
    translations = {}
    
    try:
        # Use GoogleTranslator's translate_batch method from deep_translator
        translated_batch = GoogleTranslator(source=source_language, target=target_language).translate_batch(words)

        # Match original words with their translations
        for original, translated in zip(words, translated_batch):
            translations[original] = translated

    except Exception as e:
        # Handle exceptions and assign None to the words that failed to translate
        for word in words:
            translations[word] = None
        print(f"An error occurred during batch translation: {e}")
    
    return translations

# # other way to translate
# from googletrans import Translator
# import asyncio

# async def translate_word_async(word, src_language='auto', target_language='en'):
#     """
#     Asynchronously translates a single word or phrase using googletrans.
#     """
#     translator = Translator()
#     try:
#         # Await translation for the specific word
#         translation = await translator.translate(word, src=src_language, dest=target_language)
#         return translation.text
#     except Exception as e:
#         print(f"Error while translating '{word}': {e}")
#         return word  # Return the original word in case of an error

# async def translate_all_words_async(words, src_language='auto', target_language='en'):
#     """
#     Asynchronously translates a list of words as fast as possible.
#     """
#     # Create a list of translation tasks
#     tasks = [translate_word_async(word, src_language, target_language) for word in words]
    
#     # Run all tasks concurrently and wait for their completion
#     results = await asyncio.gather(*tasks)
#     return results