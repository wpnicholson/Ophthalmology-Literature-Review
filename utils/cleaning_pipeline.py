import re
import pandas as pd


def to_lowercase(cols: list[str], df_in: pd.DataFrame) -> pd.DataFrame:
    """Converts the text in the specified columns, of the provided DataFrame, to lowercase.

    Creates a new column by appending "_lowercase" to the column name.

    Args:
        cols (list[str]): list of columns to convert to lowercase.
        df_in (pd.DataFrame): Pandas DataFrame containing the columns to convert.

    Returns:
        pd.DataFrame: the original DataFrame with the specified columns converted to lowercase.
    """
    for col in cols:
        df_in[col + '_lowercase'] = df_in[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
    return df_in


def _replace_abbreviations(text: str, replacement_dict: dict[str, str]) -> str:
    """Replaces abbreviations in the provided text with their full form.

    Args:
        text (str): string of text to replace abbreviations in.
        replacement_dict (dict[str, str]): dictionary of abbreviations and their full form.

    Returns:
        str: the original text with abbreviations replaced.
    """
    for key, value in replacement_dict.items():

        if key.lower() == 'on':
            # Pattern to match 'ON' in uppercase only.
            pattern = re.compile(r'\bON\b')
            text = pattern.sub(value, text)  # Direct replacement without group references.

        elif key == 'al':
            # Special case handling for 'al' in 'et al.'
            # Pattern for 'al' that excludes 'et al.' and does not match 'al' as part of another word.
            # Negative lookbehind to exclude 'al' in 'et al.'
            pattern = re.compile(r'(?i)(?<!et\s)\bal\b')
            text = pattern.sub(value, text)  # Direct replacement without group references.

        else:
            # Regular expression patterns for other cases.
            # Note that the pattern '(?i)' is an inline flag for re.IGNORECASE - thereby making
            # the pattern case-insensitive.
            pattern = re.compile(
                r'(?i)(\s)' + re.escape(key) + r'\b(?=[.,;-])|' +  # key (with leading space) before punctuation, capture leading space.
                r'\b' + re.escape(key) + r'\b(\s)|' +              # key surrounded by whitespace, capture trailing space.
                r'^' + re.escape(key) + r'(\s)\b|' +               # key at the beginning of the sentence (with trailing space), capture trailing space.
                r'\(' + re.escape(key) + r'\)|'                    # key surrounded by parentheses.
                r'\[' + re.escape(key) + r'\]'                     # key surrounded by square brackets.
            )
            # Replace with the full form and the captured whitespace/punctuation.
            text = pattern.sub(r'\1' + value + r'\2', text)

    return text


def replace_abbreviations(cols: list[str], df_in: pd.DataFrame, replacement_dict: dict[str, str]) -> pd.DataFrame:
    """Replaces abbreviations in the specified columns, of the provided DataFrame, with their full form.

    Creates a new column by appending "_abbv" to the column name.

    Args:
        cols (list[str]): list of columns to replace abbreviations in.
        df_in (pd.DataFrame): Pandas DataFrame containing the columns to replace abbreviations in.
        replacement_dict (dict[str, str]): dictionary of abbreviations and their full form.

    Returns:
        pd.DataFrame: the original DataFrame with the specified columns having their abbreviations replaced.
    """
    for col in cols:
        # Concatenate a new empty column onto the existing DataFrame.
        # The new empty column should be specifically of data type pd.StringDtype().
        df_in = pd.concat([df_in, pd.DataFrame(columns=[col + "_abbv"], dtype=pd.StringDtype())], axis=1)

        df_in[col + '_abbv'] = df_in[col].apply(
            lambda x: _replace_abbreviations(x, replacement_dict) if isinstance(x, str) else x
        )

        # Reinforce the data type of the new column as pd.StringDtype().
        df_in = df_in.astype({col + '_abbv': pd.StringDtype()})
    return df_in


def _remove_duplicates(text: str, replacement_dict: dict[str, str]) -> str:
    """Removes consecutive duplicates of the specified phrases in the provided text.

    Args:
        text (str): string of text to remove duplicates from.
        replacement_dict (dict[str, str]): dictionary of phrases to remove duplicates of.

    Returns:
        str: the original text with duplicates removed.
    """
    for phrase in replacement_dict.values():
        # Create a regex pattern to match consecutive duplicates of the phrase.
        # This pattern uses \b for word boundaries and \s* for optional spaces.
        pattern = re.compile(r'(' + re.escape(phrase) + r')\s*\1', re.IGNORECASE)

        # Replace matched duplicates with a single instance of the phrase.
        text = pattern.sub(r'\1', text)

    return text


def remove_duplicates(cols: list[str], df_in: pd.DataFrame, replacement_dict: dict[str, str]) -> pd.DataFrame:
    """Removes consecutive duplicates of the specified phrases in the provided DataFrame.

    Does not create a new column i.e. the column(s) input in the function signature are overwritten.

    Args:
        cols (list[str]): list of columns to remove duplicates from.
        df_in (pd.DataFrame): Pandas DataFrame containing the columns to remove duplicates from.
        replacement_dict (dict[str, str]): dictionary of phrases to remove duplicates of.

    Returns:
        pd.DataFrame: the original DataFrame with the specified columns having their duplicates removed.
    """
    for col in cols:
        df_in[col] = df_in[col].apply(
            lambda x: _remove_duplicates(x, replacement_dict) if isinstance(x, str) else x
        )

        # Reinforce the data type of the new column as pd.StringDtype().
        df_in = df_in.astype({col: pd.StringDtype()})
    return df_in


def _normalize(text: str, nlp) -> str:
    """Normalizes the provided text.

    Args:
        text (str): string of text to normalize.
        nlp (_type_): spaCy NLP object.

    Returns:
        str: the original text normalized.
    """
    # Regarding 'p_a' ... this pattern breaks down as follows:
    #    `-`: Hyphen character.
    #    `a-z`: Lowercase letters (a to z).
    #    `A-Z`: Uppercase letters (A to Z).
    #    `'`: Apostrophe character.
    #    ` `: Space character.
    #    `[^...]`: The caret `^` inside the square brackets denotes a negated set, which means the
    #              pattern will match any character that is not listed in the brackets.
    # This pattern will keep English letters (both uppercase and lowercase), hyphens, apostrophes, and
    # spaces, and replace all other characters with spaces (as per the function's logic).
    # Remember, this pattern is designed for standard English text. If your text includes other
    # characters that are meaningful in your context you might need to include those in the pattern as well.
    p_hyphen = re.compile(r'--')  # Pattern to match two hyphens.
    p_a = re.compile(r"[^-a-zA-Z' ]")
    p_b = re.compile(r'(\s{2,})')

    # Replace two hyphens with a single space.
    text = p_hyphen.sub(' ', text)

    # Apply the cleaning steps to a single sentence.
    text = re.sub(p_a, '  ', text)  # Keep allowed characters - replacing them with a double space.
    text = re.sub(p_b, ' ', text)   # Replace multiple spaces (1st pass).
    text = text.strip()             # Trim whitespace.
    text = re.sub(p_b, ' ', text)   # Replace multiple spaces (2nd pass).

    # spaCy processing for stopword removal and lemmatization.
    doc = nlp(text)
    lemmatized_sentence = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

    return lemmatized_sentence


def normalize(cols: list[str], df_in: pd.DataFrame, nlp) -> pd.DataFrame:
    """Normalizes the text in the specified columns, of the provided DataFrame.

    Creates a new column by replacing "_lowercase_abbv" with "_normalized" in the column name.

    Args:
        cols (list[str]): list of columns to normalize.
        df_in (pd.DataFrame): Pandas DataFrame containing the columns to normalize.
        nlp (_type_): spaCy NLP object.

    Returns:
        pd.DataFrame: the original DataFrame with the specified columns normalized.
    """
    for col in cols:
        df_in[col.replace('_lowercase_abbv', '_normalized')] = df_in[col].apply(
            lambda x: _normalize(x, nlp) if isinstance(x, str) else x
        )
    return df_in


def _correct_sentence_splitting(sentences: list[str]) -> list[str]:
    """Corrects the splitting of sentences that have been split incorrectly.

    Checks if a sentence is a single word and the following sentence starts with a comma, then
    merges them accordingly.

    Args:
        sentences (list[str]): list of sentences to correct.

    Returns:
        list[str]: list of corrected sentences.
    """
    corrected_sentences = []
    i = 0

    while i < len(sentences) - 1:
        current_sentence = sentences[i].strip()
        next_sentence = sentences[i + 1].strip()

        # Check if the current sentence is a single word and the next sentence starts with a comma.
        if current_sentence.count(' ') == 0 and next_sentence.startswith(','):
            # Merge current and next sentence, removing the comma.
            merged_sentence = current_sentence + next_sentence
            corrected_sentences.append(merged_sentence)

            # Skip the next sentence as it is already merged.
            i += 2

        else:
            corrected_sentences.append(current_sentence)
            i += 1

    # Add the last sentence if it wasn't processed.
    if i < len(sentences):
        corrected_sentences.append(sentences[i].strip())

    return corrected_sentences


def _split_into_sentences(text: str, nlp) -> list[str]:
    """Splits the provided text into sentences.

    Use a library like NLTK or spaCy, both of which have sophisticated sentence tokenization that
    have advanced sentence boundary detection algorithms and so can handle many edge cases you might
    not think of if you were to write your own regex.

    Args:
        text (str): string of text to split into sentences.
        nlp (_type_): spaCy NLP object.

    Returns:
        list[str]: list of sentences.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    corrected_sentences = _correct_sentence_splitting(sentences)
    return corrected_sentences


def split_into_sentences(cols: list[str], df_in: pd.DataFrame, nlp) -> pd.DataFrame:
    """Splits the text in the specified columns, of the provided DataFrame, into sentences.

    Creates a new column by appending "_split" to the column name.
    This new column will contain a list of sentences.

    Args:
        cols (list[str]): list of columns to split into sentences.
        df_in (pd.DataFrame): Pandas DataFrame containing the columns to split into sentences.
        nlp (_type_): spaCy NLP object.

    Returns:
        pd.DataFrame: the original DataFrame with the specified columns split into sentences.
    """
    for col in cols:
        df_in[col + '_split'] = df_in[col].apply(
            lambda x: _split_into_sentences(x, nlp) if isinstance(x, str) else x
        )
    return df_in


def _remove_uppercase_colon_phrases(text: str) -> str:
    """Removes uppercase phrases, and their associated colon character, when they start a sentence.

    Removes uppercase words or phrases at the beginning of a sentence, which are immediately
    followed by a colon character. Then it strips any leading whitespace

    Args:
        text (str): string of text to remove uppercase phrases from.

    Returns:
        str: the original text with uppercase phrases removed.
    """
    # Pattern to match uppercase words or phrases at the start of a sentence, followed by a colon.
    # Includes commas and spaces in the phrase.
    # Assumes these phrases are uppercase and directly followed by a colon.
    pattern = re.compile(r'^[A-Z\s,]+:')

    # Replace matched patterns. If the entire string is a match, it becomes empty.
    # Otherwise, it removes the match and strips leading whitespace from the remaining string.
    modified_text = re.sub(pattern, '', text).lstrip()

    return modified_text


def remove_uppercase_colon_phrases(cols: list[str], df_in: pd.DataFrame) -> pd.DataFrame:
    """Removes uppercase phrases between a period and a colon, or at the start of the string before a colon.

    Does not create a new column i.e. the column(s) input in the function signature are overwritten.

    Args:
        cols (list[str]): list of columns to remove uppercase phrases from.
        df_in (pd.DataFrame): Pandas DataFrame containing the columns to remove uppercase phrases from.

    Returns:
        pd.DataFrame: the original DataFrame with the specified columns having their uppercase phrases removed.
    """
    for col in cols:
        df_in[col] = df_in[col].apply(
            lambda x: _remove_uppercase_colon_phrases(x) if isinstance(x, str) else x
        )
    return df_in


def _whitespace(text: str) -> str:
    """Normalizes the provided text for whitespace.

    Remove any whitespaces that are of size two or more, and replace them with a single space.
    Remove any hyphens that are of size two or more, and replace them with a single space.
    Strip any leading or trailing whitespace.

    Args:
        text (str): string of text to normalize.

    Returns:
        str: the original text normalized for whitespace.
    """
    # Pattern to match two hyphens.
    p_hyphen = re.compile(r'--')

    # Pattern to match multiple whitespaces.
    p_b = re.compile(r'(\s{2,})')

    # Apply the cleaning steps to a single sentence.
    text = p_hyphen.sub(' ', text)  # Replace two hyphens with a single space.
    text = re.sub(p_b, ' ', text)   # Replace multiple whitespaces with a single whitespace.
    text = text.strip()             # Trim whitespace.

    return text


def whitespace(cols: list[str], df_in: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the whitespace in the specified column(s), of the provided DataFrame.

    Does not create a new column i.e. the column(s) input in the function signature are overwritten.

    Args:
        cols (list[str]): list of columns to normalize.
        df_in (pd.DataFrame): Pandas DataFrame containing the columns to normalize.

    Returns:
        pd.DataFrame: the original DataFrame with the specified columns normalized for whitespace.
    """
    for col in cols:
        df_in[col] = df_in[col].apply(
            lambda x: _whitespace(x) if isinstance(x, str) else x
        )
    return df_in
