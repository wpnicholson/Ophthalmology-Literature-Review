import pandas as pd


def get_data(full_list: list[str], field_dict: dict[str, str], df_orig: pd.DataFrame) -> pd.DataFrame:
    lenX: int = len(full_list)

    i = 0
    previous_content = ''

    # Build a dictionary for each entry.
    # Keys are the field names (the keys from 'field_dict') and values are the content.
    current_entry = {}

    while i < lenX:
        # Get a row of input text.
        field = full_list[i]

        # Empty rows are preserved as NaN values (floats).
        # Empty rows demarcate the end of an entry (one research paper).
        if pd.isna(field) or i == lenX-1:
            field = ''
            new_entry = True
        else:
            field = field[0:4].strip()  # The longest possible field name is 4 characters long.
            new_entry = False

        if i+1 < lenX:
            # Assuming you're not at the end of the file, make the next row of input text available.
            field_next = full_list[i+1]
            if pd.isna(field_next):
                field_next = ''
            else:
                field_next = field_next[0:4].strip()  # The longest possible field name is 4 characters long.
        else:
            field_next = ''

        if field in field_dict.values() and field_next in field_dict.values():
            # If the current row contains a field name and the next row also contains a field name.
            # Typically an entry begins with the 'PMID' field name. In which case simply grab the content
            # from this row.

            # The actual interesting content is stored after the field name.
            content = full_list[i][5:].strip()

            # The field name in the 'field' variable should match with one of the keys in the 'field_dict' dictionary.
            # Retrieve the key from the 'field_dict' dictionary that matches the value in the 'field' variable.
            # I haven't used a try/except block here because the if clause above ensures that the field name
            # in the 'field' variable will always match with one of the values in the 'field_dict' dictionary.
            temp = list(field_dict.keys())[list(field_dict.values()).index(field)]
            current_entry[temp] = content.strip()

        elif field in field_dict.values() and field_next not in field_dict.values():
            # If the current row contains a field name but the next row does not contain a field name.
            # Either because the next row is the empty row sitting between two entries or because the
            # next row contains the actual content (e.g. a very long abstract split over multiple rows).

            # The actual interesting content is stored after the field name.
            previous_content = full_list[i][5:].strip()
            field_previous = field

            # Indicates that the next row is the end of the file.
            if i+1 == lenX:
                temp = list(field_dict.keys())[list(field_dict.values()).index(field_previous)]
                current_entry[temp] = previous_content.strip()

        elif field not in field_dict.values() and field_next in field_dict.values():
            # If the current row does not contain a field name but the next row does contain a field name.
            # Either because the current row is the empty row sitting between two entries or because the
            # current row contains the actual content (e.g. a very long abstract split over multiple rows)
            # and the next row contains a field name.

            if pd.isna(full_list[i]):
                # If the current row is the empty row sitting between two entries.
                content = ''
            else:
                # If the current row contains the actual content (e.g. a very long abstract split over multiple rows).
                content = full_list[i].strip()

            content = previous_content + " " + content

            temp = list(field_dict.keys())[list(field_dict.values()).index(field_previous)]
            current_entry[temp] = content.strip()

        else:
            # If the current row does not contain a field name and the next row also does not contain a field name.
            # Can only be if both the current row and the next row both contain the actual content (e.g. a very
            # long abstract split over multiple rows).
            content = full_list[i].strip()
            content = previous_content + " " + content
            previous_content = content

        if new_entry:
            th = pd.DataFrame.from_dict(current_entry, orient='index').transpose()
            df_orig = pd.concat([df_orig, th], ignore_index=True)
            # Start a new dictionary for the next entry.
            current_entry = {}

        i += 1

    return df_orig
