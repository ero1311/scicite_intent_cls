import re

def clean_text(x):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', x)
    return text


def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)

    return x